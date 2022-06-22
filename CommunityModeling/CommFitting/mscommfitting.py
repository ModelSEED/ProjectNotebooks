# -*- coding: utf-8 -*-
# cython: language_level=3
from pandas import read_table, read_csv, DataFrame
from optlang import Variable, Constraint, Objective, Model
from modelseedpy.core.fbahelper import FBAHelper
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from sympy.core.add import Add
# from pprint import pprint
from time import sleep, process_time
from warnings import warn
import numpy as np
# from cplex import Cplex
import cython
import os

def _variable_name(name, suffix, trial, time):
    return '-'.join([name+suffix, trial, time])

def _constraint_name(name, suffix, trial, time):
    return '_'.join([name+suffix, trial, time])

@cython.cclass
class MSCommFitting():   # explicit typing for cython
    parameters: dict = {}; variables: dict = {}; constraints: dict = {}; dataframes: dict = {}; signal_species: dict = {}; 
    phenotypes_parsed_df: np.ndarray; problem: object; species_phenotypes_bool_df: object; zipped_output:list
        
    def __init__(self, 
                 community_members: dict = {},         # the model of the community that was experimentally investigated and will be examined via fitting, which includes the permanent KBase ID of the media (e.g. 93465/3/1) that describe each respective community model
                 kbase_token: str = None,              # the KBase user token that must be provided to access permanent_media_id
                 solver:str = 'glpk',                  # the LP solver that will optimize the community model in the given media
                 signal_tsv_paths: dict = {},          # the dictionary of index names for each paths to signal TSV data that will be fitted
                 phenotypes_csv_path: str = None,      # a custom CSV of media phenotypic data
                 ):
        if phenotypes_csv_path:
            # process a predefined exchanges table
            fluxes_df = read_csv(phenotypes_csv_path)
            fluxes_df.index = fluxes_df['rxn']
            to_drop = [col for col in fluxes_df.columns if ' ' in col]
            for col in to_drop+['rxn']:
                fluxes_df.drop(col, axis=1, inplace=True)
            print(f'The {to_drop+["rxn"]} columns were dropped from the phenotypes CSV.')
        elif community_members:
            import cobrakbase
            if kbase_token:
                kbase = cobrakbase.KBaseAPI(kbase_token)
            else:
                kbase = cobrakbase.KBaseAPI()
            fluxes_df = self._assemble_fluxes(community_members, kbase, solver)
            
        fluxes_df.astype(str)
        self.phenotypes_parsed_df = FBAHelper.parse_df(fluxes_df)
        
        self.species_phenotypes_bool_df = DataFrame(columns=self.phenotypes_parsed_df[1])
        for path, name in signal_tsv_paths.items():
            self.zipped_output.append(path)
            signal = os.path.splitext(path)[0].split("_")[0]
            # define the signal dataframe
            self.signal_species[signal] = name # {name:strains}
            self.dataframes[signal] = read_table(path).iloc[1::2]  # !!! is this the proper slice of data, or should the other set of values at each well/time be used?
            self.dataframes[signal].index = self.dataframes[signal]['Well']
            for col in ['Plate', 'Cycle', 'Well']:
                self.dataframes[signal].drop(col, axis=1, inplace=True)
            self.dataframes[signal].astype(str)
            # convert the dataframe to a numpy array for greater efficiency
            self.dataframes[signal]: np.ndarray = FBAHelper.parse_df(self.dataframes[signal])
        
            # differentiate the phenotypes for each species
            self.parameters[signal]: dict = {}
            if "OD" not in signal:
                self.species_phenotypes_bool_df.loc[signal]: np.ndarray[cython.int] = np.array([
                    1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_parsed_df[1]])
                
    def _assemble_fluxes(self, community_members, kbase, solver):
        # import the media for each model
        models = OrderedDict(); ex_rxns:set = set(); species:list = []; strains:list = []
        for model, content in community_members.items():
            if isinstance(model, str):
                # load the model
                if 'json' in model:
                    from cobra.io import load_json_model as load_model
                elif 'xml' in model:
                    from cobra.io import read_sbml_model as load_model
                elif 'mat' in model:
                    from cobra.io import load_matlab_model as load_model
                model = load_model(model)
            model.medium = kbase.get_from_ws(content['permanent_media_id'])  # how can KBase media be converted into dictionaries?
            model.solver = solver
            
            # execute the model strains
            model.objective = Objective(
                model.bio1.flux_expression, direction='max')  # !!! is not maximizing the biomass the default objective? Will the biomass reaction always be bio1?
            ex_rxns.update(model.exchanges)
            species.append(content['name'])
            strains.extend(content['strains'])
            models[model] = []
            for rxn_id in content['strains']:
                with model:
                    for rxn in model.reactions:
                        if rxn.id == rxn_id:
                            rxn.upper_bound = rxn.lower_bound = -1
                            break
                    models[model].append(model.optimize())
                
        # construct the parsed table of all exchange fluxes for each strain
        fluxes_df = DataFrame(data={'bio':[sol.fluxes['bio1'] for model, sol in models.items()]}, 
                              columns=['rxn']+[spec+'-'+strain for spec in species for strain in strains])
        fluxes_df.index.name = 'rxn'
        for ex_rxn in ex_rxns:
            elements = []
            for model, sol in models.items():
                flux = 0
                if ex_rxn in model.reactions:
                    flux = sol.fluxes[rxn]  # if sol.fluxes[rxn] != 0 else 0
                elements.append(flux)
            if any([np.array(elements) != 0]):
                fluxes_df.iloc[ex_rxn.id] = elements
                
        return fluxes_df
                
    @cython.ccall # cfunc
    def define_problem(self, conversion_rates={'cvt': 0, 'cvf': 0}, 
                       constraints={'cvc': {}},
                       print_conditions: bool = True, print_lp: bool = True, zip_contents:bool = True
                       ):
        self.problem = Model()
        self.parameters.update({
            "timestep_s": 600,      # Size of time step used in simulation in seconds
            "cvct": 1,              # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf": 1,              # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv": 1,               # This is the highest fraction of biomass for a given strain that can change phenotypes in a single time step
            "cvmin": 0.1,           # This is the lowest value the limit on phenotype conversion goes, 
            "v": 1,                 # the kinetics constant that is externally adjusted 
        })
        self.parameters.update(conversion_rates)
        trial: str; time: str; name: str; strain: str; met: str;
        obj_coef:dict = {}; constraints: list = []; variables: list = []  # lists are orders-of-magnitude faster than numpy arrays for appending
        
        # 1.1E4 loops => 1 minute
        time_1 = process_time()
        for signal, parsed_df in self.dataframes.items():  # 3 loops
            for strain in self.phenotypes_parsed_df[1]:  # 6
                self.constraints['dbc_'+signal+'_'+strain]:dict = {}
                for trial in parsed_df[0]:  # 66
                    self.constraints['dbc_'+signal+'_'+strain][trial]:dict = {}
                    
        for strain in self.phenotypes_parsed_df[1]:  # 6 
            self.variables['cvt_'+strain]:dict = {}; self.variables['cvf_'+strain]:dict = {}
            self.variables['b_'+strain]:dict = {}; self.variables['g_'+strain]:dict = {}
            self.variables['v_'+strain]:dict = {}; self.variables['b+1_'+strain]:dict = {}
            self.constraints['gc_'+strain]:dict = {}; self.constraints['cvc_'+strain]:dict = {}
            for trial in parsed_df[0]:  # 66 ; the use of only one parsed_df prevents duplicative variable assignment
                self.variables['cvt_'+strain][trial]:dict = {}; self.variables['cvf_'+strain][trial]:dict = {}
                self.variables['b_'+strain][trial]:dict = {}; self.variables['g_'+strain][trial]:dict = {}
                self.variables['v_'+strain][trial]:dict = {}; self.variables['b+1_'+strain][trial]:dict = {}
                self.constraints['gc_'+strain][trial]:dict = {}; self.constraints['cvc_'+strain][trial]:dict = {}
                last_column = False
                for time in parsed_df[1]:  # 27
                    next_time = str(int(time)+1)
                    if next_time == len(parsed_df[1]):
                        last_column = True  
                    self.variables['b_'+strain][trial][time] = Variable(         # predicted biomass abundance
                        _variable_name("b_", strain, trial, time), lb=0, ub=1000)  
                    self.variables['g_'+strain][trial][time] = Variable(         # biomass growth
                        _variable_name("g_", strain, trial, time), lb=0, ub=1000)   
                    self.variables['cvt_'+strain][trial][time] = Variable(       # conversion rate to the stationary phase
                        _variable_name("cvt_", strain, trial, time), lb=0, ub=100)   
                    self.variables['cvf_'+strain][trial][time] = Variable(       # conversion from to the stationary phase
                        _variable_name("cvf_", strain, trial, time), lb=0, ub=100)                   
                    self.variables['b+1_'+strain][trial][next_time] = Variable(  # predicted biomass abundance
                        _variable_name("b+1_", strain, trial, next_time), lb=0, ub=1000) 
                    
                    # g_{strain} - b_{strain}*v = 0
                    self.constraints['gc_'+strain][trial][time] = Constraint(
                        self.variables['g_'+strain][trial][time] 
                        - self.parameters['v']*self.variables['b_'+strain][trial][time],
                        _constraint_name("gc_", strain, trial, time), lb=0, ub=0)
                    
                    # 0 <= -cvt + bcv*b_{strain} + cvmin
                    self.constraints['cvc_'+strain][trial][time] = Constraint(
                        -self.variables['cvt_'+strain][trial][time] 
                        + self.parameters['bcv']*self.variables['b_'+strain][trial][time] + self.parameters['cvmin'],
                        _constraint_name('cvc_', strain, trial, time), lb=0, ub=None)                        
                    
                    obj_coef.update({self.variables['cvt_'+strain][trial][time]: self.parameters['cvct'], 
                                     self.variables['cvf_'+strain][trial][time]: self.parameters['cvcf']})
                    variables.extend([self.variables['b+1_'+strain][trial][next_time],
                        self.variables['b_'+strain][trial][time], self.variables['g_'+strain][trial][time],
                        self.variables['cvt_'+strain][trial][time], self.variables['cvf_'+strain][trial][time]])
                    constraints.extend([self.constraints['gc_'+strain][trial][time], 
                                        self.constraints['cvc_'+strain][trial][time]])

                    if last_column:
                        break
                if last_column:
                    break
                    
        # define non-concentration variables
        # 4.3E4 total loops => 6 minutes
        time_2 = process_time()
        print(f'Done with predictory loops: {(time_2-time_1)/60} min')
        for parsed_df in self.dataframes.values():  # 1 (with the break)
            for r_index, met in enumerate(self.phenotypes_parsed_df[0]):  # 25
                self.variables["c_"+met]:dict = {}; self.variables["c+1_"+met]:dict = {}
                self.constraints['dcc_'+met]:dict = {}
                for trial in parsed_df[0]:  # 66
                    self.variables["c_"+met][trial]:dict = {}; self.variables["c+1_"+met][trial]:dict = {}
                    self.constraints['dcc_'+met][trial]:dict = {}
                    for time in parsed_df[1]:  # 27      
                        next_time = str(int(time)+1)
                        # define biomass measurement conversion variables 
                        self.variables["c_"+met][trial][time] = Variable(
                            _variable_name("c_", met, trial, time), lb=0, ub=1000)    
                        self.variables["c+1_"+met][trial][next_time] = Variable(
                            _variable_name("c+1_", met, trial, next_time), lb=0, ub=1000)    
                            
                        # c_{met} + dt*sum_k^K() - c+1_{met} = 0
                        self.constraints['dcc_'+met][trial][time] = Constraint(self.variables["c_"+met][trial][time] 
                                         - self.variables["c+1_"+met][trial][next_time] + np.dot(
                                             self.phenotypes_parsed_df[2][r_index]*self.parameters['timestep_s'], np.array([
                                                 self.variables['g_'+strain][trial][time] for strain in self.phenotypes_parsed_df[1]
                                                 ])), 
                                         ub=0, lb=0, name=_constraint_name("dcc_", met, trial, time))
                        
                        variables.extend([self.variables["c_"+met][trial][time], self.variables["c+1_"+met][trial][next_time]])
                        constraints.append(self.constraints['dcc_'+met][trial][time])
            break   # prevents duplicated construction of variables and constraints
                    
        # 6.4E4 loops => 2 minutes
        time_3 = process_time()
        print(f'Done with metabolites loops: {(time_3-time_2)/60} min')
        for signal, parsed_df in self.dataframes.items():  # 3
            conversion = Variable(signal+'__conversion', lb=0, ub=1000)
            self.variables[signal+'__bio']:dict = {}; self.variables[signal+'__diffpos']:dict = {}
            self.variables[signal+'__diffneg']:dict = {}
            self.variables[signal+'__conversion'] = conversion
            self.constraints[signal+'__bioc']:dict = {}; self.constraints[signal+'__diffc']:dict = {}  # diffc is defined latter
            for r_index, trial in enumerate(parsed_df[0]):  # 66
                self.variables[signal+'__bio'][trial]:dict = {}; self.variables[signal+'__diffpos'][trial]:dict = {}
                self.variables[signal+'__diffneg'][trial]:dict = {}
                self.constraints[signal+'__bioc'][trial]:dict = {}; self.constraints[signal+'__diffc'][trial]:dict = {}
                for time in parsed_df[1]:  # 27
                    next_time = str(int(time)+1)
                    total_biomass: Add = 0; signal_sum: Add = 0; from_sum: Add = 0; to_sum: Add = 0
                    for strain in self.phenotypes_parsed_df[1]:  # 6
                        total_biomass += self.variables["b_"+strain][trial][time]
                        if 'OD' not in signal:  # the OD strain has a different constraint
                            val = self.species_phenotypes_bool_df.loc[signal, strain]
                            signal_sum += val*self.variables["b_"+strain][trial][time]
                            from_sum += val*self.variables['cvf_'+strain][trial][time]
                            to_sum += val*self.variables['cvt_'+strain][trial][time]
                    for strain in self.phenotypes_parsed_df[1]:  # 6
                        if "stationary" in strain:
                            # b_{strain} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) - b+1_{strain} = 0
                            self.constraints['dbc_'+signal+'_'+strain][trial][time] = Constraint(
                                self.variables['b_'+strain][trial][time] - from_sum + to_sum 
                                - self.variables['b+1_'+strain][trial][next_time],
                                ub=0, lb=0, name=_constraint_name("dbc_", signal+'_'+strain, trial, time))
                        else:
                            # -b_{strain} + dt*g_{strain} + cvf - cvt - b+1_{strain} = 0
                            self.constraints['dbc_'+signal+'_'+strain][trial][time] = Constraint(self.variables['b_'+strain][trial][time]
                                + self.parameters['timestep_s']*self.variables['g_'+strain][trial][time]
                                + self.variables['cvf_'+strain][trial][time] - self.variables['cvt_'+strain][trial][time]
                                - self.variables['b+1_'+strain][trial][next_time],
                                ub=0, lb=0, name=_constraint_name("dbc_", signal+'_'+strain, trial, time))
                            
                    self.variables[signal+'__bio'][trial][time] = Variable(_variable_name(signal, '__bio', trial, time), lb=0, ub=1000)
                    self.variables[signal+'__diffpos'][trial][time] = Variable( 
                        _variable_name(signal, '__diffpos', trial, time), lb=-100, ub=100) 
                    self.variables[signal+'__diffneg'][trial][time] = Variable(  
                        _variable_name(signal, '__diffneg', trial, time), lb=-100, ub=100) 
                        
                    # {signal}__conversion*datum = {signal}__bio
                    self.constraints[signal+'__bioc'][trial][time] = Constraint(
                        conversion*parsed_df[2][r_index, int(time)-1] - self.variables[signal+'__bio'][trial][time], 
                        name=_constraint_name(signal, '__bioc', trial, time), lb=0, ub=0)
                    
                    # {speces}_bio - sum_k^K(es_k*b_{strain}) - {signal}_diffpos + {signal}_diffpos = 0
                    self.constraints[signal+'__diffc'][trial][time] = Constraint( 
                        self.variables[signal+'__bio'][trial][time]-signal_sum 
                        - self.variables[signal+'__diffpos'][trial][time]
                        + self.variables[signal+'__diffneg'][trial][time], 
                        name=_constraint_name(signal, '__diffc', trial, time), lb=0, ub=0)

                    obj_coef.update({self.variables[signal+'__diffpos'][trial][time]:1,
                                     self.variables[signal+'__diffneg'][trial][time]:-1})                            
                    variables.extend([conversion, self.variables[signal+'__bio'][trial][time], 
                                      self.variables[signal+'__diffpos'][trial][time],
                                      self.variables[signal+'__diffneg'][trial][time]])
                    constraints.extend([self.constraints[signal+'__bioc'][trial][time], 
                                        self.constraints[signal+'__diffc'][trial][time],
                                        self.constraints['dbc_'+signal+'_'+strain][trial][time]])
                            
        time_4 = process_time()
        print(f'Done with the dbc & diffc loop: {(time_4-time_3)/60} min')
        # construct the problem
        self.problem.add(set(variables))
        self.problem.update()
        self.problem.add(set(constraints))
        self.problem.update()
        self.problem.objective = Objective(Zero, direction="min") #, sloppy=True)
        time_5 = process_time()
        print(f'Done with loading the variables, constraints, and objective: {(time_5-time_4)/60} min')
        self.problem.objective.set_linear_coefficients(obj_coef)
                
        # print contents
        zipped_output = []
        if print_conditions:
            DataFrame(self.parameters).to_csv('parameters.csv')
            zipped_output.append('parameters.csv')
        if print_lp:
            zipped_output.append('mscommfitting.lp')
            with open('mscommfitting.lp', 'w') as lp:
                lp.write(self.problem.to_lp())
        if zip_contents:
            sleep(2)
            with ZipFile('msComFit.zip', 'w', compression=ZIP_LZMA) as zp:
                for file in zipped_output:
                    zp.write(file)
                    os.remove(file)
                
    def compute(self,):
        solution = self.problem.optimize()
        if "optimal" in  solution:
            print('The problem optimized optimally.')
        else:
            warn(f'The problem did not optimize optimally, with a {solution} status.')
             
        