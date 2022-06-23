# -*- coding: utf-8 -*-
# cython: language_level=3
from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from pandas import read_table, read_csv, DataFrame
from optlang import Variable, Constraint, Objective, Model
from modelseedpy.core.fbahelper import FBAHelper
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from sympy.core.add import Add
from matplotlib import pyplot
# from pprint import pprint
from time import sleep, process_time
from warnings import warn
import numpy as np
# from cplex import Cplex
# import cython
import os

def _variable_name(name, suffix, trial, time):
    return '-'.join([name+suffix, trial, time])

def _constraint_name(name, suffix, trial, time):
    return '_'.join([name+suffix, trial, time])

# @cython.cclass
class MSCommFitting():   # explicit typing for cython
    parameters: dict = {}; variables: dict = {}; constraints: dict = {}; dataframes: dict = {}; signal_species: dict = {}; values:dict = {}
    phenotypes_parsed_df: np.ndarray; problem: object; species_phenotypes_bool_df: object; zipped_output:list = []; plots:list = []
        
    def __init__(self, 
                 community_members: dict = {},         # the model of the community that was experimentally investigated and will be examined via fitting, which includes the permanent KBase ID of the media (e.g. 93465/3/1) that describe each respective community model
                 kbase_token: str = None,              # the KBase user token that must be provided to access permanent_media_id
                 solver:str = 'glpk',                  # the LP solver that will optimize the community model in the given media
                 signal_tsv_paths: dict = {},          # the dictionary of index names for each paths to signal TSV data that will be fitted
                 phenotypes_csv_path: str = None,      # a custom CSV of media phenotypic data
                 zipped_contents:bool = False          # specifies whether the input contents are in a zipped file
                 ):
        if zipped_contents:
            with ZipFile('msComFit.zip', 'r') as zp:
                zp.extractall()
        if phenotypes_csv_path:
            # process a predefined exchanges table
            self.zipped_output.append(phenotypes_csv_path)
            fluxes_df = read_csv(phenotypes_csv_path)
            fluxes_df.index = fluxes_df['rxn']
            to_drop = [col for col in fluxes_df.columns if ' ' in col]
            for col in to_drop+['rxn']:
                fluxes_df.drop(col, axis=1, inplace=True)
            print(f'The {to_drop+["rxn"]} columns were dropped from the phenotypes CSV.')
        elif community_members:
            # import the media for each model
            models = OrderedDict(); ex_rxns:set = set(); species:dict = {}
            #Using KBase media to constrain exchange reactions in model
            for model, content in community_members.items():
                model.solver = solver
                ex_rxns.update(model.exchanges)
                species.update({content['name']: content['strains'].keys()})
                models[model] = []
                for media in content['strains'].values():
                    with model:  # !!! Is this the correct method of parameterizing a media for a model?
                        pkgmgr = MSPackageManager.get_pkg_mgr(model)
                        pkgmgr.getpkg("KBaseMediaPkg").build_package(media, default_uptake=0, default_excretion=1000)
                        models[model].append(model.optimize())
                    
            # construct the parsed table of all exchange fluxes for each strain
            fluxes_df = DataFrame(data={'bio':[sol.fluxes['bio1'] for solutions in models.values() for sol in solutions]},
                                  columns=['rxn']+[spec+'-'+strain for spec, strains in species.items() for strain in strains])
            fluxes_df.index.name = 'rxn'
            for ex_rxn in ex_rxns:
                elements = []
                for model, solutions in models.items():
                    for sol in solutions:
                        elements.append(sol.fluxes[ex_rxn] if ex_rxn in sol.fluxes else 0)
                if any(np.array(elements) != 0):
                    fluxes_df.iloc[ex_rxn.id] = elements
            
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
            if "OD" not in signal:
                self.species_phenotypes_bool_df.loc[signal]: np.ndarray[int] = np.array([
                    1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_parsed_df[1]])
                
    # @cython.ccall # cfunc
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
                for time in parsed_df[1]:  # 27
                    self.constraints['dbc_'+signal+'_'+strain][time]:dict = {}
                    
        for strain in self.phenotypes_parsed_df[1]:  # 6 
            self.variables['cvt_'+strain]:dict = {}; self.variables['cvf_'+strain]:dict = {}
            self.variables['b_'+strain]:dict = {}; self.variables['g_'+strain]:dict = {}
            self.variables['v_'+strain]:dict = {}; self.variables['b+1_'+strain]:dict = {}
            self.constraints['gc_'+strain]:dict = {}; self.constraints['cvc_'+strain]:dict = {}
            for time in parsed_df[1]:  # 27
                self.variables['cvt_'+strain][time]:dict = {}; self.variables['cvf_'+strain][time]:dict = {}
                self.variables['b_'+strain][time]:dict = {}; self.variables['g_'+strain][time]:dict = {}
                self.variables['v_'+strain][time]:dict = {}; self.variables['b+1_'+strain][time]:dict = {}
                self.constraints['gc_'+strain][time]:dict = {}; self.constraints['cvc_'+strain][time]:dict = {}
                last_column = False
                for trial in parsed_df[0]:  # 66 ; the use of only one parsed_df prevents duplicative variable assignment
                    next_time = str(int(time)+1)
                    if next_time == len(parsed_df[1]):
                        last_column = True  
                    self.variables['b_'+strain][time][trial] = Variable(         # predicted biomass abundance
                        _variable_name("b_", strain, trial, time), lb=0, ub=1000)  
                    self.variables['g_'+strain][time][trial] = Variable(         # biomass growth
                        _variable_name("g_", strain, trial, time), lb=0, ub=1000)   
                    self.variables['cvt_'+strain][time][trial] = Variable(       # conversion rate to the stationary phase
                        _variable_name("cvt_", strain, trial, time), lb=0, ub=100)   
                    self.variables['cvf_'+strain][time][trial] = Variable(       # conversion from to the stationary phase
                        _variable_name("cvf_", strain, trial, time), lb=0, ub=100)                   
                    self.variables['b+1_'+strain][time][next_time] = Variable(  # predicted biomass abundance
                        _variable_name("b+1_", strain, trial, next_time), lb=0, ub=1000) 
                    
                    # g_{strain} - b_{strain}*v = 0
                    self.constraints['gc_'+strain][time][trial] = Constraint(
                        self.variables['g_'+strain][time][trial] 
                        - self.parameters['v']*self.variables['b_'+strain][time][trial],
                        _constraint_name("gc_", strain, trial, time), lb=0, ub=0)
                    
                    # 0 <= -cvt + bcv*b_{strain} + cvmin
                    self.constraints['cvc_'+strain][time][trial] = Constraint(
                        -self.variables['cvt_'+strain][time][trial] 
                        + self.parameters['bcv']*self.variables['b_'+strain][time][trial] + self.parameters['cvmin'],
                        _constraint_name('cvc_', strain, trial, time), lb=0, ub=None)                        
                    
                    obj_coef.update({self.variables['cvt_'+strain][time][trial]: self.parameters['cvct'], 
                                     self.variables['cvf_'+strain][time][trial]: self.parameters['cvcf']})
                    variables.extend([self.variables['b+1_'+strain][time][next_time],
                        self.variables['b_'+strain][time][trial], self.variables['g_'+strain][time][trial],
                        self.variables['cvt_'+strain][time][trial], self.variables['cvf_'+strain][time][trial]])
                    constraints.extend([self.constraints['gc_'+strain][time][trial], 
                                        self.constraints['cvc_'+strain][time][trial]])

                    if last_column:
                        break
                if last_column:
                    break
                    
        # define non-concentration variables
        # 4.3E4 total loops => 6 minutes
        time_2 = process_time()
        print(f'Done with biomass loop: {(time_2-time_1)/60} min')
        for parsed_df in self.dataframes.values():  # 1 (with the break)
            for r_index, met in enumerate(self.phenotypes_parsed_df[0]):  # 25
                self.variables["c_"+met]:dict = {}; self.variables["c+1_"+met]:dict = {}
                self.constraints['dcc_'+met]:dict = {}
                for time in parsed_df[1]:  # 27  
                    self.variables["c_"+met][time]:dict = {}; self.variables["c+1_"+met][time]:dict = {}
                    self.constraints['dcc_'+met][time]:dict = {}
                    for trial in parsed_df[0]:  # 66    
                        next_time = str(int(time)+1)
                        # define biomass measurement conversion variables 
                        self.variables["c_"+met][time][trial] = Variable(
                            _variable_name("c_", met, trial, time), lb=0, ub=1000)    
                        self.variables["c+1_"+met][time][next_time] = Variable(
                            _variable_name("c+1_", met, trial, next_time), lb=0, ub=1000)    
                            
                        # c_{met} + dt*sum_k^K() - c+1_{met} = 0
                        self.constraints['dcc_'+met][time][trial] = Constraint(self.variables["c_"+met][time][trial] 
                                         - self.variables["c+1_"+met][time][next_time] + np.dot(
                                             self.phenotypes_parsed_df[2][r_index]*self.parameters['timestep_s'], np.array([
                                                 self.variables['g_'+strain][time][trial] for strain in self.phenotypes_parsed_df[1]
                                                 ])), 
                                         ub=0, lb=0, name=_constraint_name("dcc_", met, trial, time))
                        
                        variables.extend([self.variables["c_"+met][time][trial], self.variables["c+1_"+met][time][next_time]])
                        constraints.append(self.constraints['dcc_'+met][time][trial])
            break   # prevents duplicated construction of variables and constraints
                    
        # 6.4E4 loops => 2 minutes
        time_3 = process_time()
        print(f'Done with metabolites loop: {(time_3-time_2)/60} min')
        for signal, parsed_df in self.dataframes.items():  # 3
            self.variables[signal+'__conversion'] = Variable(signal+'__conversion', lb=0, ub=1000)
            variables.append(self.variables[signal+'__conversion'])
            
            self.variables[signal+'__bio']:dict = {}; self.variables[signal+'__diffpos']:dict = {}
            self.variables[signal+'__diffneg']:dict = {}
            self.constraints[signal+'__bioc']:dict = {}; self.constraints[signal+'__diffc']:dict = {}  # diffc is defined latter
            for time in parsed_df[1]:  # 27
                self.variables[signal+'__bio'][time]:dict = {}; self.variables[signal+'__diffpos'][time]:dict = {}
                self.variables[signal+'__diffneg'][time]:dict = {}
                self.constraints[signal+'__bioc'][time]:dict = {}; self.constraints[signal+'__diffc'][time]:dict = {}
                for r_index, trial in enumerate(parsed_df[0]):  # 66
                    next_time = str(int(time)+1)
                    total_biomass: Add = 0; signal_sum: Add = 0; from_sum: Add = 0; to_sum: Add = 0
                    for strain in self.phenotypes_parsed_df[1]:  # 6
                        total_biomass += self.variables["b_"+strain][time][trial]
                        if 'OD' not in signal:  # the OD strain has a different constraint
                            val = self.species_phenotypes_bool_df.loc[signal, strain]
                            signal_sum += val*self.variables["b_"+strain][time][trial]
                            from_sum += val*self.variables['cvf_'+strain][time][trial]
                            to_sum += val*self.variables['cvt_'+strain][time][trial]
                    for strain in self.phenotypes_parsed_df[1]:  # 6
                        if "stationary" in strain:
                            # b_{strain} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) - b+1_{strain} = 0
                            self.constraints['dbc_'+signal+'_'+strain][time][trial] = Constraint(
                                self.variables['b_'+strain][time][trial] - from_sum + to_sum 
                                - self.variables['b+1_'+strain][time][next_time],
                                ub=0, lb=0, name=_constraint_name("dbc_", signal+'_'+strain, trial, time))
                        else:
                            # -b_{strain} + dt*g_{strain} + cvf - cvt - b+1_{strain} = 0
                            self.constraints['dbc_'+signal+'_'+strain][time][trial] = Constraint(self.variables['b_'+strain][time][trial]
                                + self.parameters['timestep_s']*self.variables['g_'+strain][time][trial]
                                + self.variables['cvf_'+strain][time][trial] - self.variables['cvt_'+strain][time][trial]
                                - self.variables['b+1_'+strain][time][next_time],
                                ub=0, lb=0, name=_constraint_name("dbc_", signal+'_'+strain, trial, time))
                            
                    self.variables[signal+'__bio'][time][trial] = Variable(_variable_name(signal, '__bio', trial, time), lb=0, ub=1000)
                    self.variables[signal+'__diffpos'][time][trial] = Variable( 
                        _variable_name(signal, '__diffpos', trial, time), lb=-100, ub=100) 
                    self.variables[signal+'__diffneg'][time][trial] = Variable(  
                        _variable_name(signal, '__diffneg', trial, time), lb=-100, ub=100) 
                        
                    # {signal}__conversion*datum = {signal}__bio
                    self.constraints[signal+'__bioc'][trial][time] = Constraint(
                        self.variables[signal+'__conversion']*parsed_df[2][r_index, int(time)-1] 
                        - self.variables[signal+'__bio'][trial][time], 
                        name=_constraint_name(signal, '__bioc', trial, time), lb=0, ub=0)
                    
                    # {speces}_bio - sum_k^K(es_k*b_{strain}) - {signal}_diffpos + {signal}_diffpos = 0
                    self.constraints[signal+'__diffc'][time][trial] = Constraint( 
                        self.variables[signal+'__bio'][time][trial]-signal_sum 
                        - self.variables[signal+'__diffpos'][time][trial]
                        + self.variables[signal+'__diffneg'][time][trial], 
                        name=_constraint_name(signal, '__diffc', trial, time), lb=0, ub=0)

                    obj_coef.update({self.variables[signal+'__diffpos'][trial][time]:1,
                                     self.variables[signal+'__diffneg'][trial][time]:-1})                            
                    variables.extend([self.variables[signal+'__bio'][trial][time], 
                                      self.variables[signal+'__diffpos'][trial][time],
                                      self.variables[signal+'__diffneg'][trial][time]])
                    constraints.extend([self.constraints[signal+'__bioc'][trial][time], 
                                        self.constraints[signal+'__diffc'][trial][time],
                                        self.constraints['dbc_'+signal+'_'+strain][trial][time]])
                            
        time_4 = process_time()
        print(f'Done with the dbc & diffc loop: {(time_4-time_3)/60} min')
        # construct the problem
        self.problem.add(variables)
        self.problem.update()
        self.problem.add(constraints)
        self.problem.update()
        self.problem.objective = Objective(Zero, direction="min") #, sloppy=True)
        self.problem.objective.set_linear_coefficients(obj_coef)
        time_5 = process_time()
        print(f'Done with loading the variables, constraints, and objective: {(time_5-time_4)/60} min')
                
        # print contents
        if print_conditions:
            DataFrame(data=list(self.parameters.values()), 
                      index=list(self.parameters.keys()), 
                      columns=['values']).to_csv('parameters.csv')
            self.zipped_output.append('parameters.csv')
        if print_lp:
            self.zipped_output.append('mscommfitting.lp')
            with open('mscommfitting.lp', 'w') as lp:
                lp.write(self.problem.to_lp())
        if zip_contents:
            sleep(2)
            with ZipFile('msComFit.zip', 'w', compression=ZIP_LZMA) as zp:
                for file in self.zipped_output:
                    zp.write(file)
                    os.remove(file)
                    
        time_6 = process_time()
        print(f'Done exporting the content: {(time_6-time_5)/60} min')
                
    def compute(self, graphs:list = []):
        solution = self.problem.optimize()
        if "optimal" in  solution:
            print('The solution is optimal.')
        else:
            warn(f'The solution is sub-optimal, with a {solution} status.')
        
        # categorize the primal values by trial and time
        for variable, value in self.problem.primal_values.items():
            if 'conversion' not in variable:
                basename, trial, time = variable.split('-')
                if not trial in self.values:
                    self.values[trial]:dict = {}
                if not basename in self.values[trial]:
                    self.values[trial][basename]:dict = {}
                self.values[trial][basename][time] = value
        
        # plot the content for desired trials 
        for graph in graphs:
            pyplot.rcParams['figure.figsize'] = (11, 7)
            pyplot.rcParams['figure.dpi'] = 150
            fig, ax = pyplot.subplots()
            for trial, basenames in self.values.items():
                content = graph['content']
                if graph['content'] == 'biomass':
                    content = 'b'
                if graph['content'] == 'growth':
                    content = 'g'   
                if trial == graph['trial']:
                    labels:list = []
                    for basename in basenames:
                        if all([x in basename for x in [graph['species'], graph['strain'], content]]):
                            labels.append(basename)
                            ax.plot(self.values[trial][basename].keys(), 
                                    self.values[trial][basename].values(),
                                    label=basename)
                            ax.legend(labels)
                            ax.set_xticks(list(x for x in self.values[trial][basename].keys() if int(x)%20 == 0))
                    if labels != []:
                        ax.set_xlabel('Time point')
                        ax.set_ylabel('Variable value')
                        ax.set_title(f'{graph["content"]} of the {graph["strain"]} {graph["species"]} strain in the {trial} trial')
                        fig_name = f'{"_".join([trial, graph["species"], graph["strain"], graph["content"]])}.png'
                        fig.savefig(fig_name)
                        self.plots.append(fig_name)
        
        # combine the figures with the other cotent
        with ZipFile('msComFit.zip', 'a') as zp:
            for plot in self.plots:
                zp.write(plot)
