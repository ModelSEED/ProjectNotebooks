# -*- coding: utf-8 -*-
# cython: language_level=3
from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from modelseedpy.core.exceptions import FeasibilityError
from pandas import read_table, read_csv, DataFrame
from optlang import Variable, Constraint, Objective, Model
from modelseedpy.core.fbahelper import FBAHelper
from scipy.constants import hour
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from sympy.core.add import Add
from matplotlib import pyplot
# from pprint import pprint
from time import sleep, process_time
import numpy as np
# from cplex import Cplex
# import cython
import json, os, re

def _variable_name(name, suffix, time, trial):
    return '-'.join([name+suffix, time, trial])

def _constraint_name(name, suffix, time, trial):
    return '_'.join([name+suffix, time, trial])

# @cython.cclass
class MSCommFitting():   # explicit typing for cython
    parameters: dict = {}; variables: dict = {}; constraints: dict = {}; dataframes: dict = {}; signal_species: dict = {}; values:dict = {}
    phenotypes_parsed_df: np.ndarray; problem: object; species_phenotypes_bool_df: object; zipped_output:list = []; plots:list = []; 
    simulation_time: float; data_timesteps: float
        
    def __init__(self, 
                 community_members: dict = {},         # the model of the community that was experimentally investigated and will be examined via fitting, which includes the permanent KBase ID of the media (e.g. 93465/3/1) that describe each respective community model
                 kbase_token: str = None,              # the KBase user token that must be provided to access permanent_media_id
                 solver:str = 'glpk',                  # the LP solver that will optimize the community model in the given media
                 signal_tsv_paths: dict = {},          # the dictionary of index names for each paths to signal TSV data that will be fitted
                 phenotypes_csv_path: str = None,      # a custom CSV of media phenotypic data
                 media_conc_path:str = None,           # a CSV of the media concentrations
                 species_abundance_path:str = None,    # a CSV over the series of species abundances for a range of trials
                 carbon_conc_series: dict = None,      # the concentrations of carbon sources in the media
                 ignore_trials:dict = {},              # the trials (row+column coordinates) that will be ignored by the model
                 ignore_timesteps:list=[],             # tiemsteps that will be ignored 
                 significant_deviation:float = 2,      # the lowest multiple of a trial mean from its initial value that will permit its inclusion in the model fit
                 zip_path:str = None             # specifies whether the input contents are in a zipped file
                 ):
        self.zipped_output = []
        if zip_path:
            with ZipFile(zip_path, 'r') as zp:
                zp.extractall()
        if species_abundance_path:
            self.species_abundances = self._process_csv(species_abundance_path, 'trial_column')
        if phenotypes_csv_path:
            # process a predefined exchanges table
            self.zipped_output.append(phenotypes_csv_path)
            fluxes_df = read_csv(phenotypes_csv_path)
            fluxes_df.index = fluxes_df['rxn']
            to_drop = [col for col in fluxes_df.columns if ' ' in col]
            for col in to_drop+['rxn']:
                fluxes_df.drop(col, axis=1, inplace=True)
            print(f'The {to_drop+["rxn"]} columns were dropped from the phenotypes CSV.')
            
            # import and process the media concentrations CSV
            self.media_conc = self._process_csv(media_conc_path, 'media_compound')
        elif community_members:
            # import the media for each model
            models = OrderedDict(); ex_rxns:set = set(); species:dict = {}
            #Using KBase media to constrain exchange reactions in model
            for model, content in community_members.items():
                model.solver = solver
                ex_rxns.update(model.exchanges)
                species.update({content['name']: content['phenotypes'].keys()})
                models[model] = []
                for media in content['phenotypes'].values():
                    with model:  # !!! Is this the correct method of parameterizing a media for a model?
                        pkgmgr = MSPackageManager.get_pkg_mgr(model)
                        pkgmgr.getpkg("KBaseMediaPkg").build_package(media, default_uptake=0, default_excretion=1000)
                        models[model].append(model.optimize())
                    
            # construct the parsed table of all exchange fluxes for each phenotype
            fluxes_df = DataFrame(
                data={'bio':[sol.fluxes['bio1'] for solutions in models.values() for sol in solutions]},
                columns=['rxn']+[spec+'-'+phenotype for spec, phenotypes in species.items() for phenotype in phenotypes]
                +[spec+'-stationary' for spec in species.keys()])
            fluxes_df.index.name = 'rxn'
            fluxes_df.drop('rxn', axis=1, inplace=True)
            for ex_rxn in ex_rxns:
                elements = []
                for model, solutions in models.items():
                    for sol in solutions:
                        elements.append(sol.fluxes[ex_rxn] if ex_rxn in sol.fluxes else 0)
                if any(np.array(elements) != 0):
                    fluxes_df.iloc[ex_rxn.id] = elements
            
        # define only species for which data is defined
        signal_tsv_paths
        modeled_species = list(signal_tsv_paths.values()); modeled_species.remove('OD')
        removed_phenotypes = [col for col in fluxes_df if not any([species in col for species in modeled_species])]
        for col in removed_phenotypes:
            fluxes_df.drop(col, axis=1, inplace=True)
        if removed_phenotypes != []:
            print(f'The {removed_phenotypes} phenotypes were removed since their species is not among those that are defined with data: {modeled_species}.')
        fluxes_df.astype(str)
        self.phenotypes_parsed_df = FBAHelper.parse_df(fluxes_df)
        self.species_phenotypes_bool_df = DataFrame(columns=self.phenotypes_parsed_df[1])
        
        if 'columns' not in carbon_conc_series:
            carbon_conc_series['columns'] = {}
        if 'rows' not in ignore_trials:
            ignore_trials['rows'] = {}
        self.carbon_conc = carbon_conc_series
        
        self.parameters["data_timestep_hr"] = []
        if 'columns' not in ignore_trials:
            ignore_trials['columns'] = []
        if 'rows' not in ignore_trials:
            ignore_trials['rows'] = []
        if 'wells' not in ignore_trials:
            ignore_trials['wells'] = []
        ignore_trials['columns'] = list(map(str, ignore_trials['columns']))
        ignore_trials['rows'] = list(map(str, ignore_trials['rows']))
        ignore_timesteps = list(map(str, ignore_timesteps))
        for path, name in signal_tsv_paths.items():
            self.zipped_output.append(path)
            signal = os.path.splitext(path)[0].split("_")[0]
            # define the signal dataframe
            self.signal_species[signal] = name # {name:phenotypes}
            self.dataframes[signal] = read_table(path)
            self.simulation_time = self.dataframes[signal].iloc[0,-1]/hour
            self.parameters["data_timestep_hr"].append(self.simulation_time/int(self.dataframes[signal].columns[-1]))
            self.dataframes[signal] = self.dataframes[signal].iloc[1::2]  # excludes the times
            self.dataframes[signal].index = self.dataframes[signal]['Well']
            # filter data contents
            dropped_trials = []
            for trial in self.dataframes[signal].index:
                if any([trial[0] in ignore_trials['rows'], trial[1:] in ignore_trials['columns'],
                        trial in ignore_trials['wells']]):
                    self.dataframes[signal].drop(trial, axis=0, inplace=True)
                    dropped_trials.append(trial)
            if dropped_trials != []:
                print(f'The {dropped_trials} trials were dropped from the {name} measurements.')
            for col in ['Plate', 'Cycle', 'Well']:
                self.dataframes[signal].drop(col, axis=1, inplace=True)
            for col in self.dataframes[signal]:
                if col in ignore_timesteps:
                    self.dataframes[signal].drop(col, axis=1, inplace=True)
            if 'OD' not in signal:
                removed_trials = []
                for trial, row in self.dataframes[signal].iterrows():
                    row_array = np.array(row.to_list())
                    if row_array[-1]/row_array[0] < significant_deviation:
                        self.dataframes[signal].drop(trial, axis=0, inplace=True)
                        removed_trials.append(trial)
                if removed_trials != []:
                    print(f'The {removed_trials} trials were removed from the {name} measurements, with their deviation over time being less than the threshold of {significant_deviation}.')
            
            # process the data for subsequent operations and optimal efficiency
            self.dataframes[signal].astype(str)
            self.dataframes[signal]: np.ndarray = FBAHelper.parse_df(self.dataframes[signal])
            
            # differentiate the phenotypes for each species
            if "OD" not in signal:
                self.species_phenotypes_bool_df.loc[signal]: np.ndarray[int] = np.array([
                    1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_parsed_df[1]])
        
        self.parameters["data_timestep_hr"] = sum(self.parameters["data_timestep_hr"])/len(self.parameters["data_timestep_hr"])
        self.data_timesteps = int(self.simulation_time/self.parameters["data_timestep_hr"])
                
    def _process_csv(self, csv_path, index_col):
        self.zipped_output.append(csv_path)
        csv = read_csv(csv_path)
        csv.index = csv[index_col]
        csv.drop(index_col, axis=1, inplace=True)
        csv.astype(str)
        return csv
                
    # @cython.ccall # cfunc
    def define_problem(self, parameters={},        # parameters that will overwrite the default options
                       zip_name=None,              # the name of the export zip file
                       print_conditions: bool = True, print_lp: bool = True
                       ):
        self.parameters.update({
            "timestep_hr": self.parameters['data_timestep_hr'],  # Timestep size of the simulation in hours 
            "cvct": 1,                     # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf": 1,                     # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv": 1,                      # This is the highest fraction of biomass for a given species that can change phenotypes in a single time step
            "cvmin": 0,                    # This is the lowest value the limit on phenotype conversion goes, 
            "v": 1000,                     # the kinetics constant that is externally adjusted 
            'carbon_sources': ['cpd00136', 'cpd00179']  # 4hb, maltose
        })
        self.parameters.update(parameters)
        self.problem = Model()
        trial: str; time: str; name: str; phenotype: str; met: str
        obj_coef:dict = {}; constraints: list = []; variables: list = []  # lists are orders-of-magnitude faster than numpy arrays for appending
        self.simulation_timesteps = list(map(str, range(1, int(self.simulation_time/self.parameters['timestep_hr'])+1)))
        time_1 = process_time()
        for signal, parsed_df in self.dataframes.items():
            for met in self.phenotypes_parsed_df[0]:
                met_id = re.sub('(\_\w\d+)', '', met)
                met_id = met_id.replace('EX_', '', 1)
                if met_id != 'cpd00001':
                    self.variables["c_"+met]:dict = {}; self.constraints['dcc_'+met]:dict = {}
                    initial_time = True; final_time = False
                    for time in self.simulation_timesteps:
                        self.variables["c_"+met][time]:dict = {}; self.constraints['dcc_'+met][time]:dict = {}
                        if time == self.simulation_timesteps[-1]:
                            final_time = True
                        for trial in parsed_df[0]:
                            # define biomass measurement conversion variables 
                            self.variables["c_"+met][time][trial] = Variable(
                                _variable_name("c_", met, time, trial), lb=0, ub=1000)
                            # constrain initial time concentrations to the media or a large number if it is not explicitly defined
                            if initial_time and not 'bio' in met_id:
                                initial_val = self.media_conc.at[met_id,'mM'] if met_id in list(self.media_conc.index) else 100
                                if met_id in self.carbon_conc['rows'] and trial[0] in self.carbon_conc['rows'][met_id]:
                                    initial_val = self.carbon_conc['rows'][met_id][trial[0]]
                                if met_id in self.carbon_conc['columns'] and trial[1:] in self.carbon_conc['columns'][met_id]:
                                    initial_val = self.carbon_conc['columns'][met_id][trial[1:]]
                                self.variables["c_"+met][time][trial] = Variable(
                                    _variable_name("c_", met, time, trial), lb=initial_val, ub=initial_val)
                            # mandate complete carbon consumption
                            if final_time and met_id in self.parameters['carbon_sources']:
                                self.variables["c_"+met][time][trial] = Variable(
                                    _variable_name("c_", met, time, trial), lb=0, ub=0)
                            variables.append(self.variables["c_"+met][time][trial])
                        initial_time = False
            break   # prevents duplicated variables 
        for signal, parsed_df in self.dataframes.items():
            if 'OD' not in signal:
                for phenotype in self.phenotypes_parsed_df[1]:
                    if self.signal_species[signal] in phenotype:
                        self.constraints['dbc_'+phenotype]:dict = {}
                        for time in self.simulation_timesteps:
                            self.constraints['dbc_'+phenotype][time]:dict = {}
                  
        for phenotype in self.phenotypes_parsed_df[1]:
            self.variables['cvt_'+phenotype]:dict = {}; self.variables['cvf_'+phenotype]:dict = {}
            self.variables['b_'+phenotype]:dict = {}; self.variables['g_'+phenotype]:dict = {}
            self.variables['v_'+phenotype]:dict = {} 
            self.constraints['gc_'+phenotype]:dict = {}; self.constraints['cvc_'+phenotype]:dict = {}
            for time in self.simulation_timesteps:
                self.variables['cvt_'+phenotype][time]:dict = {}; self.variables['cvf_'+phenotype][time]:dict = {}
                self.variables['b_'+phenotype][time]:dict = {}; self.variables['g_'+phenotype][time]:dict = {}
                self.variables['v_'+phenotype][time]:dict = {}
                self.constraints['gc_'+phenotype][time]:dict = {}; self.constraints['cvc_'+phenotype][time]:dict = {}
                for trial in parsed_df[0]:
                    self.variables['b_'+phenotype][time][trial] = Variable(         # predicted biomass abundance
                        _variable_name("b_", phenotype, time, trial), lb=0, ub=100)
                    self.variables['g_'+phenotype][time][trial] = Variable(         # biomass growth
                        _variable_name("g_", phenotype, time, trial), lb=0, ub=1000)   
                    
                    if 'stationary' not in phenotype:
                        self.variables['cvt_'+phenotype][time][trial] = Variable(       # conversion rate to the stationary phase
                            _variable_name("cvt_", phenotype, time, trial), lb=0, ub=100)  
                        self.variables['cvf_'+phenotype][time][trial] = Variable(       # conversion from to the stationary phase
                            _variable_name("cvf_", phenotype, time, trial), lb=0, ub=100)   
                        
                        # 0 <= -cvt + bcv*b_{phenotype} + cvmin
                        self.constraints['cvc_'+phenotype][time][trial] = Constraint(
                            -self.variables['cvt_'+phenotype][time][trial] 
                            + self.parameters['bcv']*self.variables['b_'+phenotype][time][trial] + self.parameters['cvmin'],
                            lb=0, ub=None, name=_constraint_name('cvc_', phenotype, time, trial))  
                        
                        # g_{phenotype} - b_{phenotype}*v = 0
                        self.constraints['gc_'+phenotype][time][trial] = Constraint(
                            self.variables['g_'+phenotype][time][trial] 
                            - self.parameters['v']*self.variables['b_'+phenotype][time][trial],
                            lb=0, ub=0, name=_constraint_name("gc_", phenotype, time, trial))
                
                        obj_coef.update({self.variables['cvf_'+phenotype][time][trial]: self.parameters['cvcf'],
                                         self.variables['cvt_'+phenotype][time][trial]: self.parameters['cvct']})
                        variables.extend([self.variables['cvf_'+phenotype][time][trial], self.variables['cvt_'+phenotype][time][trial]])
                        constraints.extend([self.constraints['cvc_'+phenotype][time][trial], self.constraints['gc_'+phenotype][time][trial]])
                    
                    variables.extend([self.variables['b_'+phenotype][time][trial], self.variables['g_'+phenotype][time][trial]])
                    
        # define non-concentration variables  
        half_dt = self.parameters['data_timestep_hr']/2
        time_2 = process_time()
        print(f'Done with biomass loop: {(time_2-time_1)/60} min')
        for parsed_df in self.dataframes.values():
            for r_index, met in enumerate(self.phenotypes_parsed_df[0]):
                if 'cpd00001' not in met:
                    for trial in parsed_df[0]:
                        last_column = False
                        for time in self.simulation_timesteps:
                            next_time = str(int(time)+1)
                            if next_time == self.simulation_timesteps[-1]:
                                last_column = True  
                            # c_{met} + dt*sum_k^K() - c+1_{met} = 0
                            self.constraints['dcc_'+met][time][trial] = Constraint(
                                self.variables["c_"+met][time][trial] - self.variables["c_"+met][next_time][trial] 
                                + np.dot(
                                    self.phenotypes_parsed_df[2][r_index]*half_dt, np.array([
                                        self.variables['g_'+phenotype][time][trial]+self.variables['g_'+phenotype][next_time][trial]
                                        for phenotype in self.phenotypes_parsed_df[1]])),
                                ub=0, lb=0, name=_constraint_name("dcc_", met, time, trial))
                            
                            constraints.append(self.constraints['dcc_'+met][time][trial])
                            if last_column:
                                break
            break   # prevents duplicated constraints
                    
        time_3 = process_time()
        print(f'Done with metabolites loop: {(time_3-time_2)/60} min')
        for signal, parsed_df in self.dataframes.items():
            data_timestep = 1
            self.variables[signal+'__conversion'] = Variable(signal+'__conversion', lb=0, ub=1000)
            variables.append(self.variables[signal+'__conversion'])
            
            self.variables[signal+'__bio']:dict = {}; self.variables[signal+'__diffpos']:dict = {}
            self.variables[signal+'__diffneg']:dict = {}
            self.constraints[signal+'__bioc']:dict = {}; self.constraints[signal+'__diffc']:dict = {}  # diffc is defined latter
            for time in self.simulation_timesteps:
                if int(time)*self.parameters['timestep_hr'] >= data_timestep*self.parameters['data_timestep_hr']:  # synchronizes user timesteps with data timesteps
                    data_timestep += 1
                    if int(data_timestep) > self.data_timesteps:
                        break
                    next_time = str(int(time)+1)
                    self.variables[signal+'__bio'][time]:dict = {}; self.variables[signal+'__diffpos'][time]:dict = {}
                    self.variables[signal+'__diffneg'][time]:dict = {}
                    self.constraints[signal+'__bioc'][time]:dict = {}; self.constraints[signal+'__diffc'][time]:dict = {}
                    for r_index, trial in enumerate(parsed_df[0]):
                        total_biomass: Add = 0; signal_sum: Add = 0; from_sum: Add = 0; to_sum: Add = 0
                        for phenotype in self.phenotypes_parsed_df[1]:
                            total_biomass += self.variables["b_"+phenotype][time][trial]
                            val = 1 if 'OD' in signal else self.species_phenotypes_bool_df.loc[signal, phenotype]
                            signal_sum += val*self.variables["b_"+phenotype][time][trial]
                            if all(['OD' not in signal, self.signal_species[signal] in phenotype, 'stationary' not in phenotype]):
                                from_sum += val*self.variables['cvf_'+phenotype][time][trial] 
                                to_sum += val*self.variables['cvt_'+phenotype][time][trial]  
                        for phenotype in self.phenotypes_parsed_df[1]:
                            if 'OD' not in signal and self.signal_species[signal] in phenotype:
                                if "stationary" in phenotype:
                                    # b_{phenotype} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) - b+1_{phenotype} = 0
                                    self.constraints['dbc_'+phenotype][time][trial] = Constraint(
                                        self.variables['b_'+phenotype][time][trial] 
                                        - from_sum + to_sum 
                                        - self.variables['b_'+phenotype][next_time][trial],
                                        ub=0, lb=0, name=_constraint_name("dbc_", phenotype, time, trial))
                                else:
                                    # -b_{phenotype} + dt*g_{phenotype} + cvf - cvt - b+1_{phenotype} = 0
                                    self.constraints['dbc_'+phenotype][time][trial] = Constraint( 
                                        self.variables['b_'+phenotype][time][trial] - self.variables['b_'+phenotype][next_time][trial]
                                        + half_dt*(self.variables['g_'+phenotype][time][trial]+self.variables['g_'+phenotype][next_time][trial])
                                        + self.variables['cvf_'+phenotype][time][trial] - self.variables['cvt_'+phenotype][time][trial],
                                        ub=0, lb=0, name=_constraint_name("dbc_", phenotype, time, trial))
                                
                                constraints.append(self.constraints['dbc_'+phenotype][time][trial])
                                
                        self.variables[signal+'__bio'][time][trial] = Variable(
                            _variable_name(signal, '__bio', time, trial), lb=0, ub=1000)
                        self.variables[signal+'__diffpos'][time][trial] = Variable( 
                            _variable_name(signal, '__diffpos', time, trial), lb=0, ub=100) 
                        self.variables[signal+'__diffneg'][time][trial] = Variable(  
                            _variable_name(signal, '__diffneg', time, trial), lb=0, ub=100) 
                            
                        # {signal}__conversion*datum = {signal}__bio
                        self.constraints[signal+'__bioc'][time][trial] = Constraint(
                            self.variables[signal+'__conversion']*parsed_df[2][r_index, int(data_timestep)-1] 
                            - self.variables[signal+'__bio'][time][trial], 
                            name=_constraint_name(signal, '__bioc', time, trial), lb=0, ub=0)
                        
                        # {speces}_bio - sum_k^K(es_k*b_{phenotype}) - {signal}_diffpos + {signal}_diffneg = 0
                        self.constraints[signal+'__diffc'][time][trial] = Constraint( 
                            self.variables[signal+'__bio'][time][trial]-signal_sum 
                            - self.variables[signal+'__diffpos'][time][trial]
                            + self.variables[signal+'__diffneg'][time][trial], 
                            name=_constraint_name(signal, '__diffc', time, trial), lb=0, ub=0)
    
                        obj_coef.update({self.variables[signal+'__diffpos'][time][trial]:1,
                                         self.variables[signal+'__diffneg'][time][trial]:1})                            
                        variables.extend([self.variables[signal+'__bio'][time][trial], 
                                          self.variables[signal+'__diffpos'][time][trial],
                                          self.variables[signal+'__diffneg'][time][trial]])
                        constraints.extend([self.constraints[signal+'__bioc'][time][trial],
                                           self.constraints[signal+'__diffc'][time][trial]])
                
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
            self.zipped_output.append('parameters.csv')
            DataFrame(data=list(self.parameters.values()),
                      index=list(self.parameters.keys()), 
                      columns=['values']
                      ).to_csv('parameters.csv')
        if print_lp:
            self.zipped_output.append('mscommfitting.lp')
            with open('mscommfitting.lp', 'w') as lp:
                lp.write(self.problem.to_lp())
        if zip_name:
            self.zip_name = zip_name
            sleep(2)
            with ZipFile(self.zip_name, 'w', compression=ZIP_LZMA) as zp:
                for file in self.zipped_output:
                    zp.write(file)
                    os.remove(file)
                    
        time_6 = process_time()
        print(f'Done exporting the content: {(time_6-time_5)/60} min')
                
    def compute(self, graphs:list = [],     # the graph specifications that will be parsed from the primal values
                zip_name=None,              # the name of the export zip file
                ):
        solution = self.problem.optimize()
        
        # categorize the primal values by trial and time
        for variable, value in self.problem.primal_values.items():
            if 'conversion' not in variable:
                basename, time, trial = variable.split('-')
                time = int(time)*self.parameters['data_timestep_hr']
                if not trial in self.values:
                    self.values[trial]:dict = {}
                if not basename in self.values[trial]:
                    self.values[trial][basename]:dict = {}
                self.values[trial][basename][time] = value
                
        # export the processed primal values for graphing
        with open('primal_values.json', 'w') as out:
            json.dump(self.values, out, indent=3)
        if not zip_name:
            if hasattr(self, zip_name()):
                zip_name = self.zip_name
        if zip_name:
            with ZipFile(zip_name, 'a', compression=ZIP_LZMA) as zp:
                zp.write('primal_values.json')
                os.remove('primal_values.json')
        
        if graphs != []:
            self.graph(graphs, zip_name = zip_name)
            
        if "optimal" in  solution:
            print('The solution is optimal.')
        else:
            raise FeasibilityError(f'The solution is sub-optimal, with a {solution} status.')
                
    def graph(self, graphs:list = [], 
              primal_values_filename = None,                  # the name of the primal value JSON file ('primal_values.json')
              primal_values_zip_path = None,                  # the path of the zip file that contains the primal values file
              zip_name=None,                                  # the name of the zip file to which images will be exported
              ):
        def add_plot(ax, labels, basename, trial):
            labels.append(basename)
            ax.plot(self.values[trial][basename].keys(),
                    self.values[trial][basename].values(),
                    label=basename)
            ax.legend(labels)
            ax.set_xticks(list(self.values[trial][basename].keys())[::int(2/self.parameters['data_timestep_hr']/timestep_ratio)])
            return ax, labels
        
        # def recursive_sum(array_list):
        #     y1 = array_list[0]
        #     for y in array_list[1:]:
        #         y1 = y1+y
        #     return y1
        
        timestep_ratio = self.parameters['data_timestep_hr']/self.parameters['timestep_hr']
        if primal_values_zip_path:
            with ZipFile(primal_values_zip_path, 'r') as zp:
                zp.extractall()
                
        if primal_values_filename:  
            with open(primal_values_filename, 'r', encoding='utf-8') as primal:
                self.values = json.load(primal)
        
        # plot the content for desired trials 
        self.plots = []
        for graph in graphs:
            if 'total' in graph['content']:
                ys = []
            print(graph)
            pyplot.rcParams['figure.figsize'] = (11, 7)
            pyplot.rcParams['figure.dpi'] = 150
            fig, ax = pyplot.subplots()
            y_label = 'Variable value'
            for trial, basenames in self.values.items():
                content = graph['content']
                if 'biomass' in graph['content']:
                    y_label = 'grams'
                    content = 'b'
                if graph['content'] == 'growth':
                    content = 'g'   
                    y_label = 'grams/hour'
                if trial == graph['trial']:
                    labels:list = []
                    for basename in basenames:
                        # parse for non-concentration variables
                        print(basename)
                        if graph['phenotype'] == '*' and all([x in basename for x in [graph['species'], content]]):
                            if 'total' in graph['content']:
                                labels = [basename]
                                xs = self.values[trial][basename].keys()
                                ys.append(self.values[trial][basename].values())
                            else:
                                ax, labels = add_plot(ax, labels, basename, trial)
                            print('1')
                        elif all([x in basename for x in [graph['species'], graph['phenotype'], content]]):
                            ax, labels = add_plot(ax, labels, basename, trial)
                            print('2')
                        elif 'EX_' in basename and graph['content'] in basename:
                            ax, labels = add_plot(ax, labels, basename, trial)   
                            y_label = 'Concentration (mM)'
                            print('3')
                            
                    if labels != []:
                        if 'total' in graph['content']:
                            print(xs, ys)
                            ax.plot(xs, sum(ys), label=graph['content'])
                        phenotype_id = graph['phenotype'] if graph['phenotype'] != '*' else "all"
                        ax.set_xlabel('Time (hr)')
                        ax.set_ylabel(y_label)
                        ax.set_title(f'{graph["content"]} of {graph["species"]} ({phenotype_id}) in the {trial} trial')
                        fig_name = f'{"_".join([trial, graph["species"], phenotype_id, graph["content"]])}.jpg'
                        fig.savefig(fig_name)
                        self.plots.append(fig_name)
        
        # combine the figures with the other cotent
        if not zip_name:
            if hasattr(self, zip_name()):
                zip_name = self.zip_name
        if zip_name:
            with ZipFile(zip_name, 'a') as zp:
                for plot in self.plots:
                    zp.write(plot)
                    os.remove(plot)
