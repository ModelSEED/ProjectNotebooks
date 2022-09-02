# -*- coding: utf-8 -*-
# from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError
from pandas import read_csv, DataFrame, ExcelFile
from optlang import Variable, Constraint, Objective, Model
from cobra.core.metabolite import Metabolite
from cobra.medium import minimal_medium
from modelseedpy.core.fbahelper import FBAHelper
from modelseedpy.core.optlanghelper import OptlangHelper
from scipy.constants import hour
from scipy.optimize import newton
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from itertools import chain
from copy import deepcopy
from sympy.core.add import Add
from matplotlib import pyplot
from typing import Union, Iterable
from pprint import pprint
from time import sleep, process_time
import numpy as np
# from cplex import Cplex
import json, os, re


def isnumber(string):
    try:
        float(string)
    except:
        return False
    return True

names = []
def _name(name, suffix, time, trial, catch_duplicates=False):
    name = '-'.join([name+suffix, time, trial])
    if catch_duplicates:
        if name not in names:
            names.append(name)
            return name
        else:
            raise ObjectAlreadyDefinedError(f"The object {name} is already defined for the problem.")
    else:
        return name

class MSCommFitting(): 

    def __init__(self):
        self.parameters, self.variables, self.constraints, self.dataframes, self.signal_species = {}, {}, {}, {}, {}
        self.phenotypes_parsed_df: np.ndarray; self.problem: object; self.species_phenotypes_bool_df: object
        self.zipped_output, self.plots = [], []
        
    def _process_csv(self, csv_path, index_col):
        self.zipped_output.append(csv_path)
        csv = read_csv(csv_path)
        csv.index = csv[index_col]
        csv.drop(index_col, axis=1, inplace=True)
        csv.astype(str)
        return csv
    
    def _df_construction(self, name, signal, ignore_trials, ignore_timesteps, significant_deviation):
        # parse the DataFrame for values
        self.signal_species[signal] = name
        self.simulation_time = self.dataframes[signal].iloc[0,-1]/hour
        self.parameters["data_timestep_hr"].append(self.simulation_time/int(self.dataframes[signal].columns[-1]))
        
        # refine the DataFrame
        self.dataframes[signal] = self.dataframes[signal].iloc[1::2]  # excludes the times
        self.dataframes[signal].columns = map(str, self.dataframes[signal].columns)
        self.dataframes[signal].index = self.dataframes[signal]['Well']
        for col in self.dataframes[signal].columns:
            if any([x in col for x in ['Plate', 'Well', 'Cycle']]):
                self.dataframes[signal].drop(col, axis=1, inplace=True)
        self.dataframes[signal].columns = map(float, self.dataframes[signal].columns)
        self.dataframes[signal].columns = map(int, self.dataframes[signal].columns)
        
        # filter data contents
        dropped_trials = []
        if isinstance(ignore_trials, dict):
            ignore_trials['columns'] = list(map(str, ignore_trials['columns'])) if 'columns' in ignore_trials else []
            ignore_trials['rows'] = list(map(str, ignore_trials['rows'])) if 'rows' in ignore_trials else []
            ignore_trials['wells'] = ignore_trials['wells'] if 'wells' in ignore_trials else []
        elif isinstance(ignore_trials, list):
            if ignore_trials[0][0].isalpha() and isnumber(ignore_trials[0][1:]):
                short_code = True  # !!! drop trials with respect to the short codes, and not the full codes
        for trial in self.dataframes[signal].index:
            if isinstance(ignore_trials, dict) and any(
                    [trial[0] in ignore_trials['rows'], trial[1:] in ignore_trials['columns'], trial in ignore_trials['wells']]
                    ) or isinstance(ignore_trials, list) and trial in ignore_trials:
                self.dataframes[signal].drop(trial, axis=0, inplace=True)
                dropped_trials.append(trial)
            elif isinstance(ignore_trials, list) and trial in ignore_trials:
                self.dataframes[signal].drop(trial, axis=0, inplace=True)
                dropped_trials.append(trial)
        if dropped_trials:
            print(f'The {dropped_trials} trials were dropped from the {name} measurements.')

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
            if removed_trials:
                print(f'The {removed_trials} trials were removed from the {name} measurements, with their deviation over time being less than the threshold of {significant_deviation}.')
        
        # process the data for subsequent operations and optimal efficiency
        self.dataframes[signal].astype(str)
        self.dataframes[signal]: np.ndarray = FBAHelper.parse_df(self.dataframes[signal])
        
        # differentiate the phenotypes for each species
        if "OD" not in signal:
            self.species_phenotypes_bool_df.loc[signal]: np.ndarray[int] = np.array([
                1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_parsed_df[1]])
            
    def _export_model_json(self, json_model, path): 
        with open(path, 'w') as lp:
            json.dump(json_model, lp, indent=3)
    
    def load_data(self, base_media, community_members: dict = {}, solver:str = 'glpk', phenotype_met:dict = {}, signal_csv_paths:dict = {},
                  phenotypes_csv_path: str = None, species_abundance_path:str = None, carbon_conc_series: dict = {},
                  ignore_trials:Union[dict,list]=None, ignore_timesteps:list=[], significant_deviation:float = 2, extract_zip_path:str = None):
        self.community_members = {content["name"]: list(content["phenotypes"].keys())+["stationary"] for member, content in community_members.items()}
        self.media_conc = {cpd.id: cpd.concentration for cpd in base_media.mediacompounds}
        self.zipped_output = []
        self.phenotype_met = phenotype_met
        if extract_zip_path:
            with ZipFile(extract_zip_path, 'r') as zp:
                zp.extractall()
        if species_abundance_path:
            self.species_abundances = self._process_csv(species_abundance_path, 'trial_column')
        if community_members:
            carbon_sources = [c for content in carbon_conc_series.values() for c in content]
            models = OrderedDict()
            solutions = []
            for org_model, content in community_members.items():  # excludes the stationary phenotype
                # calculate and apply the minimal_medium of each respective model
                model = org_model.copy()
                model.medium = minimal_medium(model)
                model_rxns = [rxn.id for rxn in model.reactions]
                model.solver = solver
                models[model] = {"exchanges":FBAHelper.exchange_reactions(model), "solutions":{},
                    "name": content["name"], "phenotypes": self.community_members[content["name"]]}
                # print(content['phenotypes'])
                for pheno, cpds in content['phenotypes'].items():
                    for cpdID, bounds in cpds.items():
                        rxnID = "EX_"+cpdID+"_e0"
                        if rxnID not in model_rxns:
                            model.add_boundary(metabolite=model.metabolites.get_by_id(cpdID), reaction_id=rxnID, type="exchange")
                        model.reactions.get_by_id(rxnID).bounds = bounds
                    models[model]["solutions"][content["name"]+'_'+pheno] = model.optimize()
                    solutions.append(models[model]["solutions"][content["name"]+'_'+pheno].objective_value)
                    
            # construct the parsed table of all exchange fluxes for each phenotype
            if all(np.array(solutions) == 0):
                raise NoFluxError("The metabolic models did not grow.")
            cols = {}
            ## biomass row
            cols["rxn"] = ["bio"]
            for model, content in models.items():
                for phenotype in content["phenotypes"]:
                    col = content["name"]+'_'+phenotype
                    cols[col] = [0]
                    if col in content["solutions"]:
                        bio_rxns = [x for x in content["solutions"][col].fluxes.index if "bio" in x]
                        flux = np.mean([content["solutions"][col].fluxes[rxn] for rxn in bio_rxns if content["solutions"][col].fluxes[rxn] != 0])
                        cols[col] = [flux]

            ## exchange reactions rows
            looped_cols = cols.copy(); looped_cols.pop("rxn")
            for model, content in models.items():
                for ex_rxn in content["exchanges"]:
                    cols["rxn"].append(ex_rxn.id)
                    for col in looped_cols:
                        if col in content["solutions"]:
                            ### reactions that are not present in the columns are ignored
                            if ex_rxn.id in list(content["solutions"][col].fluxes.index):
                                cols[col].append(content["solutions"][col].fluxes[ex_rxn.id])
                        else:
                            cols[col].append(0)

            ## construct the DataFrame
            fluxes_df = DataFrame(data=cols)
            fluxes_df.index = fluxes_df['rxn']; fluxes_df.drop('rxn', axis=1, inplace=True)
            fluxes_df = fluxes_df.groupby(fluxes_df.index).sum()
            fluxes_df = fluxes_df.loc[(fluxes_df != 0).any(axis=1)]
            fluxes_df.to_csv("fluxes.csv")
            self.zipped_output.append("fluxes.csv")
            
        # define only species for which data is defined
        modeled_species = list(v for v in signal_csv_paths.values() if ("OD" not in v and " " not in v))
        removed_phenotypes = [col for col in fluxes_df if not any([species in col for species in modeled_species])]
        for col in removed_phenotypes:
            fluxes_df.drop(col, axis=1, inplace=True)
        if removed_phenotypes:
            print(f'The {removed_phenotypes} phenotypes were removed since their species is not among those that are defined with data: {modeled_species}.')
        fluxes_df.astype(str)
        self.phenotypes_parsed_df = FBAHelper.parse_df(fluxes_df)
        self.species_phenotypes_bool_df = DataFrame(columns=self.phenotypes_parsed_df[1])
        
        # define carbon concentrations for each trial
        if 'columns' not in carbon_conc_series:
            carbon_conc_series['columns'] = {}
        if 'rows' not in carbon_conc_series:
            carbon_conc_series['rows'] = {}
        self.carbon_conc = carbon_conc_series
        
        # define the set of used trials
        self.parameters["data_timestep_hr"] = []
        ignore_timesteps = list(map(str, ignore_timesteps))
        
        # import and parse the raw CSV data
        self.zipped_output.append(signal_csv_paths['path'])
        if ".csv" in signal_csv_paths['path']:
            raw_data = read_csv(signal_csv_paths['path'])
        elif ".xls" in signal_csv_paths["path"]:
            raw_data = ExcelFile(signal_csv_paths['path'])
        for org_sheet, name in signal_csv_paths.items():
            if org_sheet != 'path':
                sheet = org_sheet.replace(' ', '_')
                if ".csv" in signal_csv_paths['path']:
                    self.dataframes[sheet] = raw_data
                elif ".xls" in signal_csv_paths["path"]:
                    self.dataframes[sheet] = raw_data.parse(org_sheet)
                self.dataframes[sheet].columns = self.dataframes[sheet].iloc[6]
                self.dataframes[sheet] = self.dataframes[sheet].drop(self.dataframes[sheet].index[:7])
                self._df_construction(name, sheet, ignore_trials, ignore_timesteps, significant_deviation)
        
        self.parameters["data_timestep_hr"] = np.mean(self.parameters["data_timestep_hr"])
        self.data_timesteps = int(self.simulation_time/self.parameters["data_timestep_hr"])
        self.trials = set(chain.from_iterable([list(df.index) for df in self.dataframes.values()]))
        
    def _met_id_parser(self, met):
        met_id = re.sub('(\_\w\d+)', '', met)
        met_id = met_id.replace('EX_', '', 1)
        met_id = met_id.replace('c_', '', 1)
        return met_id
    
    def _update_problem(self, contents: Iterable):
        for content in contents:
            self.problem.add(content)
            self.problem.update()
                
    def define_problem(self, parameters={}, export_zip_name:str=None, export_parameters:bool=True, export_lp:bool=True,
                       final_relative_carbon_conc:float=None, metabolites_to_track:list=None, bad_data_timesteps:dict = None, zero_start=[]):
        self.parameters.update({
            "timestep_hr": self.parameters['data_timestep_hr'],  # Timestep size of the simulation in hours 
            "cvct": 1,                      # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf": 1,                      # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv": 1,                       # This is the highest fraction of biomass for a given species that can change phenotypes in a single time step
            "cvmin": 0,                     # This is the lowest value the limit on phenotype conversion goes, 
            "v": 0.1,                       # the kinetics constant that is externally adjusted
            'carbon_sources': ['cpd00136', 'cpd00179'],  # 4hb, maltose
            'diffpos': 1, 'diffneg': 1, # objective coefficients to the diffpos and diffneg variables that correspond with the components of difference between experimental and predicted bimoass values
        })
        self.parameters.update(parameters)
        self.simulation_timesteps = list(map(str, range(1, int(self.simulation_time/self.parameters['timestep_hr'])+1)))
        self.problem = Model()
        print("Solver:",type(self.problem))
        
        # refine the applicable range of bad_data_timesteps
        if bad_data_timesteps:
            for trial in bad_data_timesteps:
                if ':' in bad_data_timesteps[trial]:
                    start, end = bad_data_timesteps[trial].split(':')
                    end = end or self.simulation_timesteps[-1]
                    bad_data_timesteps[trial] = list(range(int(start), int(end)))
            if '*' in bad_data_timesteps:
                bad_data_timesteps = {trial:bad_data_timesteps['*'] for trial in self.trials}
        
        # construct the problem        
        obj_coef = {}
        constraints, variables = [], []  # lists are orders-of-magnitude faster than numpy arrays for appending
        time_1 = process_time()
        for signal, parsed_df in self.dataframes.items():
            for met in self.phenotypes_parsed_df[0]:
                met_id = self._met_id_parser(met)
                if not metabolites_to_track and met_id != 'cpd00001' or metabolites_to_track and met_id in metabolites_to_track:
                    self.variables["c_"+met] = {}; self.constraints['dcc_'+met] = {}
                    for time in self.simulation_timesteps:
                        self.variables["c_"+met][time] = {}; self.constraints['dcc_'+met][time] = {}
                        for trial in parsed_df[0]:
                            # define biomass measurement conversion variables 
                            self.variables["c_"+met][time][trial] = Variable(
                                _name("c_", met, time, trial), lb=0, ub=1000)
                            # constrain initial time concentrations to the media or a large number if it is not explicitly defined
                            if time == self.simulation_timesteps[0] and not 'bio' in met_id:
                                initial_val = 100 if met_id not in self.media_conc else self.media_conc[met_id]
                                initial_val = initial_val if met_id not in zero_start else 0
                                if met_id in self.carbon_conc['rows'] and trial[0] in self.carbon_conc['rows'][met_id]:
                                    initial_val = self.carbon_conc['rows'][met_id][trial[0]]
                                if met_id in self.carbon_conc['columns'] and trial[1:] in self.carbon_conc['columns'][met_id]:
                                    initial_val = self.carbon_conc['columns'][met_id][trial[1:]]
                                self.variables["c_"+met][time][trial] = Variable(
                                    _name("c_", met, time, trial), lb=initial_val, ub=initial_val)
                            # mandate complete carbon consumption
                            if time == self.simulation_timesteps[-1] and met_id in self.parameters['carbon_sources']:
                                self.variables["c_"+met][time][trial] = Variable(
                                    _name("c_", met, time, trial), lb=0, ub=0)
                                if final_relative_carbon_conc:
                                    self.variables["c_"+met][time][trial] = Variable(
                                        _name("c_", met, time, trial), 
                                        lb=0, ub=self.variables["c_"+met]["1"][trial].lb*final_relative_carbon_conc)
                            variables.append(self.variables["c_"+met][time][trial])
            break   # prevents duplicated variables 
        for signal, parsed_df in self.dataframes.items():
            if 'OD' not in signal:
                for phenotype in self.phenotypes_parsed_df[1]:
                    if self.signal_species[signal] in phenotype:
                        self.constraints['dbc_'+phenotype] = {}
                        for time in self.simulation_timesteps:
                            self.constraints['dbc_'+phenotype][time] = {}
            
        
        for phenotype in self.phenotypes_parsed_df[1]:
            self.variables['cvt_'+phenotype] = {}; self.variables['cvf_'+phenotype] = {}
            self.variables['b_'+phenotype] = {}; self.variables['g_'+phenotype] = {}
            self.variables['v_'+phenotype] = {} 
            self.constraints['gc_'+phenotype] = {}; self.constraints['cvc_'+phenotype] = {}
            for time in self.simulation_timesteps:
                    self.variables['cvt_'+phenotype][time] = {}; self.variables['cvf_'+phenotype][time] = {}
                    self.variables['b_'+phenotype][time] = {}; self.variables['g_'+phenotype][time] = {}
                    self.variables['v_'+phenotype][time] = {}
                    self.constraints['gc_'+phenotype][time] = {}; self.constraints['cvc_'+phenotype][time] = {}
                    for trial in self.trials:
                        self.variables['b_'+phenotype][time][trial] = Variable(         # predicted biomass abundance
                            _name("b_", phenotype, time, trial), lb=0, ub=100)
                        self.variables['g_'+phenotype][time][trial] = Variable(         # biomass growth
                            _name("g_", phenotype, time, trial), lb=0, ub=1000)   
                        
                        if 'stationary' not in phenotype:
                            self.variables['cvt_'+phenotype][time][trial] = Variable(       # conversion rate to the stationary phase
                                _name("cvt_", phenotype, time, trial), lb=0, ub=100)  
                            self.variables['cvf_'+phenotype][time][trial] = Variable(       # conversion from to the stationary phase
                                _name("cvf_", phenotype, time, trial), lb=0, ub=100)   
                            
                            # 0 <= -cvt + bcv*b_{phenotype} + cvmin
                            self.constraints['cvc_'+phenotype][time][trial] = Constraint(
                                -self.variables['cvt_'+phenotype][time][trial] 
                                + self.parameters['bcv']*self.variables['b_'+phenotype][time][trial] + self.parameters['cvmin'],
                                lb=0, ub=None, name=_name('cvc_', phenotype, time, trial))  
                            
                            # g_{phenotype} - b_{phenotype}*v = 0
                            self.constraints['gc_'+phenotype][time][trial] = Constraint(
                                self.variables['g_'+phenotype][time][trial] 
                                - self.parameters['v']*self.variables['b_'+phenotype][time][trial],
                                lb=0, ub=0, name=_name("gc_", phenotype, time, trial))
                    
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
                met_id = self._met_id_parser(met)
                if not metabolites_to_track and 'cpd00001' != met_id or metabolites_to_track and met_id in metabolites_to_track:
                    for trial in parsed_df[0]:
                        for time in self.simulation_timesteps:
                            next_time = str(int(time)+1)
                            # c_{met} + dt*sum_k^K() - c+1_{met} = 0
                            self.constraints['dcc_'+met][time][trial] = Constraint(
                                self.variables["c_"+met][time][trial] - self.variables["c_"+met][next_time][trial] 
                                + np.dot(
                                    self.phenotypes_parsed_df[2][r_index]*half_dt, np.array([
                                        self.variables['g_'+phenotype][time][trial]+self.variables['g_'+phenotype][next_time][trial]
                                        for phenotype in self.phenotypes_parsed_df[1]])),
                                ub=0, lb=0, name=_name("dcc_", met, time, trial))
                            
                            constraints.append(self.constraints['dcc_'+met][time][trial])
                            if next_time == self.simulation_timesteps[-1]:
                                break
            break   # prevents duplicated constraints
                    
        time_3 = process_time()
        print(f'Done with metabolites loop: {(time_3-time_2)/60} min')
        for signal, parsed_df in self.dataframes.items():
            data_timestep = 1
            self.variables[signal+'__conversion'] = Variable(signal+'__conversion', lb=0, ub=1000)
            variables.append(self.variables[signal+'__conversion'])
            
            self.variables[signal+'__bio'] = {}; self.variables[signal+'__diffpos'] = {}
            self.variables[signal+'__diffneg'] = {}
            self.constraints[signal+'__bioc'] = {}; self.constraints[signal+'__diffc'] = {}  # diffc is defined latter
            for time in self.simulation_timesteps:
                if int(time)*self.parameters['timestep_hr'] >= data_timestep*self.parameters['data_timestep_hr']:  # synchronizes user timesteps with data timesteps
                    data_timestep += 1
                    if int(data_timestep) > self.data_timesteps:
                        break
                    next_time = str(int(time)+1)
                    self.variables[signal+'__bio'][time] = {}; self.variables[signal+'__diffpos'][time] = {}
                    self.variables[signal+'__diffneg'][time] = {}
                    self.constraints[signal+'__bioc'][time] = {}; self.constraints[signal+'__diffc'][time] = {}
                    for r_index, trial in enumerate(parsed_df[0]):
                        if not bad_data_timesteps or trial not in bad_data_timesteps or time not in bad_data_timesteps[trial]:
                            total_biomass:Add = 0; signal_sum: Add = 0; from_sum: Add = 0; to_sum: Add = 0
                            for phenotype in self.phenotypes_parsed_df[1]:
                                total_biomass += self.variables["b_"+phenotype][time][trial]
                                # print(signal, phenotype)
                                # display(self.species_phenotypes_bool_df)
                                # OD is included in every signal
                                val = 1 if 'OD' in signal else self.species_phenotypes_bool_df.loc[signal, phenotype]
                                val = val if isnumber(val) else val.values[0]
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
                                            ub=0, lb=0, name=_name("dbc_", phenotype, time, trial))
                                    else:
                                        # -b_{phenotype} + dt*g_{phenotype} + cvf - cvt - b+1_{phenotype} = 0
                                        self.constraints['dbc_'+phenotype][time][trial] = Constraint( 
                                            self.variables['b_'+phenotype][time][trial] - self.variables['b_'+phenotype][next_time][trial]
                                            + half_dt*(self.variables['g_'+phenotype][time][trial]+self.variables['g_'+phenotype][next_time][trial])
                                            + self.variables['cvf_'+phenotype][time][trial] - self.variables['cvt_'+phenotype][time][trial],
                                            ub=0, lb=0, name=_name("dbc_", phenotype, time, trial))
                                    
                                    constraints.append(self.constraints['dbc_'+phenotype][time][trial])
                                    
                            self.variables[signal+'__bio'][time][trial] = Variable(
                                _name(signal, '__bio', time, trial), lb=0, ub=1000)
                            self.variables[signal+'__diffpos'][time][trial] = Variable( 
                                _name(signal, '__diffpos', time, trial), lb=0, ub=100) 
                            self.variables[signal+'__diffneg'][time][trial] = Variable(  
                                _name(signal, '__diffneg', time, trial), lb=0, ub=100) 
                                
                            # {signal}__conversion*datum = {signal}__bio
                            self.constraints[signal+'__bioc'][time][trial] = Constraint(
                                self.variables[signal+'__conversion']*parsed_df[2][r_index, int(data_timestep)-1] 
                                - self.variables[signal+'__bio'][time][trial], 
                                name=_name(signal, '__bioc', time, trial), lb=0, ub=0)
                            
                            # {speces}_bio - sum_k^K(es_k*b_{phenotype}) - {signal}_diffpos + {signal}_diffneg = 0
                            self.constraints[signal+'__diffc'][time][trial] = Constraint( 
                                self.variables[signal+'__bio'][time][trial] - signal_sum 
                                - self.variables[signal+'__diffpos'][time][trial]
                                + self.variables[signal+'__diffneg'][time][trial], 
                                name=_name(signal, '__diffc', time, trial), lb=0, ub=0)
        
                            obj_coef.update({self.variables[signal+'__diffpos'][time][trial]: self.parameters['diffpos'],
                                             self.variables[signal+'__diffneg'][time][trial]: self.parameters['diffneg']})                            
                            variables.extend([self.variables[signal+'__bio'][time][trial], 
                                              self.variables[signal+'__diffpos'][time][trial],
                                              self.variables[signal+'__diffneg'][time][trial]])
                            constraints.extend([self.constraints[signal+'__bioc'][time][trial],
                                               self.constraints[signal+'__diffc'][time][trial]])
                
        time_4 = process_time()
        print(f'Done with the dbc & diffc loop: {(time_4-time_3)/60} min')
        # construct the problem
        self._update_problem([variables, constraints])
        self.problem.objective = Objective(Zero, direction="min") #, sloppy=True)
        self.problem.objective.set_linear_coefficients(obj_coef)
        time_5 = process_time()
        print(f'Done with loading the variables, constraints, and objective: {(time_5-time_4)/60} min')
                
        # print contents
        if export_parameters:
            self.zipped_output.append('parameters.csv')
            DataFrame(data=list(self.parameters.values()),
                      index=list(self.parameters.keys()), 
                      columns=['values']
                      ).to_csv('parameters.csv')
        if export_lp:
            self.zipped_output.extend(['mscommfitting.lp', 'mscommfitting.json'])
            with open('mscommfitting.lp', 'w') as lp:
                lp.write(self.problem.to_lp())
            self._export_model_json(self.problem.to_json(), 'mscommfitting.json')
        if export_zip_name:
            self.zip_name = export_zip_name
            sleep(2)
            with ZipFile(self.zip_name, 'w', compression=ZIP_LZMA) as zp:
                for file in self.zipped_output:
                    zp.write(file)
                    os.remove(file)
                    
        time_6 = process_time()
        print(f'Done exporting the content: {(time_6-time_5)/60} min')
                
    def compute(self, graphs:list = [], export_zip_name=None, publishing=False):
        self.values = {}
        solution = self.problem.optimize()
        # categorize the primal values by trial and time
        if all(np.array(list(self.problem.primal_values.values())) == 0):
            raise NoFluxError("The simulation lacks any flux.")
        for variable, value in self.problem.primal_values.items():
            if 'conversion' not in variable:
                basename, time, trial = variable.split('-')
                time = int(time)*self.parameters['data_timestep_hr']
                if not trial in self.values:
                    self.values[trial] = {}
                if not basename in self.values[trial]:
                    self.values[trial][basename] = {}
                self.values[trial][basename][time] = value
                
        # export the processed primal values for graphing
        with open('primal_values.json', 'w') as out:
            json.dump(self.values, out, indent=3)
        if not export_zip_name:
            if hasattr(self, 'zip_name'):
                export_zip_name = self.zip_name
        if export_zip_name:
            with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
                zp.write('primal_values.json')
                os.remove('primal_values.json')
        
        if graphs != []:
            self.graph(graphs, zip_name=export_zip_name, publishing=publishing)
            
        if "optimal" in  solution:
            print('The solution is optimal.')
        else:
            raise FeasibilityError(f'The solution is sub-optimal, with a {solution} status.')
                
    def graph(self, graphs = [], primal_values_filename:str = None, primal_values_zip_path:str = None, zip_name:str = None,
              data_timestep_hr:float = 0.163, publishing:bool = False, title:str=None):
        def add_plot(ax, labels, basename, trial, linestyle="solid"):
            labels.append(basename.split('-')[-1])
            ax.plot(list(self.values[trial][basename].keys()),
                    list(self.values[trial][basename].values()),
                    label=basename, linestyle=linestyle)
            ax.legend(labels)
            x_ticks = np.around(np.array(list(self.values[trial][basename].keys())), 0)
            ax.set_xticks(x_ticks[::int(2/data_timestep_hr/timestep_ratio)])
            return ax, labels
        
        timestep_ratio = 1
        if self.parameters != {}:
            data_timestep_hr = self.parameters['data_timestep_hr']
            timestep_ratio = self.parameters['data_timestep_hr']/self.parameters['timestep_hr']
        if primal_values_filename:
            if primal_values_zip_path:
                with ZipFile(primal_values_zip_path, 'r') as zp:
                    zp.extract(primal_values_filename)
            with open(primal_values_filename, 'r', encoding='utf-8') as primal:
                self.values = json.load(primal)
        
        # plot the content for desired trials 
        self.plots = []
        contents = {"biomass":'b', "growth":'g'}
        print(self.signal_species)
        for graph in graphs:
            content = graph['content'] if graph['content'] not in contents else contents[graph['content']]
            y_label = 'Variable value'
            x_label = 'Time (hr)'
            if any([x in graph['content'] for x in ['total', 'OD', 'all_biomass']]):
                ys = {}
                for name in self.signal_species.values():
                    ys[name] = []
            if any(x == graph['content'] for x in ['OD', 'all_biomass']):
                y_label = 'Biomass concentration (g/L)'
                graph['phenotype'] = graph['species'] = '*'
            elif 'biomass' in graph['content']:
                y_label = 'Biomass concentration (g/L)'
            elif graph['content'] == 'growth':
                y_label = 'Biomass growth (g/hr)'
            # elif 'stress-test' in graph['content']:
            #     content = graph['content'].split('_')[1]
            #     y_label = graph['species']+' coculture %'
            #     x_label = content+' (mM)'
            if "species" not in graph or graph['species'] == '*':
                graph['species'] = list(self.signal_species.values())
            if "phenotype" not in graph or graph['phenotype'] == '*':
                graph['phenotype'] = set(chain(*[phenos for species, phenos in self.community_members.items() if species in graph["species"]]))
            pprint(graph)
            
            # figure specifications
            pyplot.rcParams['figure.figsize'] = (11, 7)
            pyplot.rcParams['figure.dpi'] = 150
            if publishing:
                pyplot.rc('axes', titlesize=20, labelsize=20)
                pyplot.rc('xtick', labelsize=20) 
                pyplot.rc('ytick', labelsize=20) 
                pyplot.rc('legend', fontsize=18)
            fig, ax = pyplot.subplots()
            x_ticks = None
            
            # define the figure contents
            for trial, basenames in self.values.items():
                if trial in graph['trial']:
                    labels = []
                    for basename in basenames:
                        # parse for non-concentration variables
                        print(basename)
                        if any([x in content for x in ['total', 'all_biomass', 'OD']]):
                            if 'b_' in basename:
                                var_name, species, phenotype = basename.split('_')
                                label = f'{species}_biomass (model)'
                                if publishing:
                                    for signal, species_name in self.signal_species.items():
                                        if species == species_name:
                                             break
                                    if species == 'ecoli':
                                        species_name = 'E. coli'  
                                    elif species == 'pf':
                                        species_name = 'P. fluorescens'
                                    label = f'{species_name} biomass from optimized model'
                                labels.append({species:label})
                                xs = np.array(list(self.values[trial][basename].keys()))
                                ax.set_xticks(x_ticks[::int(3/data_timestep_hr/timestep_ratio)])
                                if any([x in content for x in ['all_biomass', 'OD']]):
                                    ys['OD'].append(np.array(list(self.values[trial][basename].values())))
                                if any([x in content for x in ['all_biomass', 'total']]):
                                    ys[species].append(np.array(list(self.values[trial][basename].values())))
                            if 'experimental_data' in graph and graph['experimental_data']:
                                print('exp', basename)
                                if any(['__bio' in basename and content == 'all_biomass',
                                        basename == 'OD__bio' and content == 'OD']):
                                    signal = basename.split('_')[0]
                                    label = basename
                                    if publishing:
                                        if self.signal_species[signal] == 'ecoli':
                                            species = 'E. coli'  
                                        elif self.signal_species[signal] == 'pf':
                                            species = 'P. fluorescens'
                                        label = f'Experimental {species} profile (from {signal})'
                                        if signal == 'OD':
                                            label = 'Experimental total biomass (from OD)'
                                    labels.append(label)
                                    exp_xs = np.array(list(self.values[trial][basename].keys()))
                                    exp_xs = exp_xs.astype(np.float32)
                                    ax.plot(exp_xs, list(self.values[trial][basename].values()), label=label)
                                    x_ticks = np.around(exp_xs, 0)
                        elif graph['phenotype'] == '*' and content in basename and all([x in basename for x in graph['species']]):
                            if 'total' in content:
                                labels = [basename]
                                xs = np.array(list(self.values[trial][basename].keys()))
                                ys.append(np.array(list(self.values[trial][basename].values())))
                                ax.set_xticks(x_ticks[::int(3/data_timestep_hr/timestep_ratio)])
                            else:
                                ax, labels = add_plot(ax, labels, basename, trial)
                            print('1')
                        # 
                        elif any([x in basename for x in graph['phenotype']]):
                            if any([x in basename for x in graph['species']]) and content in basename:
                                linestyle="solid" if "ecoli" in basename else "dashed"
                                ax, labels = add_plot(ax, labels, basename, trial, linestyle)
                                print(basename, '2')
                        # concentration plots
                        elif 'EX_' in basename and content in basename:
                            ax, labels = add_plot(ax, labels, basename, trial)   
                            y_label = 'Concentration (mM)'
                            print('3')
                       
                    if labels != []:
                        if any([x in content for x in ['OD', 'all_biomass', 'total']]):
                            xs = xs.astype(np.float32)
                            for name in ys:
                                if ys[name] != []:
                                    label=f'{name}_biomass (model)'
                                    if publishing:
                                        if name == 'OD':
                                            label = 'Total biomass from optimized model'
                                        else:
                                            for lbl in labels:
                                                if isinstance(lbl,dict):
                                                    if name in lbl:
                                                        label = lbl[name]
                                    ax.plot(xs, sum(ys[name]), label=label)
                        phenotype_id = graph['phenotype'] if isinstance(graph['phenotype'], str) else f"{','.join(graph['phenotype'])} phenotypes"
                        species_id = graph["species"] if graph["species"] != '*' and isinstance(graph["species"], str) else 'all species'
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        if len(labels) > 1:
                            ax.legend()
                        if not publishing:
                            if not title:
                                org_content = content if content not in contents else list(contents.keys())[list(contents.values()).index(content)]
                                this_title = f'{org_content} of {species_id} ({phenotype_id}) in the {trial} trial'
                                if "cpd" in content:
                                    this_title=f"{org_content} in the {trial} trial"
                                ax.set_title(this_title)
                            else:
                                ax.set_title(title)
                        fig_name = f'{"_".join([trial, species_id, phenotype_id, content])}.jpg'
                        fig.savefig(fig_name)
                        self.plots.append(fig_name)
        
        # combine the figures with the other cotent
        if not zip_name:
            if hasattr(self, 'zip_name'):
                zip_name = self.zip_name
        if zip_name:
            with ZipFile(zip_name, 'a', compression=ZIP_LZMA) as zp:
                for plot in self.plots:
                    zp.write(plot)
                    os.remove(plot)
                    
    def load_model(self, mscomfit_json_path:str=None, zip_name:str = None, model_to_load:dict=None):
        if zip_name:
            with ZipFile(zip_name, 'r') as zp:
                zp.extract(mscomfit_json_path)
        if mscomfit_json_path:
            with open(mscomfit_json_path, 'r') as mscmft:
                return json.load(mscmft)
        if model_to_load:
            print(model_to_load.keys())
            self.problem = Model.from_json(model_to_load)
        
    def change_parameters(self, cvt=None, cvf=None, diff=None, vmax={}, km={}, error_threshold:float=1, strain:str=None, graphs:list=None,
                          mscomfit_json_path='mscommfitting.json', primal_values_filename:str=None, export_zip_name=None, extract_zip_name=None,
                          final_concentrations:dict=None, final_relative_carbon_conc:float=None, previous_relative_conc:float=None):
        def change_param(param, time, trial):
            if not isinstance(param, dict):
                return param
            if time in param:
                if trial in param[time]:
                    return param[time][trial]
                return param[time]
            return param['default']

        def change_vmax(mscomfit_json, vmax):
            for arg in mscomfit_json['constraints']:  # !!! specify as phenotype-specific, as well as the Km
                name, time, trial = arg['name'].split('-')
                if 'gc' in name:
                    arg['expression']['args'][1]['args'][0]['value'] = change_param(vmax, time, trial)
            return mscomfit_json
        
        def universalize(param, met_id, variable):
            new_param = param.copy()
            for time in variable:
                new_param[met_id][time] = {}
                vmax_val = param[met_id] if not isinstance(vmax_val, dict) else vmax_val[time]
                for trial in variable[time]:
                    vmax_val = vmax_val if not isinstance(vmax_val, dict) else vmax_val[trial]
                    new_param[met_id][time][trial] = vmax_val
            return new_param

        # load the model
        time_1 = process_time()
        if not os.path.exists(mscomfit_json_path):
            extract_zip_name = extract_zip_name or self.zip_name
            mscomfit_json = self.load_model(mscomfit_json_path, zip_name=extract_zip_name)
        else:
            mscomfit_json = self.load_model(mscomfit_json_path)
        time_2 = process_time()
        print(f'Done loading the JSON: {(time_2-time_1)/60} min')
        
        # change objective coefficients
        if any([cvf, cvt, diff]):
            for arg in mscomfit_json['objective']['expression']['args']:
                name, time, trial = arg['args'][1]['name'].split('-')
                if cvf and 'cvf' in name:
                    arg['args'][0]['value'] = change_param(cvf, time, trial)
                if cvt and 'cvt' in name:
                    arg['args'][0]['value'] = change_param(cvt, time, trial)
                if diff and 'diff' in name:
                    arg['args'][0]['value'] = change_param(diff, time, trial)

        if km and not vmax:
            raise ParameterError(f'A Vmax must be defined with the Km of {km}.')
        if final_relative_carbon_conc or final_concentrations or km and vmax:
            # uploads primal values when they are not in RAM
            if not hasattr(self, 'values'):
                with open(primal_values_filename, 'r') as pv:
                    self.values = json.load(pv)
            initial_concentrations = {} ; already_constrained = []
            for var in mscomfit_json['variables']:
                if 'EX_' in var['name']:
                    met = var ; met_name, time, trial = met['name'].split('-')
                    if time == self.simulation_timesteps[0]:
                        initial_concentrations[met_name] = met["ub"]
                    if time == self.simulation_timesteps[-1]:
                        # assign an absolute final concentration
                        if final_concentrations and met_name in final_concentrations:
                            met['lb'] = 0 ; met['ub'] = final_concentrations[met_name]
                        # assign a relative final concentration
                        elif final_relative_carbon_conc and any([x in met_name for x in self.parameters['carbon_sources']]):
                            print(met['ub'])
                            met['lb'] = 0 ; met['ub'] = initial_concentrations[met_name]*final_relative_carbon_conc
                            if previous_relative_conc:
                                met['ub'] /= previous_relative_conc
                                print(met['ub'])

                    if met_name not in already_constrained:
                        already_constrained.append(met_name)
                        # change growth kinetics
                        met_id = self._met_id_parser(met_name)
                        if met_id in self.phenotype_met.values() and vmax:
                            # defines the Vmax for each metabolite, or applies a constant Vmax for all instances in the same dict structure
                            vmax = vmax if not isinstance(vmax[met_id], (float,int)) else universalize(vmax, met_id, self.variables[met_name])
                            # calculate the Michaelis-Menten kinetic rate: vmax / (km + [maltose])
                            if km:  # appropriate starting values are Vmax=2.2667 & Km=2, given [maltose]=5, to yield the ideal 0.3 constant
                                vmax_var = vmax.copy() ; conc_tracker = {}
                                print(met_id)
                                count = error = last_conc_same_count = last_conc = 0
                                while (last_conc_same_count < 5):  # unknown necessary threshold 
                                    error = 0
                                    for time in self.variables[met_name]:
                                        time_hr = int(time)*self.parameters['timestep_hr']
                                        conc_tracker[time] = {} if time not in conc_tracker else conc_tracker[time]
                                        for trial in self.variables[met_name][time]:
                                            if trial in conc_tracker[time]:
                                                error += (conc_tracker[time][trial]-self.values[trial][met_name][time_hr])**2
                                            conc_tracker[time][trial] = self.values[trial][met_name][time_hr]
                                            vmax_var[met_id][time][trial] = -vmax[met_id][time][trial]/(km[met_id]+conc_tracker[time][trial])
                                            print('new growth rate: ', vmax_var[met_id][time][trial])
                                            count += 1
                                    last_conc_same_count += 1 if last_conc == conc_tracker[time][trial] else 0
                                    last_conc = conc_tracker[time][trial]
                                    mscomfit_json = change_vmax(mscomfit_json, vmax_var[met_id])
                                    # self._export_model_json(mscomfit_json, mscomfit_json_path)
                                    self.load_model(model_to_load=mscomfit_json)
                                    self.compute(graphs)#, export_zip_name)
                                    if error != 0:
                                        error = (error/count)**0.5
                                        print("Error:",error)
                            else:
                                mscomfit_json = change_vmax(mscomfit_json, vmax[met_id])
        
        self._export_model_json(mscomfit_json, mscomfit_json_path)
        export_zip_name = export_zip_name or self.zip_name
        with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
            zp.write(mscomfit_json_path)
            os.remove(mscomfit_json_path)
        time_3 = process_time()
        print(f'Done exporting the model: {(time_3-time_2)/60} min')
            
        self.problem = Model.from_json(mscomfit_json)
        time_4 = process_time()
        print(f'Done loading the model: {(time_4-time_3)/60} min')
    
    def parameter_optimization(self,):
        with ZipFile(self.zip_name, 'r') as zp:
            zp.extract('mscommfitting.json')
            
        newton
