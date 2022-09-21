# -*- coding: utf-8 -*-
# from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from pandas import read_csv, DataFrame, ExcelFile
from optlang import Model
from cobra.medium import minimal_medium
from modelseedpy.core.fbahelper import FBAHelper
from scipy.constants import hour
from scipy.optimize import newton
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from itertools import chain
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


def dict_keys_exists(dic, *keys):
    if keys[0] in dic:
        remainingKeys = keys[1:]
        if len(remainingKeys) > 0:
            dict_keys_exists(dic[keys[0]], keys[1:])
        return True
    return False


def default_dict_values(dic, key, default):
    return default if not dict_keys_exists(dic, key) else dic[key]


# define data objects
names = []


def _name(name, suffix, time, trial):
    name = '-'.join([name+suffix, time, trial])
    if name not in names:
        names.append(name)
        return name
    else:
        raise ObjectAlreadyDefinedError(f"The object {name} is already defined for the problem.")


class MSCommFitting:

    def __init__(self):
        self.parameters, self.variables, self.constraints, self.dataframes, self.signal_species = {}, {}, {}, {}, {}
        self.zipped_output, self.plots = [], []
        
    def _process_csv(self, csv_path, index_col):
        self.zipped_output.append(csv_path)
        csv = read_csv(csv_path) ; csv.index = csv[index_col]
        csv.drop(index_col, axis=1, inplace=True)
        csv.astype(str)
        return csv
    
    def _df_construction(self, name, signal, ignore_trials, ignore_timesteps, significant_deviation):
        # parse the DataFrame for values
        self.signal_species[signal] = name
        self.simulation_time = self.dataframes[signal].iloc[0,-1]/hour
        self.parameters["data_timestep_hr"].append(self.simulation_time/int(self.dataframes[signal].columns[-1]))
        
        # refine the DataFrame
        # TODO - this must be replaced for the standardized experimental data
        self.dataframes[signal] = self.dataframes[signal].iloc[1::2]  # excludes the times
        self.dataframes[signal].columns = map(str, self.dataframes[signal].columns)
        self.dataframes[signal].index = self.dataframes[signal]['Well']
        for col in self.dataframes[signal].columns:
            if any([x in col for x in ['Plate', 'Well', 'Cycle']]):
                self.dataframes[signal].drop(col, axis=1, inplace=True)
        self.dataframes[signal].columns = map(float, self.dataframes[signal].columns)
        self.dataframes[signal].columns = map(int, self.dataframes[signal].columns)
        
        # filter data contents
        # TODO - this must be replaced for the standardized experimental data
        dropped_trials = []
        if isinstance(ignore_trials, dict):
            ignore_trials['columns'] = list(map(str, ignore_trials['columns'])) if 'columns' in ignore_trials else []
            ignore_trials['rows'] = list(map(str, ignore_trials['rows'])) if 'rows' in ignore_trials else []
            ignore_trials['wells'] = ignore_trials['wells'] if 'wells' in ignore_trials else []
        elif isIterable(ignore_trials):
            if ignore_trials[0][0].isalpha() and isnumber(ignore_trials[0][1:]):
                short_code = True  # !!! drop trials with respect to the short codes, and not the full codes
        for trial in self.dataframes[signal].index:
            if isinstance(ignore_trials, dict) and any(
                    [trial[0] in ignore_trials['rows'], trial[1:] in ignore_trials['columns'], trial in ignore_trials['wells']]
                    ) or isIterable(ignore_trials) and trial in ignore_trials:
                self.dataframes[signal].drop(trial, axis=0, inplace=True)
                dropped_trials.append(trial)
            elif isIterable(ignore_trials) and trial in ignore_trials:
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
                print(f'The {removed_trials} trials were removed from the {name} measurements, '
                      f'with their deviation over time being less than the threshold of {significant_deviation}.')

        # process the data for subsequent operations and optimal efficiency
        self.dataframes[signal].astype(str)
        self.dataframes[signal] = FBAHelper.parse_df(self.dataframes[signal])

        # differentiate the phenotypes for each species
        if "OD" not in signal:
            self.species_phenos_df.loc[signal]: np.ndarray[int] = np.array([
                1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenos_tup.columns])

    def _export_model_json(self, json_model, path):
        with open(path, 'w') as lp:
            json.dump(json_model, lp, indent=3)
    
    def load_data(self, base_media, community_members: dict, solver:str = 'glpk', phenotype_met:dict = None, signal_csv_paths:dict = None,
                  species_abundance_path:str = None, carbon_conc_series: dict = None, ignore_trials:Union[dict,list]=None,
                  ignore_timesteps:list=None, significant_deviation:float = 2, extract_zip_path:str = None):
        # define default values
        ignore_timesteps = ignore_timesteps or []
        signal_csv_paths = signal_csv_paths or {}
        carbon_conc_series = carbon_conc_series or {}

        self.community_members = {content["name"]: list(content["phenotypes"].keys())+["stationary"] for member, content in community_members.items()}
        self.phenotype_met = phenotype_met or {content["name"]:list(
            v.keys()) for content in community_members.values() for k,v in content["phenotypes"].items()}
        self.media_conc = {cpd.id: cpd.concentration for cpd in base_media.mediacompounds}
        self.zipped_output = []
        if extract_zip_path:
            with ZipFile(extract_zip_path, 'r') as zp:
                zp.extractall()
        if species_abundance_path:
            self.species_abundances = self._process_csv(species_abundance_path, 'trial_column')

        # log information of each respective model
        models = OrderedDict()
        solutions = []
        for org_model, content in community_members.items():  # community_members excludes the stationary phenotype
            ## define the model
            model = org_model.copy()
            model.medium = minimal_medium(model)
            model_rxns = [rxn.id for rxn in model.reactions]
            model.solver = solver
            ## log the model
            models[model] = {"exchanges":FBAHelper.exchange_reactions(model), "solutions":{},
                "name": content["name"], "phenotypes": self.community_members[content["name"]]}
            for pheno, cpds in content['phenotypes'].items():
                col = content["name"] + '_' + pheno
                for cpdID, bounds in cpds.items():
                    rxnID = "EX_"+cpdID+"_e0"
                    if rxnID not in model_rxns:
                        model.add_boundary(metabolite=model.metabolites.get_by_id(cpdID), reaction_id=rxnID, type="exchange")
                    model.reactions.get_by_id(rxnID).bounds = bounds
                models[model]["solutions"][col] = model.optimize()
                solutions.append(models[model]["solutions"][col].objective_value)

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
                    ### reactions that are not present in the columns are ignored
                    flux = 0 if col not in content["solutions"] or \
                        ex_rxn.id not in list(content["solutions"][col].fluxes.index) else content["solutions"][col].fluxes[ex_rxn.id]
                    cols[col].append(flux)

        ## construct the DataFrame
        fluxes_df = DataFrame(data=cols)
        fluxes_df.index = fluxes_df['rxn'] ; fluxes_df.drop('rxn', axis=1, inplace=True)
        fluxes_df = fluxes_df.groupby(fluxes_df.index).sum()
        fluxes_df = fluxes_df.loc[(fluxes_df != 0).any(axis=1)]
        fluxes_df.astype(str)
        fluxes_df.to_csv("fluxes.csv")
        self.zipped_output.append("fluxes.csv")
            
        # define only species for which data is defined
        modeled_species = list(v for v in signal_csv_paths.values() if ("OD" not in v and " " not in v))
        removed_phenotypes = [col for col in fluxes_df if not any([species in col for species in modeled_species])]
        for col in removed_phenotypes:
            fluxes_df.drop(col, axis=1, inplace=True)
        if removed_phenotypes:
            print(f'The {removed_phenotypes} phenotypes were removed '
                  f'since their species is not among those with data: {modeled_species}.')
        self.phenos_tup = FBAHelper.parse_df(fluxes_df)
        self.species_phenos_df = DataFrame(columns=self.phenos_tup.columns)

        # define carbon concentrations for each trial
        # carbon_sources = [c for content in carbon_conc_series.values() for c in content]
        carbon_conc_series['columns'] = default_dict_values(carbon_conc_series, "columns", {})
        carbon_conc_series['rows'] = default_dict_values(carbon_conc_series, "rows", {})
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
                # TODO - this must be changed to accommodate the standardized experimental format
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
                
    def define_problem(self, parameters=None, export_zip_name:str=None, export_parameters:bool=True, export_lp:bool=True,
                       final_rel_c12_conc:float=0, mets_to_track:list=None, bad_data_timesteps:dict = None, zero_start=None):
        # define default values
        mets_to_track, zero_start = mets_to_track or [], zero_start or []
        parameters, bad_data_timesteps = parameters or {}, bad_data_timesteps or {}
        self.parameters.update({
            "timestep_hr": self.parameters['data_timestep_hr'],  # Simulation timestep magnitude in hours
            "cvct": 1, "cvcf": 1,           # Minimization coefficients of the phenotype conversion to and from the stationary phase.
            "bcv": 1,                       # The highest fraction of species biomass that can change phenotypes in a timestep
            "cvmin": 0,                     # The lowest value the limit on phenotype conversion goes,
            "v": 0.1,                       # The kinetics constant that is externally adjusted
            'carbon_sources': ['cpd00136', 'cpd00179'],  # 4hb, maltose
            'diffpos': 1, 'diffneg': 1, # diffpos and diffneg coefficients that weight difference between experimental and predicted biomass
        })
        self.parameters.update(parameters)
        self.simulation_timesteps = list(map(str, range(1, int(self.simulation_time/self.parameters['timestep_hr'])+1)))
        # self.problem = Model()
        objective = tupObjective("minimize variance and phenotypic transitions", [], "min")

        # refine the applicable range of bad_data_timesteps
        if bad_data_timesteps:  # {trial:[times]}
            for trial in bad_data_timesteps:
                if ':' in bad_data_timesteps[trial]:
                    start, end = bad_data_timesteps[trial].split(':')
                    start = int(start or self.simulation_timesteps[0])
                    end = int(end or self.simulation_timesteps[-1])
                    bad_data_timesteps[trial] = list(map(str, list(range(start, end+1))))
            if '*' in bad_data_timesteps:
                for time in bad_data_timesteps['*']:
                    self.simulation_timesteps.remove(time)

        # construct the problem
        constraints, variables = [], []
        time_1 = process_time()
        for signal, parsed_df in self.dataframes.items():
            for met in self.phenos_tup.index:
                met_id = self._met_id_parser(met)
                if not mets_to_track and met_id != 'cpd00001' or met_id in mets_to_track:
                    self.variables["c_"+met] = {} ; self.constraints['dcc_'+met] = {}
                    for time in self.simulation_timesteps:
                        self.variables["c_"+met][time] = {} ; self.constraints['dcc_'+met][time] = {}
                        for trial in parsed_df.index:
                            ## define biomass measurement conversion variables
                            conc_var = tupVariable(_name("c_", met, time, trial))
                            ## constrain initial time concentrations to the media or a large default
                            if time == self.simulation_timesteps[0] and not 'bio' in met_id:
                                initial_val = 100 if met_id not in self.media_conc else self.media_conc[met_id]
                                initial_val = 0 if met_id in zero_start else initial_val
                                # !!! Is the circumstance of both trues desirable, or should this be elifs?
                                if dict_keys_exists(self.carbon_conc['rows'], met_id, trial[0]):
                                    initial_val = self.carbon_conc['rows'][met_id][trial[0]]
                                if dict_keys_exists(self.carbon_conc['columns'], met_id, trial[1:]):
                                    initial_val = self.carbon_conc['columns'][met_id][trial[1:]]
                                conc_var = conc_var._replace(bounds=Bounds(initial_val, initial_val))
                            ## mandate complete carbon consumption
                            if time == self.simulation_timesteps[-1] and met_id in self.parameters['carbon_sources']:
                                final_bound = self.variables["c_" + met]["1"][trial].bounds.lb*final_rel_c12_conc
                                conc_var = conc_var._replace(bounds=Bounds(0, final_bound))
                            self.variables["c_" + met][time][trial] = conc_var
                            variables.append(self.variables["c_"+met][time][trial])
            break   # prevents duplicated variables
        for signal in [signal for signal in self.dataframes if 'OD' not in signal]:
            for phenotype in self.phenos_tup.columns:
                if self.signal_species[signal] in phenotype:
                    self.constraints['dbc_'+phenotype] = {time:{} for time in self.simulation_timesteps}

        # define growth and biomass variables and constraints
        for pheno in self.phenos_tup.columns:
            self.variables['cvt_'+pheno] = {} ; self.variables['cvf_'+pheno] = {}
            self.variables['b_'+pheno] = {} ; self.variables['g_'+pheno] = {}
            self.variables['v_'+pheno] = {}
            self.constraints['gc_'+pheno] = {} ; self.constraints['cvc_'+pheno] = {}
            for time in self.simulation_timesteps:
                self.variables['cvt_'+pheno][time] = {} ; self.variables['cvf_'+pheno][time] = {}
                self.variables['b_'+pheno][time] = {} ; self.variables['g_'+pheno][time] = {}
                self.variables['v_'+pheno][time] = {}
                self.constraints['gc_'+pheno][time] = {} ; self.constraints['cvc_'+pheno][time] = {}
                for trial in self.trials:
                    # predicted biomass abundance and biomass growth
                    self.variables['b_'+pheno][time][trial] = tupVariable(_name("b_", pheno, time, trial), Bounds(0, 100))
                    self.variables['g_'+pheno][time][trial] = tupVariable(_name("g_", pheno, time, trial))
                    variables.extend([self.variables['b_' + pheno][time][trial], self.variables['g_' + pheno][time][trial]])

                    if 'stationary' not in pheno:
                        # the conversion rates to and from the stationary phase
                        self.variables['cvt_'+pheno][time][trial] = tupVariable(_name("cvt_", pheno, time, trial), Bounds(0, 100))
                        self.variables['cvf_'+pheno][time][trial] = tupVariable(_name("cvf_", pheno, time, trial), Bounds(0, 100))
                        variables.extend([self.variables['cvf_' + pheno][time][trial], self.variables['cvt_' + pheno][time][trial]])

                        # cvt <= bcv*b_{pheno} + cvmin
                        self.constraints['cvc_'+pheno][time][trial] = tupConstraint(
                            _name('cvc_', pheno, time, trial), (0, None), {
                                "elements": [
                                    self.parameters['cvmin'],
                                    {"elements": [self.parameters['bcv'],
                                                  self.variables['b_'+pheno][time][trial].name],
                                     "operation": "Mul"},
                                    {"elements": [
                                        -1, self.variables['cvt_' + pheno][time][trial].name],
                                     "operation": "Mul"}],
                                "operation": "Add"
                            })
                        # g_{pheno} = b_{pheno}*v
                        self.constraints['gc_'+pheno][time][trial] = tupConstraint(
                            name=_name('gc_', pheno, time, trial),
                            expr={
                                "elements": [
                                    self.variables['g_' + pheno][time][trial].name,
                                    {"elements": [-1, self.parameters['v'],
                                                  self.variables['b_'+pheno][time][trial].name],
                                     "operation": "Mul"}],
                                "operation": "Add"
                            })
                        constraints.extend([self.constraints['cvc_' + pheno][time][trial], self.constraints['gc_' + pheno][time][trial]])
                        objective.expr.extend([{
                            "elements": [
                                {"elements": [self.parameters['cvcf'], self.variables['cvf_'+pheno][time][trial].name],
                                 "operation": "Mul"},
                                {"elements": [self.parameters['cvct'], self.variables['cvt_' + pheno][time][trial].name],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        }])

        # define the concentration constraint
        half_dt = self.parameters['data_timestep_hr']/2
        time_2 = process_time()
        print(f'Done with concentrations and biomass loops: {(time_2-time_1)/60} min')
        for r_index, met in enumerate(self.phenos_tup.index):
            met_id = self._met_id_parser(met)
            if not mets_to_track and met_id != 'cpd00001' or met_id in mets_to_track:
                for trial in list(self.dataframes.values())[0].index:
                    for time in self.simulation_timesteps[:-1]:
                        # c_{met} + dt/2*sum_k^K(n_{k,met} * (g_{pheno}+g+1_{pheno} ) = c+1_{met}
                        next_time = str(int(time) + 1)
                        growth_phenos = [[self.variables['g_'+pheno][next_time][trial].name,
                             self.variables['g_'+pheno][time][trial].name] for pheno in self.phenos_tup.columns]
                        self.constraints['dcc_'+met][time][trial] = tupConstraint(
                            name=_name("dcc_", met, time, trial),
                            expr={
                                "elements": [
                                    self.variables["c_"+met][time][trial].name,
                                    {"elements": [-1, self.variables["c_"+met][next_time][trial].name],
                                     "operation": "Mul"},
                                    *OptlangHelper.dot_product(growth_phenos,
                                        heuns_coefs=half_dt*self.phenos_tup.values[r_index])],
                                "operation": "Add"
                            })
                        constraints.append(self.constraints['dcc_'+met][time][trial])

        time_3 = process_time()
        print(f'Done with DCC loop: {(time_3-time_2)/60} min')
        for signal, parsed_df in self.dataframes.items():
            data_timestep = 1
            self.variables[signal+'__conversion'] = tupVariable(signal+'__conversion')
            variables.append(self.variables[signal+'__conversion'])
            self.variables[signal+'__bio'] = {} ; self.variables[signal+'__diffpos'] = {}
            self.variables[signal+'__diffneg'] = {}
            self.constraints[signal+'__bioc'] = {} ; self.constraints[signal+'__diffc'] = {}
            for time in self.simulation_timesteps[:-1]:
                ## the user timestep and data timestep must be synchronized
                if int(time)*self.parameters['timestep_hr'] >= data_timestep*self.parameters['data_timestep_hr']:
                    data_timestep += 1
                    if int(data_timestep) > self.data_timesteps:
                        break
                    next_time = str(int(time)+1)
                    self.variables[signal+'__bio'][time] = {} ; self.variables[signal+'__diffpos'][time] = {}
                    self.variables[signal+'__diffneg'][time] = {}
                    self.constraints[signal+'__bioc'][time] = {} ; self.constraints[signal+'__diffc'][time] = {}
                    for r_index, trial in enumerate(parsed_df.index):
                        if not dict_keys_exists(bad_data_timesteps, trial, time):
                            total_biomass, signal_sum, from_sum, to_sum = [], [], [], []
                            for pheno_index, pheno in enumerate(self.phenos_tup.columns):
                                # define the collections of signal and pheno terms
                                val = 1 if 'OD' in signal else self.species_phenos_df.loc[signal, pheno]
                                val = val if isnumber(val) else val.values[0]
                                signal_sum.append({"operation": "Mul", "elements": [
                                    -1, val, self.variables["b_"+pheno][time][trial].name]})
                                # total_biomass.append(self.variables["b_"+pheno][time][trial].name)
                                if all(['OD' not in signal, self.signal_species[signal] in pheno, 'stationary' not in pheno]):
                                    from_sum.append({"operation": "Mul", "elements": [
                                        -val, self.variables["cvf_" + pheno][time][trial].name]})
                                    to_sum.append({"operation": "Mul", "elements": [
                                        val, self.variables["cvt_"+pheno][time][trial].name]})
                            for pheno in self.phenos_tup.columns:
                                if 'OD' not in signal and self.signal_species[signal] in pheno:
                                    if "stationary" in pheno:
                                        # b_{phenotype} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) = b+1_{phenotype}
                                        self.constraints['dbc_'+pheno][time][trial] = tupConstraint(
                                            name=_name("dbc_", pheno, time, trial),
                                            expr={
                                                "elements": [
                                                    self.variables['b_'+pheno][time][trial].name,
                                                    *from_sum, *to_sum,
                                                    {"elements": [-1, self.variables["b_"+pheno][next_time][trial].name],
                                                     "operation": "Mul"}],
                                                "operation": "Add"
                                            })
                                    else:
                                        # b_{phenotype} + dt/2*(g_{phenotype} + g+1_{phenotype}) + cvf-cvt = b+1_{phenotype}
                                        self.constraints['dbc_'+pheno][time][trial] = tupConstraint(
                                            name=_name("dbc_", pheno, time, trial),
                                            expr={
                                                "elements": [
                                                    self.variables['b_'+pheno][time][trial].name,
                                                    self.variables['cvf_'+pheno][time][trial].name,
                                                    {"elements": [half_dt, self.variables['g_'+pheno][time][trial].name],
                                                     "operation": "Mul"},
                                                    {"elements": [half_dt, self.variables['g_'+pheno][next_time][trial].name],
                                                     "operation": "Mul"},
                                                    {"elements": [-1, self.variables['cvt_'+pheno][time][trial].name],
                                                     "operation": "Mul"},
                                                    {"elements": [-1, self.variables['b_'+pheno][next_time][trial].name],
                                                     "operation": "Mul"}],
                                                "operation": "Add"
                                            })
                                    constraints.append(self.constraints['dbc_'+pheno][time][trial])

                            self.variables[signal+'__bio'][time][trial] = tupVariable(_name(signal, '__bio', time, trial))
                            self.variables[signal+'__diffpos'][time][trial] = tupVariable(_name(signal, '__diffpos', time, trial), Bounds(0, 100))
                            self.variables[signal+'__diffneg'][time][trial] = tupVariable(_name(signal, '__diffneg', time, trial), Bounds(0, 100))
                            variables.extend([self.variables[signal+'__bio'][time][trial],
                                              self.variables[signal+'__diffpos'][time][trial],
                                              self.variables[signal+'__diffneg'][time][trial]])

                            # {signal}__conversion*datum = {signal}__bio
                            self.constraints[signal+'__bioc'][time][trial] = tupConstraint(
                                name=_name(signal, '__bioc', time, trial),
                                expr={
                                    "elements": [
                                        {"elements": [-1, self.variables[signal+'__bio'][time][trial].name],
                                         "operation": "Mul"},
                                        {"elements": [self.variables[signal+'__conversion'].name,
                                                      parsed_df.values[r_index, int(data_timestep)-1]],
                                         "operation": "Mul"}],
                                    "operation": "Add"
                                })

                            # {speces}_bio + {signal}_diffneg-{signal}_diffpos = sum_k^K(es_k*b_{phenotype})
                            self.constraints[signal+'__diffc'][time][trial] = tupConstraint(
                                name=_name(signal, '__diffc', time, trial),
                                expr={
                                    "elements": [
                                        self.variables[signal+'__bio'][time][trial].name, *signal_sum,
                                        self.variables[signal+'__diffneg'][time][trial].name,
                                        {"elements": [-1, self.variables[signal+'__diffpos'][time][trial].name],
                                         "operation": "Mul"}],
                                    "operation": "Add"
                                })
                            constraints.extend([self.constraints[signal+'__bioc'][time][trial],
                                               self.constraints[signal+'__diffc'][time][trial]])

                            objective.expr.extend([{
                                "elements": [
                                    {"elements": [self.parameters['diffpos'], self.variables[signal+'__diffpos'][time][trial].name],
                                     "operation": "Mul"},
                                    {"elements": [self.parameters['diffneg'], self.variables[signal+'__diffneg'][time][trial].name],
                                     "operation": "Mul"}],
                                "operation": "Add"
                            }])

        time_4 = process_time()
        print(f'Done with the dbc & diffc loop: {(time_4-time_3)/60} min')

        # construct the problem
        self.problem = OptlangHelper.define_model("MSCommFitting model", variables, constraints, objective, True)
        print("Solver:", type(self.problem))
        time_5 = process_time()
        print(f'Done with loading the variables, constraints, and objective: {(time_5-time_4)/60} min')

        # print contents
        if export_parameters:
            self.zipped_output.append('parameters.csv')
            DataFrame(data=list(self.parameters.values()), index=list(self.parameters.keys()), columns=['values']).to_csv('parameters.csv')
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
                
    def compute(self, graphs:list=None, export_zip_name=None, figures_zip_name=None, publishing=False):
        self.values = {}
        solution = self.problem.optimize()
        # categorize the primal values by trial and time
        if all(np.array(list(self.problem.primal_values.values())) == 0):
            raise NoFluxError("The simulation lacks any flux.")
        for variable, value in self.problem.primal_values.items():
            if 'conversion' not in variable:
                basename, time, trial = variable.split('-')
                time_hr = int(time)*self.parameters['data_timestep_hr']
                self.values[trial] = default_dict_values(self.values, trial, {})
                self.values[trial][basename] = default_dict_values(self.values[trial], basename, {})
                self.values[trial][basename][time_hr] = value
                
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
        if "optimal" not in solution:
            raise FeasibilityError(f'The solution is sub-optimal, with a {solution} status.')

        # visualize the specified information
        if graphs:
            self.graph(graphs, export_zip_name=figures_zip_name or export_zip_name, publishing=publishing)

    def _add_plot(self, ax, labels, basename, trial, x_axis_split, linestyle="solid"):
        labels.append(basename.split('-')[-1])
        ax.plot(list(self.values[trial][basename].keys()),
                list(self.values[trial][basename].values()),
                label=basename, linestyle=linestyle)
        ax.legend(labels)
        x_ticks = np.around(np.array(list(self.values[trial][basename].keys())), 0)
        ax.set_xticks(x_ticks[::x_axis_split])
        return ax, labels

    def graph(self, graphs, primal_values_filename:str = None, primal_values_zip_path:str = None,
              export_zip_name:str = None, data_timestep_hr:float = 0.163, publishing:bool = False, title:str=None):
        # define the default timestep ratio as 1
        data_timestep_hr = self.parameters.get('data_timestep_hr', data_timestep_hr)
        timestep_ratio = data_timestep_hr/self.parameters.get('timestep_hr', data_timestep_hr)
        if primal_values_filename:
            if primal_values_zip_path:
                with ZipFile(primal_values_zip_path, 'r') as zp:
                    zp.extract(primal_values_filename)
            with open(primal_values_filename, 'r', encoding='utf-8') as primal:
                self.values = json.load(primal)
        
        # plot the content for desired trials
        x_axis_split = int(2/data_timestep_hr/timestep_ratio)
        self.plots = []
        contents = {"biomass":'b', "growth":'g'}
        for graph_index, graph in enumerate(graphs):
            content = contents.get(graph['content'], graph['content'])
            y_label = 'Variable value' ; x_label = 'Time (hr)'
            if any([x in graph['content'] for x in ['total', 'OD', 'all_biomass']]):
                ys = {name:[] for name in self.signal_species.values()}
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
            graph["experimental_data"] = default_dict_values(graph, "experimental_data", False)
            if "species" not in graph or graph['species'] == '*':
                graph['species'] = list(self.signal_species.values())
            if "phenotype" not in graph or graph['phenotype'] == '*':
                graph['phenotype'] = set(chain(*[phenos for species, phenos in self.community_members.items() if species in graph["species"]]))
            print(f"graph_{graph_index}") ; pprint(graph)

            # define figure specifications
            if publishing:
                pyplot.rc('axes', titlesize=20, labelsize=20)
                pyplot.rc('xtick', labelsize=20)
                pyplot.rc('ytick', labelsize=20)
                pyplot.rc('legend', fontsize=18)
            fig, ax = pyplot.subplots(dpi=200, figsize=(11, 7))
            x_ticks = None
            
            # define the figure contents
            for trial, basenames in self.values.items():
                if trial not in graph['trial']:
                    continue
                labels = []
                for basename in basenames:
                    # graph comprehensive overlaid figures of biomass plots
                    if any([x in content for x in ['total', 'all_biomass', 'OD']]):
                        if 'b_' in basename:
                            var_name, species, phenotype = basename.split('_')
                            label = f'{species}_biomass (model)'
                            if publishing:
                                if any([species == species_name for signal, species_name in self.signal_species.items()]):
                                    break
                                if species == 'ecoli':
                                    species_name = 'E. coli'
                                elif species == 'pf':
                                    species_name = 'P. fluorescens'
                                label = f'{species_name} biomass from optimized model'
                            labels.append({species:label})
                            xs = np.array(list(self.values[trial][basename].keys()))
                            vals = np.array(list(self.values[trial][basename].values()))
                            ax.set_xticks(xs[::int(3/data_timestep_hr/timestep_ratio)])
                            if any([x in content for x in ['all_biomass', 'OD']]):
                                ys['OD'].append(vals)
                            if any([x in content for x in ['all_biomass', 'total']]):
                                ys[species].append(vals)
                        if all([graph['experimental_data'], '__bio' in basename,
                                any([content in basename])]):  # TODO - the any() clauses must be expanded to accommodate all_biomass and total
                            print("experimental_data")
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
                            ax, labels = self._add_plot(ax, labels, basename, trial, x_axis_split)
                    # graph an aspect of a specific species across all phenotypes
                    elif all([graph['phenotype'] == '*', content in basename,
                              all([x in basename for x in graph['species']])]):
                        if 'total' in content:  # TODO - this logic appears erroneous by not using _add_plot()
                            labels = [basename]
                            xs = np.array(list(self.values[trial][basename].keys()))
                            ys.append(np.array(list(self.values[trial][basename].values())))
                            ax.set_xticks(x_ticks[::int(3/data_timestep_hr/timestep_ratio)])
                        else:
                            ax, labels = self._add_plot(ax, labels, basename, trial, x_axis_split)
                        # print('species content of all phenotypes')
                    # graph all phenotypes
                    elif any([x in basename for x in graph['phenotype']]):
                        if any([x in basename for x in graph['species']]) and content in basename:
                            linestyle = "solid" if "ecoli" in basename else "dashed"
                            ax, labels = self._add_plot(ax, labels, basename, trial, x_axis_split, linestyle)
                            # print('all content over all phenotypes')
                    # graph media concentration plots
                    elif 'EX_' in basename and content in basename:
                        ax, labels = self._add_plot(ax, labels, basename, trial, x_axis_split)
                        y_label = 'Concentration (mM)'
                        # print('media concentration')

                if labels:  # this flag represents whether a graph was constructed
                    if any([x in content for x in ['OD', 'all_biomass', 'total']]):
                        for name in ys:
                            if ys[name] != []:
                                label = f'{name}_biomass (model)'
                                if publishing:
                                    if name == 'OD':
                                        label = 'Total biomass from optimized model'
                                    else:
                                        if isinstance(labels[-1],dict) and name in labels[-1]:
                                            label = labels[-1][name]
                                ax.plot(xs.astype(np.float32), sum(ys[name]), label=label)
                    phenotype_id = graph['phenotype'] if isinstance(graph['phenotype'], str) else f"{','.join(graph['phenotype'])} phenotypes"
                    species_id = graph["species"] if graph["species"] != '*' and isinstance(graph["species"], str) else 'all species'
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    if len(labels) > 1:
                        ax.legend()
                    if not publishing:
                        if not title:
                            org_content = content if content not in contents.values() else list(
                                contents.keys())[list(contents.values()).index(content)]
                            this_title = f'{org_content} of {species_id} ({phenotype_id}) in the {trial} trial'
                            if "cpd" in content:
                                this_title=f"{org_content} in the {trial} trial"
                            ax.set_title(this_title)
                        else:
                            ax.set_title(title)
                    fig_name = f'{"_".join([trial, species_id, phenotype_id, content])}.jpg'
                    fig.savefig(fig_name)
                    self.plots.append(fig_name)
        
        # export the figures with other simulation content
        if export_zip_name:
            with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
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
            self.problem = Model.from_json(model_to_load)
        
    def change_parameters(self, cvt=None, cvf=None, diff=None, vmax=None, km=None,graphs:list=None,
                          mscomfit_json_path='mscommfitting.json', primal_values_filename:str=None,
                          export_zip_name=None, extract_zip_name=None,final_abs_concs:dict=None,
                          final_rel_c12_conc:float=None, previous_relative_conc:float=None):
        def change_param(param, param_time, param_trial):
            if not isinstance(param, dict):
                return param
            if param_time in param:
                if param_trial in param[param_time]:
                    return param[param_time][param_trial]
                return param[param_time]
            return param['default']

        def change_vmax(vmax):
            for vmax_arg in mscomfit_json['constraints']:  # !!! specify as phenotype-specific, as well as the Km
                vmax_name, vmax_time, vmax_trial = vmax_arg['name'].split('-')
                if 'gc' in vmax_name:
                    vmax_arg['expression']['args'][1]['args'][0]['value'] = change_param(vmax, vmax_time, vmax_trial)

        def universalize(param, met_id, variable):
            new_param = param.copy()
            for time in variable:
                new_param[met_id][time] = {}
                vmax_val = param[met_id] if not isinstance(vmax_val, dict) else vmax_val[time]
                for trial in variable[time]:
                    vmax_val = vmax_val if not isinstance(vmax_val, dict) else vmax_val[trial]
                    new_param[met_id][time][trial] = vmax_val
            return new_param

        # load the model JSON
        vmax, km = vmax or {}, km or {}
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
        if any([final_rel_c12_conc, final_abs_concs, vmax]):
            # uploads primal values when they are not in RAM
            if not hasattr(self, 'values'):
                with open(primal_values_filename, 'r') as pv:
                    self.values = json.load(pv)
            initial_concentrations = {} ; already_constrained = []
            for var in mscomfit_json['variables']:
                if 'cpd' not in var['name']:
                    continue
                met = var.copy()
                met_name, time, trial = met['name'].split('-')
                # assign initial concentration
                if time == self.simulation_timesteps[0]:
                    initial_concentrations[met_name] = met["ub"]
                    print("initial_concentrations", initial_concentrations[met_name])
                # assign final concentration
                elif time == self.simulation_timesteps[-1]:
                    if final_abs_concs and dict_keys_exists(final_abs_concs, met_name):
                        met['lb'] = met['ub'] = final_abs_concs[met_name]
                    elif final_rel_c12_conc and any([x in met_name for x in self.parameters['carbon_sources']]):
                        print("ub 1", met['ub'])
                        met['lb'] = met['ub'] = initial_concentrations[met_name]*final_rel_c12_conc
                        if previous_relative_conc:
                            met['ub'] /= previous_relative_conc
                            print("ub 2", met['ub'])
                            met['lb'] /= previous_relative_conc
                            print("ub 3", met['lb'])

                if met_name not in already_constrained:
                    already_constrained.append(met_name)
                    # change growth kinetics
                    met_id = self._met_id_parser(met_name)
                    if met_id in list(chain(*self.phenotype_met.values())):
                        # defines the Vmax for each metabolite, or applies a constant Vmax for all instances in the same dict structure
                        vmax = universalize(vmax, met_id, self.variables[met_name
                            ]) if isinstance(vmax[met_id], (float,int)) else vmax
                        # calculate the Michaelis-Menten kinetic rate: vmax / (km + [maltose])
                        if km:  # Vmax=2.2667 & Km=2 to start, given [maltose]=5, which yields 0.3
                            vmax_var, conc_tracker = vmax.copy(), {}
                            print(met_id)
                            count = last_conc_same_count = last_conc = 0
                            while (last_conc_same_count < 5):  # guessed loop threshold
                                error = 0
                                for time in self.variables[met_name]:
                                    time_hr = int(time)*self.parameters['timestep_hr']
                                    conc_tracker[time] = default_dict_values(conc_tracker, time, {})
                                    for trial in self.variables[met_name][time]:
                                        if trial in conc_tracker[time]:
                                            error += (conc_tracker[time][trial]-self.values[trial][met_name][time_hr])**2
                                        conc_tracker[time][trial] = self.values[trial][met_name][time_hr]
                                        vmax_var[met_id][time][trial] /= -(km[met_id]+conc_tracker[time][trial])
                                        print('new growth rate: ', vmax_var[met_id][time][trial])
                                        count += 1
                                last_conc_same_count += 1 if last_conc == conc_tracker[time][trial] else 0
                                last_conc = conc_tracker[time][trial]
                                change_vmax(vmax_var[met_id])
                                # self._export_model_json(mscomfit_json, mscomfit_json_path)
                                self.load_model(model_to_load=mscomfit_json)
                                self.compute(graphs)#, export_zip_name)
                                error = (error/count)**0.5 if error != 0 else 0
                                print("Error:",error)
                        else:
                            change_vmax(vmax[met_id])

        # export and load the edited model
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
