# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:44:07 2022

@author: Andrew Freiburger
"""
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError, ObjectiveError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from modelseedpy.fbapkg.elementuptakepkg import ElementUptakePkg
from modelseedpy.community.mscompatibility import MSCompatibility
from modelseedpy.core.msmodelutl import MSModelUtil
from modelseedpy.core.msminimalmedia import minimizeFlux_withGrowth, bioFlux_check
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Constraint
from optlang.symbolics import Zero
from scipy.constants import hour
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from itertools import chain
from typing import Union, Iterable
from cobra.medium import minimal_medium
from cobra.flux_analysis import pfba
from pprint import pprint
from icecream import ic
# from cplex import Cplex
from math import inf, isclose
import logging, json, os, re
from pandas import read_csv, DataFrame, ExcelFile, read_excel
import numpy as np


import logging
logger = logging.getLogger(__name__)

def isnumber(string):
    try:
        float(string)
    except:
        return False
    return True

def _findDate(string, numerical=False):
    monthNames = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
    monthNums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = list(range(31, 0, -1))  # [f"{num}-" for num in list(range(31,0,-1))]
    years = list(range(2010, 2025))+list(range(10,25))  # [f"-{num}" for num in list(range(2000, 2100))]
    americanDates = [f"{mon}-{day}-{year}" for mon in monthNums for day in days for year in years]

    for date in americanDates:
        if re.search(date, string):
            month, day, year = date.split("-")
            if numerical:
                return "-".join([day, month, year])
            return f"{monthNames[int(month)-1][:3]} {day}, {year}"
    # # determine the month
    # for monName in monthNames:
    #     if re.search(monName, string):
    #         month = monName
    #         break
    # if not month:
    #     for monNum in monthNums:
    #         if re.search(monNum, string):
    #             month = monNum  # maybe should be converted to the Name for standardization
    # # determine the day
    # for dayNum in days:
    #     if re.search(dayNum, string):
    #         day = dayNum
    #         break
    # # determine the year
    # for yearNum in years:
    #     if re.search(yearNum, string):
    #         year = yearNum
    #         break
    # return day+month+year

def dict_keys_exists(dic, *keys):
    if keys[0] in dic:
        remainingKeys = keys[1:]
        if len(remainingKeys) > 0:
            dict_keys_exists(dic[keys[0]], keys[1:])
        return True
    return False

def find_dic_number(dic):
    for k, v in dic.items():
        if isnumber(v):
            return v
        num = find_dic_number(dic[k])
    return num

def default_dict_values(dic, key, default):
    return default if not key in dic else dic[key]

def trial_contents(short_code, indices_tup, values):
    matches = [ele == short_code for ele in indices_tup]
    return np.array(values)[matches]

def _spreadsheet_extension_load(path):
    if ".csv" in path:
        return read_csv(path)
    elif ".xls" in path:
        return ExcelFile(path)

def _spreadsheet_extension_parse(path, raw_data, org_sheet):
    if ".csv" in path:
        return raw_data
    elif ".xls" in path:
        return raw_data.parse(org_sheet)

def _met_id_parser(met):
    met_id = re.sub('(\_\w\d+)', '', met)
    met_id = met_id.replace('EX_', '', 1)
    met_id = met_id.replace('c_', '', 1)
    return met_id

def _column_reduction(org_df):
    dataframe = org_df.copy()  # this prevents an irrelevant warning from pandas
    dataframe.columns = map(str, dataframe.columns)
    dataframe.index = dataframe['Well']
    dataframe.drop('Well', axis=1, inplace=True)
    for col in dataframe.columns:
        if any([x in col for x in ['Plate', 'Well', 'Cycle']]):
            dataframe.drop(col, axis=1, inplace=True)
    dataframe.columns = list(map(int, list(map(float, dataframe.columns))))
    return dataframe

def _remove_trials(org_df, ignore_trials, signal, name, significant_deviation):
    # refine the ignore_trials parameter
    if isinstance(ignore_trials, dict):
        ignore_trials['columns'] = list(map(str, ignore_trials['columns'])) if 'columns' in ignore_trials else []
        ignore_trials['rows'] = list(map(str, ignore_trials['rows'])) if 'rows' in ignore_trials else []
        ignore_trials['wells'] = ignore_trials['wells'] if 'wells' in ignore_trials else []
    elif isIterable(ignore_trials):
        if ignore_trials[0][0].isalpha() and isnumber(ignore_trials[0][1:]):
            short_code = True  # TODO - drop trials with respect to the short codes, and not the full codes

    dataframe = org_df.copy()  # this prevents an irrelevant warning from pandas
    dropped_trials = []
    for trial in dataframe.index:
        if isinstance(ignore_trials, dict) and any(
                [trial[0] in ignore_trials['rows'], trial[1:] in ignore_trials['columns'], trial in ignore_trials['wells']]
        ) or isIterable(ignore_trials) and trial in ignore_trials:
            dataframe.drop(trial, axis=0, inplace=True)
            dropped_trials.append(trial)
        elif isIterable(ignore_trials) and trial in ignore_trials:
            dataframe.drop(trial, axis=0, inplace=True)
            dropped_trials.append(trial)
    removed_trials = []
    if 'OD' not in signal:
        for trial, row in dataframe.iterrows():
            row_array = np.array(row.to_list())
            ## remove trials for which the biomass growth did not change by the determined minimum deviation
            if row_array[-1] / row_array[0] < significant_deviation:
                dataframe.drop(trial, axis=0, inplace=True)
                removed_trials.append(trial)
        if removed_trials:
            print(f'The {removed_trials} trials were removed from the {name} measurements, '
                  f'with their deviation over time being less than the threshold of {significant_deviation}.')
    if dropped_trials:
        print(f'The {dropped_trials} trials were dropped from the {name} measurements '
              'per the ignore_trials parameter.')
    return dataframe, dropped_trials+removed_trials

def _check_plateau(org_df, signal, name, significant_deviation, timesteps_len):
    significant_deviation = max([2, significant_deviation])
    dataframe = org_df.copy()  # this prevents an irrelevant warning from pandas
    dropped = []
    for trial, row in dataframe.iterrows():
        row_array = np.array(row.to_list())
        values = []
        tracking = False
        ## remove trials for which the biomass growth did not change by the determined minimum deviation
        for index, val in enumerate(row_array):
            if val / row_array[0] >= significant_deviation or tracking:
                tracking = True
                values.append(val)
                if len(values) > timesteps_len:
                    del values[0]
                remaining_values = list(dataframe.columns[index-timesteps_len+1:])
                if all([len(values) == timesteps_len, values[-1] <= values[0],
                        remaining_values[0] <= remaining_values[-1]*1.1]):
                    # the entire plateau, minus the first point of plateau, are removed
                    dropped = remaining_values
                    break
        if dropped:
            break
    if dropped:
        content = f"{name} {signal}" if name != signal else signal
        print(f"The {dropped} timesteps (with {row_array[index-len(values)+1:]} values) were removed "
              f"from the {content} data since the OD plateaued and is no longer valid.")
    return dropped


def _remove_timesteps(org_df, ignore_timesteps, name, signal):
    dataframe = org_df.copy()  # this prevents an irrelevant warning from pandas
    if ignore_timesteps:
        dropped = []
        for col in dataframe:
            if col in ignore_timesteps:
                dataframe.drop(col, axis=1, inplace=True)
                dropped.append(col)
        if dropped == ignore_timesteps:
            print(f"The ignore_timesteps columns were dropped for the {name} {signal} data.")
        else:
            raise ParameterError(f"The ignore_timesteps values {ignore_timesteps} "
                                 f"were unsuccessfully dropped for the {name} {signal} data.")
    return dataframe, ignore_timesteps

def _df_construction(name, df_name, ignore_trials, ignore_timesteps, significant_deviation, dataframes, row_num, buffer_col1=True):
    # refine the DataFrames
    time_df = _column_reduction(dataframes[df_name].iloc[0::2])
    values_df = _column_reduction(dataframes[df_name].iloc[1::2])

    # remove specified data trials
    if ignore_trials:
        values_df, removed_trials = _remove_trials(
            values_df, ignore_trials, df_name, name, significant_deviation)
        for row in removed_trials:
            time_df.drop(row, axis=0, inplace=True)

    # remove specified data timesteps
    if ignore_timesteps:
        values_df, removed_timesteps = _remove_timesteps(
            values_df, ignore_timesteps, name, df_name)
        for col in list(map(int, removed_timesteps)):
            time_df.drop(col, axis=1, inplace=True)

    # remove undefined trials
    if buffer_col1:
        possible_rows = [chr(ord("A")+row) for row in range(1, row_num+1)]
        for trial_code in values_df.index:
            if trial_code[0] not in possible_rows:
                values_df.drop(trial_code, axis=0, inplace=True)
                time_df.drop(trial_code, axis=0, inplace=True)

    # process the data for subsequent operations and optimal efficiency
    values_df.astype(str); time_df.astype(str)
    return time_df, values_df

def _find_culture(string):
    matches = re.findall(r"([A-Z]{2}\+?[A-Z]*)", string)
    return [m for m in matches if not any([x in m for x in ["BIOLOG", "III"]])]

def _process_csv(self, csv_path, index_col):
    self.zipped_output.append(csv_path)
    csv = read_csv(csv_path) ; csv.index = csv[index_col]
    csv.drop(index_col, axis=1, inplace=True)
    csv.astype(str)
    return csv

def add_rel_flux_cons(model, ex, phenoRXN, carbon_ratio, rel_flux=0.2):
    # {ex.id}_uptakeLimit: {net_{carbonous_ex}} >= {net_{carbon_source}}*{rel_flux}*{carbon_ratio}
    #  The negative flux sign of influxes specifies that the carbon_source value must be lesser than the other
    #  carbon influx that is being constrained.
    cons = Constraint(Zero, lb=0, ub=None, name=f"{ex.id}_uptakeLimit")
    model.add_cons_vars(cons)
    cons.set_linear_coefficients({
            ex.forward_variable:1, ex.reverse_variable:-1,
            phenoRXN.forward_variable:-rel_flux*carbon_ratio, phenoRXN.reverse_variable:rel_flux*carbon_ratio})
    return model, cons


class GrowthData:

    @staticmethod
    def process(community_members: dict, base_media=None, solver: str = 'glpk',
                data_paths: dict = None, species_abundances: str = None, carbon_conc_series: dict = None,
                ignore_trials: Union[dict, list] = None, ignore_timesteps: list = None, species_identities_rows=None,
                significant_deviation: float = 2, extract_zip_path: str = None):
        row_num = len(species_identities_rows)
        if "rows" in carbon_conc_series and carbon_conc_series["rows"]:
            row_num = len(list(carbon_conc_series["rows"].values())[0])
        (media_conc, zipped_output, data_timestep_hr, simulation_time, dataframes, trials, fluxes_df) = GrowthData.load_data(
            base_media, community_members, solver, data_paths, ignore_trials,
            ignore_timesteps, significant_deviation, row_num, extract_zip_path
        )
        experimental_metadata, standardized_carbon_conc, trial_name_conversion = GrowthData.metadata(
            base_media, community_members, species_abundances, carbon_conc_series,
            species_identities_rows, row_num, _findDate(data_paths["path"])
        )
        # invert the trial_name_conversion keys and values
        # for row in trial_name_conversion:
        #     short_code_ID = {contents[0]:contents[1] for contents in trial_name_conversion[row].values()}
        growth_dfs = GrowthData.data_process(dataframes, trial_name_conversion)
        return (experimental_metadata, growth_dfs, fluxes_df, standardized_carbon_conc,
                trial_name_conversion, np.mean(data_timestep_hr), simulation_time, media_conc)

    @staticmethod
    def phenotypes(base_media, community_members, solver="glpk"):
        # log information of each respective model
        named_community_members = {content["name"]: list(content["phenotypes"].keys()) + ["stationary"]
                                   for member, content in community_members.items()}
        models = OrderedDict()
        solutions = []
        media_conc = set()
        # calculate all phenotype profiles for all members
        for org_model, content in community_members.items():  # community_members excludes the stationary phenotype
            org_model.solver = solver
            model_util = MSModelUtil(org_model)
            model_util.standard_exchanges()
            models[org_model.id] = {"exchanges": model_util.exchange_list(), "solutions": {},
                                    "name": content["name"], "phenotypes": named_community_members[content["name"]]}
            for pheno, pheno_cpds in content['phenotypes'].items():
                pheno_util = MSModelUtil(org_model)
                pheno_util.model.solver = solver
                # pheno_util.compatibilize()   only for non-ModelSEED models
                ## define the media and uptake fluxes, which are 100 except for O_2, where its inclusion is interpreted as an aerobic model
                # if base_media:
                #     # pheno_util.model.medium = {"EX_" + cpd.id + "_e0": abs(cpd.minFlux) for cpd in base_media.mediacompounds}
                #     pheno_util.add_medium({"EX_" + cpd.id + "_e0": abs(cpd.minFlux) for cpd in base_media.mediacompounds})
                #     media_conc = {cpd.id: cpd.concentration for cpd in base_media.mediacompounds}
                # else:
                #     # pheno_util.model.medium = minimal_medium(pheno_util.model, minimize_components=True)
                #     pheno_util.add_medium(minimal_medium(pheno_util.model, minimize_components=True))
                #     media_conc.update(list(pheno_util.model.medium.keys()))
                # pheno_util.model.medium = {cpd: 100 for cpd, flux in pheno_util.model.medium.items()}

                ## The default complete media will be used until this method is developed and ready for refinement
                phenoRXNs = [pheno_util.model.reactions.get_by_id("EX_"+pheno_cpd+"_e0")
                             for pheno_cpd in pheno_cpds["consumed"]]
                media = {cpd: 100 for cpd, flux in pheno_util.model.medium.items()}
                media.update({phenoRXN.id: 1000 for phenoRXN in phenoRXNs})
                ### eliminate hydrogen absorption
                media.update({"EX_cpd11640_e0": 0})
                pheno_util.add_medium(media)
                if "pf " in content["name"]:
                    pheno_util.model.reactions.EX_r277_e0.lower_bound = pheno_util.model.reactions.EX_r1423_e0.lower_bound = 0
                ### define an oxygen absorption relative to the phenotype carbon source
                # O2_consumption: EX_cpd00007_e0 <= sum(primary carbon fluxes)    # formerly <= 2 * sum(primary carbon fluxes)
                coef = {phenoRXN.reverse_variable:1 for phenoRXN in phenoRXNs}
                # coef.update({phenoRXN.forward_variable: 1 for phenoRXN in phenoRXNs})
                coef.update({pheno_util.model.reactions.get_by_id("EX_cpd00007_e0").reverse_variable:-1})
                FBAHelper.create_constraint(
                    pheno_util.model, Constraint(Zero, lb=0, ub=None, name="EX_cpd00007_e0_limitation"), coef=coef)

                col = content["name"] + '_' + pheno
                ## minimize the influx of all non-phenotype compounds at a fixed biomass growth
                ### Penalization of only uptake.
                min_growth = .1
                FBAHelper.add_minimal_objective_cons(pheno_util.model, min_growth)
                FBAHelper.add_objective(pheno_util.model, sum([
                    ex.reverse_variable for ex in pheno_util.carbon_exchange_list() if ex not in phenoRXNs]), "min")
                # with open("minimize_cInFlux.lp", 'w') as out:
                #     out.write(pheno_util.model.solver.to_lp())
                sol = pheno_util.model.optimize()
                bioFlux_check(pheno_util.model, sol)
                ### parameterize the optimization fluxes as lower bounds of the net flux, without exceeding the upper_bound
                for ex in pheno_util.carbon_exchange_list():
                    if ex not in phenoRXNs:
                        ex.reverse_variable.ub = abs(min(0, sol.fluxes[ex.id]))
                # print(sol.status, sol.objective_value, [(ex.id, ex.bounds) for ex in pheno_util.exchange_list()])

                ## maximize the phenotype yield with the previously defined growth and constraints
                obj = [pheno_util.model.reactions.get_by_id("EX_"+pheno_cpd+"_e0").reverse_variable
                       for pheno_cpd in pheno_cpds["consumed"]]
                FBAHelper.add_objective(pheno_util.model, sum(obj), "min")
                # with open("maximize_phenoYield.lp", 'w') as out:
                #     out.write(pheno_util.model.solver.to_lp())
                sol = pheno_util.model.optimize()
                bioFlux_check(pheno_util.model, sol)
                for phenoRXN in phenoRXNs:
                    phenoRXN.lower_bound = phenoRXN.upper_bound = sol.fluxes[phenoRXN.id]

                ## maximize excretion in phenotypes where the excreta is known
                if "excreted" in pheno_cpds:
                    obj = sum([pheno_util.model.reactions.get_by_id("EX_"+excreta+"_e0").flux_expression
                               for excreta in pheno_cpds["excreted"]])
                    FBAHelper.add_objective(pheno_util.model, direction="max", objective=obj)
                    # with open("maximize_excreta.lp", 'w') as out:
                    #     out.write(pheno_util.model.solver.to_lp())
                    sol = pheno_util.model.optimize()
                    bioFlux_check(pheno_util.model, sol)
                    for excreta in pheno_cpds["excreted"]:
                        excretaEX = pheno_util.model.reactions.get_by_id("EX_"+excreta+"_e0")
                        excretaEX.lower_bound = excretaEX.upper_bound = sol.fluxes["EX_"+excreta+"_e0"]

                ## minimize flux of the total simulation flux through pFBA
                try:   # TODO discover why the Pseudomonas 4HB phenotype fails this assessment
                    sol = pfba(pheno_util.model)
                except Exception as e:
                    print(f"The model {pheno_util.model} is unable to be simulated "
                          f"with pFBA and yields a < {e} > error.")
                sol_dict = FBAHelper.solution_to_variables_dict(sol, pheno_util.model)
                simulated_growth = sum([flux for var, flux in sol_dict.items() if re.search(r"(^bio\d+$)", var.name)])
                if not isclose(simulated_growth, min_growth):
                    raise ObjectiveError(f"The assigned minimal_growth of {min_growth} was not optimized"
                                         f" during the simulation, where the observed growth was {simulated_growth}.")
                pheno_influx = sum([sol.fluxes["EX_"+pheno_cpd+"_e0"] for pheno_cpd in pheno_cpds["consumed"]])

                ## normalize the fluxes to -1 for the influx of each phenotype's respective source
                if pheno_influx >= 0:
                    raise NoFluxError(f"The (+) net phenotype flux of {pheno_influx} indicates "
                                      f"implausible phenotype specifications.")
                # sol.fluxes /= abs(pheno_influx)
                models[pheno_util.model.id]["solutions"][col] = sol
                solutions.append(models[pheno_util.model.id]["solutions"][col].objective_value)

        # construct the parsed table of all exchange fluxes for each phenotype
        cols = {}
        ## biomass row
        cols["rxn"] = ["bio"]
        for content in models.values():
            for phenotype in content["phenotypes"]:
                col = content["name"] + '_' + phenotype
                cols[col] = [0]
                if col not in content["solutions"]:
                    continue
                bio_rxns = [x for x in content["solutions"][col].fluxes.index if "bio" in x]
                flux = np.mean(
                    [content["solutions"][col].fluxes[rxn] for rxn in bio_rxns if content["solutions"][col].fluxes[rxn] != 0])
                cols[col] = [flux]
        ## exchange reactions rows
        looped_cols = cols.copy()
        looped_cols.pop("rxn")
        for content in models.values():
            for ex_rxn in content["exchanges"]:
                cols["rxn"].append(ex_rxn.id)
                for col in looped_cols:
                    ### reactions that are not present in the columns are ignored
                    flux = 0 if (
                            col not in content["solutions"] or
                            ex_rxn.id not in list(content["solutions"][col].fluxes.index)
                    ) else content["solutions"][col].fluxes[ex_rxn.id]
                    cols[col].append(flux)
        ## construct the DataFrame
        fluxes_df = DataFrame(data=cols)
        fluxes_df.index = fluxes_df['rxn']
        fluxes_df.drop('rxn', axis=1, inplace=True)
        fluxes_df = fluxes_df.groupby(fluxes_df.index).sum()
        fluxes_df = fluxes_df.loc[(fluxes_df != 0).any(axis=1)]
        fluxes_df.astype(str)
        fluxes_df.to_csv("fluxes.csv")
        return fluxes_df, media_conc

    @staticmethod
    def load_data(base_media, community_members, solver, data_paths, ignore_trials, ignore_timesteps,
                  significant_deviation, row_num, extract_zip_path, min_timesteps=False):
        # define default values
        significant_deviation = significant_deviation or 0
        data_paths = data_paths or {}
        ignore_timesteps = ignore_timesteps or "0:0"
        start, end = ignore_timesteps.split(':')
        raw_data = _spreadsheet_extension_load(data_paths['path'])
        for org_sheet, name in data_paths.items():
            if org_sheet == 'path':
                continue
            df = _spreadsheet_extension_parse(
                data_paths['path'], raw_data, org_sheet)
            df.columns = df.iloc[6]
            df.drop(df.index[:7], inplace=True)
            ## acquire the default start and end indices of ignore_timesteps
            start = int(start or df.columns[0])
            end = int(end or df.columns[-1])
            break
        ignore_timesteps = list(range(start, end+1)) if start != end else None
        zipped_output = []
        if extract_zip_path:
            with ZipFile(extract_zip_path, 'r') as zp:
                zp.extractall()

        # define only species for which data is defined
        fluxes_df, media_conc = GrowthData.phenotypes(base_media, community_members, solver)
        zipped_output.append("fluxes.csv")
        modeled_species = list(v for v in data_paths.values() if ("OD" not in v and " " not in v))
        removed_phenotypes = [col for col in fluxes_df if not any([species in col for species in modeled_species])]
        for col in removed_phenotypes:
            fluxes_df.drop(col, axis=1, inplace=True)
        if removed_phenotypes:
            print(f'The {removed_phenotypes} phenotypes were removed '
                  f'since their species is not among those with data: {modeled_species}.')

        # import and parse the raw CSV data
        data_timestep_hr = []
        dataframes = {}
        zipped_output.append(data_paths['path'])
        max_timestep_cols = []
        if min_timesteps:
            for org_sheet, name in data_paths.items():
                if org_sheet == 'path' or "OD" in sheet:
                    continue
                ## define the DataFrame
                sheet = org_sheet.replace(' ', '_')
                df_name = f"{name}:{sheet}"
                dataframes[df_name] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[df_name].columns = dataframes[df_name].iloc[6]
                dataframes[df_name].drop(dataframes[df_name].index[:7], inplace=True)
                ## parse the timesteps from the DataFrame
                drop_timestep_range = GrowthData._min_significant_timesteps(
                    dataframes[df_name], ignore_timesteps, significant_deviation, ignore_trials, df_name, name)
                max_timestep_cols.append(drop_timestep_range)
            max_cols = max(list(map(len, max_timestep_cols)))
            ignore_timesteps = [x for x in max_timestep_cols if len(x) == max_cols][0]

        # remove trials for which the OD has plateaued
        for org_sheet, name in data_paths.items():
            if "OD" not in name:
                continue
            ## load the OD DataFrame
            sheet = org_sheet.replace(' ', '_')
            df_name = f"{name}:{sheet}"
            dataframes[df_name] = _spreadsheet_extension_parse(
                data_paths['path'], raw_data, org_sheet)
            dataframes[df_name].columns = dataframes[df_name].iloc[6]
            dataframes[df_name].drop(dataframes[df_name].index[:7], inplace=True)
            ## process the OD DataFrame
            data_times_df, data_values_df = _df_construction(
                name, df_name, ignore_trials, ignore_timesteps, significant_deviation, dataframes, row_num)
            plateaued_times = _check_plateau(
                data_values_df, name, name, significant_deviation, 3)
            ## define and store the final DataFrames
            for col in plateaued_times:
                if col in data_times_df.columns:
                    data_times_df.drop(col, axis=1, inplace=True)
                if col in data_values_df.columns:
                    data_values_df.drop(col, axis=1, inplace=True)
            dataframes[df_name] = (data_times_df, data_values_df)
            break

        # refine the non-OD signals
        for org_sheet, name in data_paths.items():
            if org_sheet == 'path' or "OD" in name:
                continue
            sheet = org_sheet.replace(' ', '_')
            df_name = f"{name}:{sheet}"
            if df_name not in dataframes:
                dataframes[df_name] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[df_name].columns = dataframes[df_name].iloc[6]
                dataframes[df_name].drop(dataframes[df_name].index[:7], inplace=True)
            # parse the DataFrame for values
            simulation_time = dataframes[df_name].iloc[0, -1] / hour
            data_timestep_hr.append(simulation_time / int(dataframes[df_name].columns[-1]))
            # define the times and data
            data_times_df, data_values_df = _df_construction(
                name, df_name, ignore_trials, ignore_timesteps, significant_deviation, dataframes, row_num)
            # display(data_times_df) ; display(data_values_df)
            for col in plateaued_times:
                if col in data_times_df.columns:
                    data_times_df.drop(col, axis=1, inplace=True)
                if col in data_values_df.columns:
                    data_values_df.drop(col, axis=1, inplace=True)
            dataframes[df_name] = (data_times_df, data_values_df)

        # differentiate the phenotypes for each species
        trials = set(chain.from_iterable([list(df.index) for df, times in dataframes.values()]))
        return (media_conc, zipped_output, data_timestep_hr, simulation_time, dataframes, trials, fluxes_df)

    @staticmethod
    def _min_significant_timesteps(full_df, ignore_timesteps, significant_deviation, ignore_trials, df_name, name):
        # refine the DataFrames
        values_df = _column_reduction(full_df.iloc[1::2])
        values_df = _remove_trials(values_df, ignore_trials, df_name, name, significant_deviation)
        timestep_range = list(set(list(values_df.columns)) - set(ignore_timesteps))
        start, end = ignore_timesteps[0], ignore_timesteps[-1]
        start_index = list(values_df.columns).index(start)
        end_index = list(values_df.columns).index(end)
        ## adjust the customized range such that the threshold is reached.
        for trial, row in values_df.iterrows():
            row_array = np.delete(np.array(row.to_list()), list(range(start_index, end_index + 1)))
            ## remove trials for which the biomass growth did not change by the determined minimum deviation
            while all([row_array[-1] / row_array[0] < significant_deviation,
                       end <= values_df.columns[-1], start >= values_df.columns[0]]):
                # print(timestep_range[0], values_df.columns[0], values_df.columns[-1], end, start)
                if timestep_range[0] == values_df.columns[0] and start != values_df.columns[-1]:
                    timestep_range.append(timestep_range[-1] + 1)
                    start += 1
                    print(f"The end boundary for {name} is increased to {timestep_range[-1]}", end="\r")
                elif timestep_range[-1] == values_df.columns[-1] and end != values_df.columns[0]:
                    timestep_range.append(timestep_range[0] - 1)
                    end -= 1
                    print(f"The start boundary for {name} is decreased to {timestep_range[0]}", end="\r")
                else:
                    raise ParameterError(f"All of the timesteps were omitted for {name}.")
                row_array = np.delete(np.array(row.to_list()), list(range(
                    list(values_df.columns).index(start), list(values_df.columns).index(end) + 1)))
            print("\n")
        return list(range(start, end+1))

    @staticmethod
    def metadata(base_media, community_members, species_abundances,
                 carbon_conc, species_identities_rows, row_num, date):
        # define carbon concentrations for each trial
        carbon_conc = carbon_conc or {}
        carbon_conc['columns'] = default_dict_values(carbon_conc, "columns", {})
        carbon_conc['rows'] = default_dict_values(carbon_conc, "rows", {})
        column_num = len(species_abundances)

        # define the metadata DataFrame and a few columns
        constructed_experiments = DataFrame()
        experiment_prefix = "G"
        constructed_experiments["short_code"] = [f"{experiment_prefix}{x+1}" for x in list(range(column_num*row_num))]
        base_media_path = "minimal components media" if not base_media else base_media.path[0]
        constructed_experiments["base_media"] = [base_media_path] * (column_num*row_num)

        # define community content
        species_mets = {}
        for mem in community_members.values():
            species_mets[mem["name"]] = np.array([mets["consumed"] for mets in mem["phenotypes"].values()]).flatten()

        # define the strains column
        strains, additional_compounds, experiment_ids = [], [], []
        trial_name_conversion = {}
        count = 1
        ## apply universal values to all trials
        base_row_conc = [] if '*' not in carbon_conc else [
            ':'.join([met, str(carbon_conc['*'][met][0]), str(carbon_conc['*'][met][1])])
            for met in carbon_conc['*']]
        members = list(mem["name"] for mem in community_members.values())
        for row in range(1, row_num+1):
            row_conc = base_row_conc[:]
            trial_letter = chr(ord("A") + row)
            trial_name_conversion[trial_letter] = {}
            ## add rows where the initial concentration in the first trial is non-zero
            for met, conc_dict in carbon_conc["rows"].items():
                if conc_dict[sorted(list(conc_dict.keys()))[row-1]] > 0:
                    row_conc.append(':'.join([
                        met, str(conc_dict[sorted(list(conc_dict.keys()))[row-1]]),
                        str(conc_dict[sorted(list(conc_dict.keys()), reverse=True)[-row]])]))
            row_concentration = ';'.join(row_conc)
            composition = {}
            for col in range(1, column_num+1):
                ## construct the columns of information
                additional_compounds.append(row_concentration)
                experiment_id = []
                for member in members:
                    ### define the relative community abundances
                    composition[member] = [member, f"r{species_abundances[col][member]}"]
                    ### define the member strain, where it is appropriate
                    if member in species_identities_rows[row]:
                        composition[member][0] += f"_{species_identities_rows[row][member]}"
                    ### the experimental ID is abundance+memberID
                    if int(composition[member][1][1:]) != 0:
                        experiment_id.append(f"{composition[member][1]}_{composition[member][0]}")
                    composition[member] = ':'.join(composition[member])
                strains.append(';'.join(composition[member] for member in members))
                for row2 in row_conc:
                    metID, init, end = row2.split(':')
                    ### get the met_name for the corresponding match in values
                    met_name = None
                    for index, mets in enumerate(species_mets.values()):
                        if metID in mets:
                            met_name = list(species_mets.keys())[index]
                            break
                    if "met_name" not in locals() or not met_name:
                        logger.critical(f"The specified phenotypes {species_mets} for the {members} members does not "
                                        f"include the consumption of the available sources {row_conc}; hence, the model cannot grow.")
                        content = ""
                    else:
                        content = f"{init}_{met_name}"
                    experiment_id.append(content)
                experiment_id = '-'.join(experiment_id)
                experiment_ids.append(experiment_id)
                trial_name_conversion[trial_letter][str(col+1)] = (experiment_prefix+str(count), experiment_id)
                count += 1

        # convert the variable concentrations to short codes
        standardized_carbon_conc = {}
        for met, conc in carbon_conc["rows"].items():
            standardized_carbon_conc[met] = {}
            for row, val in conc.items():
                standardized_carbon_conc[met].update({short_code:val for short_code, expID in trial_name_conversion[row].values()})
        for met, conc in carbon_conc["columns"].items():
            standardized_carbon_conc[met] = default_dict_values(standardized_carbon_conc, met, {})
            for col, val in conc.items():
                for row in trial_name_conversion:
                    standardized_carbon_conc[met][trial_name_conversion[row][str(col)][0]] = val

        # add columns to the exported dataframe
        constructed_experiments.insert(0, "trial_IDs", experiment_ids)
        constructed_experiments["additional_compounds"] = additional_compounds
        constructed_experiments["strains"] = strains
        constructed_experiments["date"] = [date] * (column_num*row_num)
        constructed_experiments.to_csv("growth_metadata.csv")
        return constructed_experiments, standardized_carbon_conc, trial_name_conversion

    @staticmethod
    def data_process(dataframes, trial_name_conversion):
        short_codes, trials_list = [], []
        values, times = {}, {}  # The times must be capture upstream
        first = True
        for df_name, (times_df, values_df) in dataframes.items():
            # print(df_name)
            # display(times_df) ; display(values_df)
            times_tup = FBAHelper.parse_df(times_df)
            values[df_name], times[df_name] = [], []
            for trial_code in values_df.index:
                row_let, col_num = trial_code[0], trial_code[1:]
                # print(trial_code, row_let, col_num)
                for trial_row_values in trial_contents(trial_code, values_df.index, values_df.values):
                    if first:
                        short_code, experimentalID = trial_name_conversion[row_let][col_num]
                        trials_list.extend([experimentalID] * len(values_df.columns))
                        short_codes.extend([short_code] * len(values_df.columns))
                    values[df_name].extend(trial_row_values)
                    times[df_name].append(times_tup.values.flatten())
            first = False
        # process the data to the smallest dataset, to accommodate heterogeneous data sizes
        minVal = min(list(map(len, values.values())))
        for df_name, data in values.items():
            values[df_name] = data[:minVal]
        times2 = times.copy()
        for df_name, data in times2.items():
            times[df_name] = []
            for ls in data:
                times[df_name].append(ls[:minVal])
        times_set = np.mean(list(times.values()), axis=0).flatten()
        # construct the growth DataFrame
        df_data = {"trial_IDs": trials_list[:minVal], "short_codes": short_codes[:minVal]}
        df_data.update({"Time (s)": times_set})  # element-wise average
        df_data.update({df_name:vals for df_name, vals in values.items()})
        growth_df = DataFrame(df_data)
        growth_df.index = growth_df["short_codes"]
        del growth_df["short_codes"]
        growth_df.to_csv("growth_spectra.csv")
        return growth_df


class BiologData:

    @staticmethod
    def process(data_paths, trial_conditions_path, culture=None, date=None, significant_deviation=None, solver="glpk"):
        row_num = 8 ; column_num = 12
        (zipped_output, data_timestep_hr, simulation_time, dataframes, trials, culture, date) = BiologData.load_data(
            data_paths, significant_deviation, row_num, culture, date, solver)
        experimental_metadata, standardized_carbon_conc, trial_name_conversion = BiologData.metadata(
            trial_conditions_path, row_num , column_num, culture, date)
        biolog_df = BiologData.data_process(dataframes, trial_name_conversion)
        return (experimental_metadata, biolog_df, standardized_carbon_conc,
                trial_name_conversion, np.mean(data_timestep_hr), simulation_time)

    @staticmethod
    def load_data(data_paths, significant_deviation, row_num, culture, date, solver):
        zipped_output = [data_paths['path'], "fluxes.csv"]
        # determine the metabolic fluxes for each member and phenotype
        # import and parse the raw CSV data
        # TODO - this may be capable of emulating leveraged functions from the GrowthData object
        data_timestep_hr = []
        dataframes = {}
        raw_data = _spreadsheet_extension_load(data_paths['path'])
        significant_deviation = significant_deviation or 2
        culture = culture or _find_culture(data_paths['path'])
        date = date or _findDate(data_paths['path'])
        for org_sheet, name in data_paths.items():
            if org_sheet == 'path':
                continue
            sheet = org_sheet.replace(" ", "_")
            df_name = f"{name}:{sheet}"
            if df_name not in dataframes:
                dataframes[df_name] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[df_name].columns = dataframes[df_name].iloc[8]
                dataframes[df_name].drop(dataframes[df_name].index[:9], inplace=True)
            # parse the DataFrame for values
            dataframes[df_name].columns = [str(x).strip() for x in dataframes[df_name].columns]
            simulation_time = dataframes[df_name].iloc[0, -1] / hour
            # display(dataframes[df_name])
            data_timestep_hr.append(simulation_time / int(float(dataframes[df_name].columns[-1])))
            # define the times and data
            data_times_df, data_values_df = _df_construction(
                name, df_name, None, None, significant_deviation, dataframes, row_num, False)
            # display(data_times_df) ; display(data_values_df)
            dataframes[df_name] = (data_times_df, data_values_df)

        # differentiate the phenotypes for each species
        trials = set(chain.from_iterable([list(df.index) for df, times in dataframes.values()]))
        return (zipped_output, data_timestep_hr, simulation_time, dataframes, trials, culture, date)

    @staticmethod
    def metadata(trial_conditions_path, row_num, column_num, culture, date):
        # define the conditions for each trial
        with open(trial_conditions_path) as trials:
            trial_conditions = json.load(trials)

        # define the metadata DataFrame and a few columns
        constructed_experiments = DataFrame()
        experiment_prefix = "B"
        constructed_experiments.index = [f"{experiment_prefix}{x+1}" for x in list(range(row_num*column_num))]
        constructed_experiments.index.name = "short_code"

        # define the strains column
        experiment_ids, trial_names = [], []
        trial_name_conversion, trial_mets = {}, {}
        count = 1
        ## apply universal values to all trials
        for row in range(row_num):
            trial_letter = chr(ord("A") + row)
            trial_name_conversion[trial_letter] = {}
            ## add rows where the initial concentration in the first trial is non-zero
            for col in range(1, column_num+1):
                ## construct the columns of information
                dataID = trial_letter+str(col)
                MSID = trial_conditions[dataID]["ModelSEED_ID"]
                short_code = experiment_prefix+str(count)

                experiment_ids.append(MSID)
                trial_names.append(trial_conditions[dataID]["name"])
                trial_name_conversion[trial_letter][str(col)] = (short_code, MSID)
                trial_mets[MSID] = {short_code:trial_conditions[dataID]["mM"]}
                count += 1

        # add columns to the exported dataframe
        constructed_experiments.insert(0, "ModelSEED_ID", experiment_ids)
        constructed_experiments.insert(0, "condition", trial_names)
        constructed_experiments["strain"] = [culture] * (column_num*row_num)
        constructed_experiments["date"] = [date] * (column_num*row_num)
        constructed_experiments.to_csv("growth_metadata.csv")
        return constructed_experiments, trial_mets, trial_name_conversion

    @staticmethod
    def data_process(dataframes, trial_name_conversion):
        short_codes, trials_list = [], []
        values, times = {}, {}  # The times must capture upstream
        first = True
        for df_name, (times_df, values_df) in dataframes.items():
            times_tup = FBAHelper.parse_df(times_df)
            average_times = np.mean(times_tup.values, axis=0)
            values[df_name], times[df_name] = [], []
            for exprID in values_df.index:
                row_let, col_num = exprID[0], exprID[1:]
                for row in trial_contents(exprID, values_df.index, values_df.values):
                    if first:
                        short_code, experimentalID = trial_name_conversion[row_let][col_num]
                        trials_list.extend([experimentalID] * len(values_df.columns))
                        short_codes.extend([short_code] * len(values_df.columns))
                    values[df_name].extend(row)
                    times[df_name].extend(average_times)
            first = False
        df_data = {"trial_IDs": trials_list, "short_codes": short_codes}
        df_data.update({"Time (s)": np.mean(list(times.values()), axis=0)})  # element-wise average
        df_data.update({df_name:vals for df_name, vals in values.items()})
        biolog_df = DataFrame(df_data)
        biolog_df.index = biolog_df["short_codes"]
        del biolog_df["short_codes"]
        biolog_df.to_csv("growth_spectra.csv")

        return biolog_df



#
# # define the environment path
# import os
#
# # local_cobrakbase_path = os.path.join('/Users/afreiburger/Documents')
# local_cobrakbase_path = os.path.join('C:', 'Users', 'Andrew Freiburger', 'Documents', 'Argonne', 'cobrakbase')
# os.environ["HOME"] = local_cobrakbase_path
#
# # import the models
# import cobrakbase
#
# # with open("/Users/afreiburger/Documents/kbase_token.txt") as token_file:
# with open("C:/Users/Andrew Freiburger/Documents/Argonne/kbase_token.txt") as token_file:
#     kbase_api = cobrakbase.KBaseAPI(token_file.readline())
# ecoli = kbase_api.get_from_ws("iML1515", 76994)
#
# import warnings
#
# warnings.filterwarnings(action='once')
#
# from pandas import set_option
#
# set_option("display.max_rows", None)
# graphs_list = [
#     {
#         'trial':'G48',
#         "phenotype": '*',
#         'content': 'biomass',
#         'experimental_data': False
#     },
#     {
#         'trial':'G48',
#         'content': "conc",
#     },
#     {
#         'trial':'G48',
#         "phenotype": '*',
#         "species":["ecoli"],
#         'content': 'biomass'
#     },
#     {
#         'trial':'G48',
#         'content': 'total_biomass',
#         'experimental_data': True
#     }
# ]
#
# growth_data_path="../Jeffs_data/PF-EC 4-29-22 ratios and 4HB changes.xlsx"
# from time import process_time
# time1 = process_time()
# experimental_metadata, growth_df, fluxes_df, standardized_carbon_conc, trial_name_conversion, species_phenos_df, data_timestep_hr, simulation_timestep, media_conc = GrowthData.process(
#     community_members = {
#         kbase_api.get_from_ws("iML1515",76994): {
#             'name': 'ecoli',
#             'phenotypes': {'acetate': {"cpd00029":[-1,-1]}, #kbase_api.get_from_ws('93465/13/1'),
#                         'malt': {"cpd00179":[-1,-1]} #kbase_api.get_from_ws("93465/23/1")} #'93465/9/1')}   # !!! The phenotype name must align with the experimental IDs for the graphs to find the appropriate data
#             }
#         },
#         kbase_api.get_from_ws("iSB1139.kb.gf",30650): {
#             'name': 'pf',
#             'phenotypes': {'acetate': {"cpd00029":[-1,-1]}, # kbase_api.get_from_ws("93465/25/1"), #'93465/11/1'),
#                         '4HB': {"cpd00136":[-1,-1]} # kbase_api.get_from_ws('	93465/27/1')} #93465/15/1')}
#             }
#         }
#     },
#     data_paths = {'path':growth_data_path, 'Raw OD(590)':'OD', 'mNeonGreen':'pf', 'mRuby':'ecoli'},
#     species_abundances = {
#         1:{"ecoli":0, "pf":1},
#         2:{"ecoli":1, "pf":50},
#         3:{"ecoli":1, "pf":20},
#         4:{"ecoli":1, "pf":10},
#         5:{"ecoli":1, "pf":3},
#         6:{"ecoli":1, "pf":1},
#         7:{"ecoli":3, "pf":1},
#         8:{"ecoli":10, "pf":1},
#         9:{"ecoli":20, "pf":1},
#         10:{"ecoli":1, "pf":0},
#         11:{"ecoli":0, "pf":0}
#       },
#     carbon_conc_series = {'rows': {
#         'cpd00136': {'B':0, 'C': 0, 'D': 1, 'E': 1, 'F': 4, 'G': 4},
#         'cpd00179': {'B':5, 'C': 5, 'D':5, 'E': 5, 'F': 5, 'G': 5},
#     }},
#     ignore_trials = {'rows': ['C', 'D', 'E', 'F', 'G'], 'columns': [1,2,3,5,6,7,8,9,10,11,12]},
#     # ignore_timesteps="10:",  # The
#     species_identities_rows = {
#         1:{"ecoli":"mRuby"},
#         2:{"ecoli":"ACS"},
#         3:{"ecoli":"mRuby"},
#         4:{"ecoli":"ACS"},
#         5:{"ecoli":"mRuby"},
#         6:{"ecoli":"ACS"}
#     }
# )
# print(f"{(process_time()-time1)/60} minutes")