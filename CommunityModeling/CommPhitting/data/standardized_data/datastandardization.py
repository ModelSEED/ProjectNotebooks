# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:44:07 2022

@author: Andrew Freiburger
"""
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError, ObjectiveError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from modelseedpy.core.msminimalmedia import minimizeFlux_withGrowth, bioFlux_check
from modelseedpy.fbapkg.elementuptakepkg import ElementUptakePkg
from modelseedpy.core.msmodelutl import MSModelUtil
from modelseedpy.core.fbahelper import FBAHelper
from pandas import read_csv, DataFrame, ExcelFile, read_excel
from cobra.medium import minimal_medium
from cobra.flux_analysis import pfba
# from commscores import GEMCompatibility
from zipfile import ZipFile, ZIP_LZMA
from collections import OrderedDict
from typing import Union, Iterable
from optlang.symbolics import Zero
from scipy.constants import hour
from optlang import Constraint
from math import inf, isclose
import logging, json, os, re
from itertools import chain
from copy import deepcopy
from pprint import pprint
from icecream import ic
# from cplex import Cplex
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

def _df_construction(name, df_name, ignore_trials, ignore_timesteps,
                     significant_deviation, dataframe, row_num, buffer_col1=True):
    # refine the DataFrames
    time_df = _column_reduction(dataframe.iloc[0::2])
    values_df = _column_reduction(dataframe.iloc[1::2])
    # display(name, time_df, values_df)

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

def strip_comp(ID):
    ID = ID.replace("-", "~")
    return re.sub("(\_\w\d)", "", ID)

def reverse_strip_comp(ID):
    return ID.replace("~", "-")

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
    def process(community_members: dict, base_media=None, solver: str = 'glpk', all_phenotypes=True,
                data_paths: dict = None, species_abundances: str = None, carbon_conc_series: dict = None,
                ignore_trials: Union[dict, list] = None, ignore_timesteps: list = None, species_identities_rows=None,
                significant_deviation: float = 2, extract_zip_path: str = None):  #, msdb_path:str=None):
        # define the number of rows in the experimental data
        row_num = len(species_identities_rows)
        if "rows" in carbon_conc_series and carbon_conc_series["rows"]:
            row_num = len(list(carbon_conc_series["rows"].values())[0])
        # load and parse data and metadata
        (media_conc, zipped_output, data_timestep_hr, simulation_time, dataframes, trials, fluxes_df) = GrowthData.load_data(
            base_media, community_members, solver, data_paths, ignore_trials, all_phenotypes,
            ignore_timesteps, significant_deviation, row_num, extract_zip_path
        )
        experimental_metadata, standardized_carbon_conc, trial_name_conversion = GrowthData.metadata(
            base_media, community_members, species_abundances, carbon_conc_series,
            species_identities_rows, row_num, _findDate(data_paths["path"])
        )
        growth_dfs = GrowthData.data_process(dataframes, trial_name_conversion)
        # display(fluxes_df)
        requisite_biomass = GrowthData.biomass_growth(
            carbon_conc_series, fluxes_df, growth_dfs.index.unique(), trial_name_conversion, data_paths,
            community_members if all_phenotypes else None)
        return (experimental_metadata, growth_dfs, fluxes_df, standardized_carbon_conc, requisite_biomass,
                trial_name_conversion, np.mean(data_timestep_hr), simulation_time, media_conc)


    @staticmethod
    def phenotypes(community_members, all_phenotypes=True, solver:str="glpk"):
        # log information of each respective model
        models = OrderedDict()
        solutions = []
        media_conc = set()
        # calculate all phenotype profiles for all members
        comm_members = community_members.copy()
        # print(community_members)
        for org_model, content in community_members.items():  # community_members excludes the stationary phenotype
            print("\n", org_model.id)
            org_model.solver = solver
            model_util = MSModelUtil(org_model, True)
            if "org_coef" not in locals():
                org_coef = {model_util.model.reactions.get_by_id("EX_cpd00007_e0").reverse_variable: -1}
            model_util.standard_exchanges()
            models[org_model.id] = {"exchanges": model_util.exchange_list(), "solutions": {}, "name": content["name"]}
            phenoRXNs = model_util.carbon_exchange_list(include_unknown=False)
            if "phenotypes" in content:
                models[org_model.id]["phenotypes"] = ["stationary"] + [
                    content["phenotypes"].keys() for member, content in comm_members.items()]
                phenoRXNs = [model_util.model.reactions.get_by_id("EX_"+pheno_cpd+"_e0")
                             for pheno, pheno_cpds in content['phenotypes'].items()
                             for pheno_cpd in pheno_cpds["consumed"]]
            for phenoRXN in phenoRXNs:
                # print(phenoRXN.id)
                pheno_util = MSModelUtil(org_model, True)
                pheno_util.model.solver = solver
                media = {cpd: 100 for cpd, flux in pheno_util.model.medium.items()}
                media.update({phenoRXN.id: 1000})
                ### eliminate hydrogen absorption
                media.update({"EX_cpd11640_e0": 0})
                pheno_util.add_medium(media)
                ### define an oxygen absorption relative to the phenotype carbon source
                # O2_consumption: EX_cpd00007_e0 <= sum(primary carbon fluxes)    # formerly <= 2 * sum(primary carbon fluxes)
                coef = org_coef.copy()
                coef.update({phenoRXN.reverse_variable: 1})
                pheno_util.create_constraint(Constraint(Zero, lb=0, ub=None, name="EX_cpd00007_e0_limitation"), coef=coef)

                ## minimize the influx of all non-phenotype compounds at a fixed biomass growth
                ### Penalization of only uptake.
                min_growth = .1
                pheno_util.add_minimal_objective_cons(min_growth)
                pheno_util.add_objective(sum([ex.reverse_variable for ex in pheno_util.carbon_exchange_list()
                                              if ex.id != phenoRXN.id]), "min")
                # with open(f"minimize_cInFlux_{phenoRXN.id}.lp", 'w') as out:
                #     out.write(pheno_util.model.solver.to_lp())
                sol = pheno_util.model.optimize()
                bioFlux_check(pheno_util.model, sol)
                ### parameterize the optimization fluxes as lower bounds of the net flux, without exceeding the upper_bound
                for ex in pheno_util.carbon_exchange_list():
                    if ex.id != phenoRXN.id:
                        ex.reverse_variable.ub = abs(min(0, sol.fluxes[ex.id]))
                # print(sol.status, sol.objective_value, [(ex.id, ex.bounds) for ex in pheno_util.exchange_list()])

                ## maximize the phenotype yield with the previously defined growth and constraints
                pheno_util.add_objective(phenoRXN.reverse_variable, "min")
                # with open(f"maximize_phenoYield_{phenoRXN.id}.lp", 'w') as out:
                #     out.write(pheno_util.model.solver.to_lp())
                sol = pheno_util.model.optimize()
                bioFlux_check(pheno_util.model, sol)
                pheno_influx = sol.fluxes[phenoRXN.id]
                if pheno_influx >= 0:
                    if not all_phenotypes:
                        pprint({rxn: flux for rxn, flux in sol.fluxes.items() if flux != 0})
                        raise NoFluxError(f"The (+) net flux of {pheno_influx} for the {phenoRXN.id} phenotype"
                                          f" indicates that it is an implausible phenotype.")
                    print(f"NoFluxError: The (+) net flux of {pheno_influx} for the {phenoRXN.id}"
                          " phenotype indicates that it is an implausible phenotype.")
                phenoRXN.lower_bound = phenoRXN.upper_bound = sol.fluxes[phenoRXN.id]

                ## maximize excretion in phenotypes where the excreta is known
                met = list(phenoRXN.metabolites)[0]
                if "excretions" in content and met.id in content["excretions"]:
                    obj = sum([pheno_util.model.reactions.get_by_id("EX_" + excreta + "_e0").flux_expression
                               for excreta in content["excretions"][met.id]])
                    pheno_util.add_objective(direction="max", objective=obj)
                    # with open("maximize_excreta.lp", 'w') as out:
                    #     out.write(pheno_util.model.solver.to_lp())
                    sol = pheno_util.model.optimize()
                    bioFlux_check(pheno_util.model, sol)
                    for excreta in content["excretions"][met.id]:
                        excretaEX = pheno_util.model.reactions.get_by_id("EX_" + excreta + "_e0")
                        excretaEX.lower_bound = excretaEX.upper_bound = sol.fluxes["EX_" + excreta + "_e0"]

                ## minimize flux of the total simulation flux through pFBA
                try:  # TODO discover why many phenotypes are infeasible with pFBA
                    sol = pfba(pheno_util.model)
                except Exception as e:
                    print(f"The {phenoRXN.id} phenotype of the {pheno_util.model} model is "
                          f"unable to be simulated with pFBA and yields a < {e} > error.")
                sol_dict = FBAHelper.solution_to_variables_dict(sol, pheno_util.model)
                simulated_growth = sum([flux for var, flux in sol_dict.items() if re.search(r"(^bio\d+$)", var.name)])
                if not isclose(simulated_growth, min_growth):
                    raise ObjectiveError(f"The assigned minimal_growth of {min_growth} was not optimized"
                                         f" during the simulation, where the observed growth was {simulated_growth}.")

                # sol.fluxes /= abs(pheno_influx)
                met_name = strip_comp(met.name).replace(" ", "-")
                col = content["name"] + '_' + met_name
                models[pheno_util.model.id]["solutions"][col] = sol
                solutions.append(models[pheno_util.model.id]["solutions"][col].objective_value)

                ## update the community_members dictionary the defined phenotypes, being either all or a specified few
                # print(community_members)
                met_name = met_name.replace("_", "-").replace("~", "-")
                if all_phenotypes:
                    if "phenotypes" not in comm_members[org_model]:
                        comm_members[org_model]["phenotypes"] = {met_name: {"consumed": [strip_comp(met.id)]}}
                    if met_name not in comm_members[org_model]["phenotypes"]:
                        comm_members[org_model]["phenotypes"].update({met_name: {"consumed": [strip_comp(met.id)]}})
                    else:
                        comm_members[org_model]["phenotypes"][met_name]["consumed"] = [strip_comp(met.id)]
                    if "excretions" in content and strip_comp(met.id) in content["excretions"]:
                        comm_members[org_model]["phenotypes"][met_name].update(
                            {"excreted": content["excretions"][strip_comp(met.id)]})
                # print(community_members)

        # construct the parsed table of all exchange fluxes for each phenotype
        cols = {}
        ## biomass row
        cols["rxn"] = ["bio"]
        for content in models.values():
            for col in content["solutions"]:
                cols[col] = [0]
                if col not in content["solutions"]:
                    continue
                bio_rxns = [x for x in content["solutions"][col].fluxes.index if "bio" in x]
                flux = np.mean([content["solutions"][col].fluxes[rxn] for rxn in bio_rxns
                                if content["solutions"][col].fluxes[rxn] != 0])
                cols[col] = [flux]
        ## exchange reactions rows
        looped_cols = cols.copy()
        looped_cols.pop("rxn")
        for content in models.values():
            for ex_rxn in content["exchanges"]:
                cols["rxn"].append(ex_rxn.id)
                for col in looped_cols:
                    ### reactions that are not present in the columns are ignored
                    flux = 0 if (col not in content["solutions"] or
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
        return fluxes_df, comm_members

    @staticmethod
    def load_data(base_media, community_members, solver, data_paths, ignore_trials, all_phenotypes,
                  ignore_timesteps, significant_deviation, row_num, extract_zip_path, min_timesteps=False):
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
        fluxes_df, comm_members = GrowthData.phenotypes(community_members, all_phenotypes, solver)
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
                name, df_name, ignore_trials, ignore_timesteps, significant_deviation,
                dataframes[df_name], row_num)
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
                name, df_name, ignore_trials, ignore_timesteps, significant_deviation,
                dataframes[df_name], row_num)
            # display(data_times_df) ; display(data_values_df)
            for col in plateaued_times:
                if col in data_times_df.columns:
                    data_times_df.drop(col, axis=1, inplace=True)
                if col in data_values_df.columns:
                    data_values_df.drop(col, axis=1, inplace=True)
            dataframes[df_name] = (data_times_df, data_values_df)

        # differentiate the phenotypes for each species
        trials = set(chain.from_iterable([list(df.index) for df, times in dataframes.values()]))
        media_conc = {} if not base_media else {cpd.id: cpd.concentration for cpd in base_media.mediacompounds}
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
        ex_prefix = "G"
        constructed_experiments["short_code"] = [f"{ex_prefix}{x+1}" for x in list(range(column_num*row_num))]
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
                        logger.critical(f"The specified phenotypes {species_mets} for the {members} members"
                                        f" does not include the consumption of the available sources"
                                        f" {row_conc}; hence, the model cannot grow.")
                        content = ""
                    else:
                        content = f"{init}_{met_name}"
                    experiment_id.append(content)
                experiment_id = '-'.join(experiment_id)
                experiment_ids.append(experiment_id)
                trial_name_conversion[trial_letter][str(col+1)] = (ex_prefix+str(count), experiment_id)
                count += 1

        # convert the variable concentrations to short codes
        standardized_carbon_conc = {}
        for met, conc in carbon_conc["rows"].items():
            standardized_carbon_conc[met] = {}
            for row, val in conc.items():
                standardized_carbon_conc[met].update({short_code:val for (
                    short_code, expID) in trial_name_conversion[row].values()})
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
    def biomass_growth(carbon_conc, fluxes_df, growth_df_trials, trial_name_conversion,
                       data_paths, community_members=None, pheno_info=None):
        # parse inputs
        # if msdb_path:
        #     from modelseedpy.biochem import from_local
        #     msdb = from_local(msdb_path)
        pheno_info = pheno_info or {f"{content['name']}_{pheno}": mets
                                    for model, content in community_members.items()
                                    for pheno, mets in content["phenotypes"].items()}
        # invert the trial_name_conversion and data_paths keys and values
        short_code_trials = {}
        for row in trial_name_conversion:
            for col, contents in trial_name_conversion[row].items():
                short_code_trials[contents[0]] = row+col
            # short_code_trials = {contents[0]:contents[1] for contents in trial_name_conversion[row].values()}
        name_signal = {name: signal for signal, name in data_paths.items()}

        # calculate the 90% concentration for each carbon source
        requisite_fluxes = {}
        for trial in [short_code_trials[ID] for ID in growth_df_trials]:
            row_letter = trial[0] ; col_number = trial[1:]
            ## add rows where the initial concentration in the first trial is non-zero
            short_code = trial_name_conversion[row_letter][col_number][0]
            requisite_fluxes[short_code] = {}
            utilized_phenos = {}
            food_gradient = carbon_conc.copy()
            for dimension, content in food_gradient.items():
                for met, conc_dict in content.items():
                    if dimension == "rows":
                        if conc_dict[row_letter] == 0 or f"EX_{met}_e0" not in fluxes_df.index:
                            continue
                        source_conc = conc_dict[row_letter]
                    elif dimension == "columns":
                        if conc_dict[int(col_number)] == 0 or f"EX_{met}_e0" not in fluxes_df.index:
                            continue
                        source_conc = conc_dict[int(col_number)]
                    for pheno, val in fluxes_df.loc[f"EX_{met}_e0"].items():
                        # pheno = strip_comp(pheno)
                        # species, phenotype = pheno.split("_", 1)
                        if val < 0:
                            utilized_phenos[pheno] = source_conc*0.9 / val
            total_consumed = sum(list(utilized_phenos.values()))
            excreta = {}
            # display(fluxes_df)
            for pheno, absorption in utilized_phenos.items():
                species, phenotype = pheno.split("_", 1)
                fluxes = fluxes_df.loc[:, pheno] * abs(utilized_phenos[pheno])*(absorption/total_consumed)
                requisite_fluxes[short_code][f"{species}|{name_signal[species]}"] = fluxes[fluxes != 0]
                pheno = reverse_strip_comp(pheno)
                if "excreted" in pheno_info[pheno]:
                    # print(pheno_info[pheno]["excreted"])
                    for met in pheno_info[pheno]["excreted"]:
                        excreta[met] = fluxes.loc[f"EX_{met}_e0"]
            ## determine the fluxes for the other members of the community through cross-feeding
            participated_species = []
            for pheno, mets in pheno_info.items():
                species, phenotype = pheno.split("_", 1)
                if any([species in ph for ph in utilized_phenos]) or species in participated_species:
                    continue
                for met in mets["consumed"]:
                    if met not in excreta:
                        continue
                    fluxes = abs(excreta[met] * 0.99 / fluxes_df.loc[f"EX_{met}_e0", pheno]) * fluxes_df.loc[:, pheno]
                    requisite_fluxes[short_code][f"{species}|{name_signal[species]}"] = fluxes[fluxes != 0]
                    participated_species.append(species)
        return requisite_fluxes

    @staticmethod
    def data_process(dataframes, trial_name_conversion):
        short_codes, trials_list = [], []
        values, times = {}, {}  # The times must capture upstream
        first = True
        for df_name, (times_df, values_df) in dataframes.items():
            # print(df_name)
            # display(times_df) ; display(values_df)
            times_tup = FBAHelper.parse_df(times_df)
            average_times = np.mean(times_tup.values, axis=0)
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
                    times[df_name].extend(average_times)
            first = False
        # process the data to the smallest dataset, to accommodate heterogeneous data sizes
        minVal = min(list(map(len, values.values())))
        for df_name, data in values.items():
            values[df_name] = data[:minVal]
        times2 = times.copy()
        for df_name, data in times2.items():
            times[df_name] = data[:minVal]
        # construct the growth DataFrame
        df_data = {"trial_IDs": trials_list[:minVal], "short_codes": short_codes[:minVal]}
        df_data.update({"Time (s)": np.mean(list(times.values()), axis=0)})  # element-wise average
        df_data.update({df_name:vals for df_name, vals in values.items()})
        growth_df = DataFrame(df_data)
        growth_df.index = growth_df["short_codes"]
        del growth_df["short_codes"]
        growth_df.to_csv("growth_spectra.csv")
        return growth_df


class BiologData:

    @staticmethod
    def process(data_paths, trial_conditions_path, community_members, col_row_num, member_conversions,
                culture=None, date=None, significant_deviation=None, solver="glpk", msdb_path:str=None):
        row_num = 8 ; column_num = 12
        (zipped_output, data_timestep_hr, simulation_time, dataframes, trials, culture, date, fluxes_df
         ) = BiologData.load_data(data_paths, significant_deviation, community_members,
                                  col_row_num, row_num, culture, date, solver)
        experimental_metadata, standardized_carbon_conc, trial_name_conversion = BiologData.metadata(
            trial_conditions_path, row_num, column_num, culture, date)
        biolog_df = BiologData.data_process(dataframes, trial_name_conversion)
        requisite_biomass = BiologData.biomass_growth(biolog_df, member_conversions)
        return (experimental_metadata, biolog_df, fluxes_df, standardized_carbon_conc, requisite_biomass,
                trial_name_conversion, np.mean(data_timestep_hr), simulation_time)

    @staticmethod
    def load_data(data_paths, significant_deviation, community_members, col_row_num,
                  row_num, culture, date, solver):
        zipped_output = [data_paths['path'], "fluxes.csv"]
        # determine the metabolic fluxes for each member and phenotype
        # import and parse the raw CSV data
        # TODO - this may be capable of emulating leveraged functions from the GrowthData object
        fluxes_df = GrowthData.phenotypes(community_members, solver)
        # fluxes_df = None
        data_timestep_hr = []
        dataframes = {}
        raw_data = _spreadsheet_extension_load(data_paths['path'])
        significant_deviation = significant_deviation or 2
        # culture = culture or _find_culture(data_paths['path'])
        culture = culture or ",".join([x for x in data_paths.values() if (x not in ["OD"] and not re.search(r"\w\.\w", x))])
        date = date or _findDate(data_paths['path'])
        for org_sheet, name in data_paths.items():
            if org_sheet == 'path':
                continue
            sheet = org_sheet.replace(" ", "_")
            df_name = f"{name}:{sheet}"
            if df_name not in dataframes:
                dataframes[df_name] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[df_name].columns = dataframes[df_name].iloc[col_row_num]
                dataframes[df_name].drop(dataframes[df_name].index[:col_row_num+1], inplace=True)
                dataframes[df_name].dropna(inplace=True)
            # parse the DataFrame for values
            dataframes[df_name].columns = [str(x).strip() for x in dataframes[df_name].columns]
            simulation_time = dataframes[df_name].iloc[0, -1] / hour
            # display(dataframes[df_name])
            data_timestep_hr.append(simulation_time / int(float(dataframes[df_name].columns[-1])))
            # define the times and data
            data_times_df, data_values_df = _df_construction(
                name, df_name, None, None, significant_deviation,
                dataframes[df_name], row_num, False)
            # display(data_times_df) ; display(data_values_df)
            dataframes[df_name] = (data_times_df, data_values_df)

        # differentiate the phenotypes for each species
        trials = set(chain.from_iterable([list(df.index) for df, times in dataframes.values()]))
        return (zipped_output, data_timestep_hr, simulation_time, dataframes, trials, culture, date, fluxes_df)

    @staticmethod
    def metadata(trial_conditions_path, row_num, column_num, culture, date):
        # define the conditions for each trial
        with open(trial_conditions_path) as trials:
            trial_conditions = json.load(trials)

        # define the metadata DataFrame and a few columns
        constructed_experiments = DataFrame()
        ex_prefix = "B"
        constructed_experiments.index = [f"{ex_prefix}{x+1}" for x in list(range(row_num*column_num))]
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
                short_code = ex_prefix+str(count)

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
            # display(df_name, times_df, values_df)
            times_tup = FBAHelper.parse_df(times_df)
            # display(DataFrame(times_tup.values))
            average_times = list(np.mean(times_tup.values, axis=0))
            # print(average_times)
            # print(len(average_times))
            values[df_name], times[df_name] = [], []
            for exprID in values_df.index:
                row_let, col_num = exprID[0], exprID[1:]
                for trial_row_values in trial_contents(exprID, values_df.index, values_df.values):
                    if first:
                        short_code, experimentalID = trial_name_conversion[row_let][col_num]
                        trials_list.extend([experimentalID] * len(values_df.columns))
                        short_codes.extend([short_code] * len(values_df.columns))
                    if len(trial_row_values) != len(average_times):
                        print(f"The length of the trial data {len(trial_row_values)} "
                              f"exceeds that of the timesteps {len(average_times)} "
                              f"which creates an incompatible DataFrame.")
                    values[df_name].extend(trial_row_values)
                    times[df_name].extend(average_times)
            first = False
        # process the data to the smallest dataset, to accommodate heterogeneous data sizes
        minVal = min(list(map(len, values.values())))
        for df_name, data in values.items():
            values[df_name] = data[:minVal]
        times2 = times.copy()
        for df_name, data in times2.items():
            times[df_name] = data[:minVal]
        df_data = {"trial_IDs": trials_list, "short_codes": short_codes}
        df_data.update({"Time (s)": list(np.mean(list(times.values()), axis=0))})  # element-wise average
        df_data.update({df_name:vals for df_name, vals in values.items()})
        biolog_df = DataFrame(df_data)
        biolog_df.index = biolog_df["short_codes"]
        del biolog_df["short_codes"]
        biolog_df.to_csv("growth_spectra.csv")

        return biolog_df

    @staticmethod
    def biomass_growth(biolog_df, member_conversions):
        requisite_biomass = {}
        for short_code in biolog_df.index.unique():
            requisite_biomass[short_code] = {}
            for signal, conversion in member_conversions.items():
                short_code_df = biolog_df[biolog_df.index == short_code]
                requisite_biomass[short_code][signal] = conversion * short_code_df[
                    signal.replace("|", ":").replace(" ", "_")].iloc[-1]
        return requisite_biomass
