# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:44:07 2022

@author: Andrew Freiburger
"""
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from modelseedpy.fbapkg.elementuptakepkg import ElementUptakePkg
from modelseedpy.community.mscompatibility import MSCompatibility
from modelseedpy.core.msmodelutl import MSModelUtil
from modelseedpy.core.msminimalmedia import minimizeFlux_withGrowth
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Constraint
from optlang.symbolics import Zero
from scipy.constants import hour
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from itertools import chain
from typing import Union, Iterable
from cobra.medium import minimal_medium
from pprint import pprint
from icecream import ic
# from cplex import Cplex
from math import inf
import logging, json, os, re
from pandas import read_csv, DataFrame, ExcelFile, read_excel
import numpy as np


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
    matches = [True if ele == short_code else False for ele in indices_tup]
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
              f"for the {content} data since the OD plateaued and is no longer valid.")
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

def _df_construction(name, signal, ignore_trials, ignore_timesteps, significant_deviation, dataframes, row_num):
    # refine the DataFrames
    time_df = _column_reduction(dataframes[signal].iloc[0::2])
    values_df = _column_reduction(dataframes[signal].iloc[1::2])

    # remove specified data trials
    if ignore_trials:
        values_df, removed_trials = _remove_trials(
            values_df, ignore_trials, signal, name, significant_deviation)
        for row in removed_trials:
            time_df.drop(row, axis=0, inplace=True)

    # remove specified data timesteps
    if ignore_timesteps:
        values_df, removed_timesteps = _remove_timesteps(
            values_df, ignore_timesteps, name, signal)
        for col in list(map(int, removed_timesteps)):
            time_df.drop(col, axis=1, inplace=True)

    # remove undefined trials
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

def add_rel_flux_cons(model, ex, exRXN, rel_flux=0.2):
    # {ex.id}_uptakeLimit: {net_{carbonous_ex}} >= {net_{carbon_source}}*0.1
    #  The negative flux sign of influxes specifies that the carbon_source value must be lesser than the other
    #  carbon influx that is being constrained.
    cons = Constraint(Zero, lb=0, ub=None, name=f"{ex.id}_uptakeLimit")
    model.add_cons_vars(cons)
    cons.set_linear_coefficients({
            ex.forward_variable:1, ex.reverse_variable:1,
            exRXN.forward_variable:-rel_flux, exRXN.reverse_variable:-rel_flux})
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
        (
            media_conc, zipped_output, data_timestep_hr, simulation_time, signal_species,
            dataframes, trials, species_phenos_df, fluxes_df
        ) = GrowthData.load_data(
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
        return (experimental_metadata, growth_dfs, fluxes_df, standardized_carbon_conc, signal_species,
                trial_name_conversion, species_phenos_df, np.mean(data_timestep_hr), simulation_time, media_conc)

    @staticmethod
    def phenotypes(base_media, community_members, solver):
        # log information of each respective model
        # TODO - implement an optimization that maximizes consumption of the phenotype compound while maintaining a given growth
        #  value. This may be an opportunity to generalize the MSMinimalMedia method of constraining growth and optimizing for something
        #  else.
        named_community_members = {content["name"]: list(content["phenotypes"].keys()) + ["stationary"]
                                   for member, content in community_members.items()}
        models = OrderedDict()
        solutions = []
        media_conc = set()
        for org_model, content in community_members.items():  # community_members excludes the stationary phenotype
            model_util = MSModelUtil(org_model)
            # model_rxns = [rxn.id for rxn in org_model.reactions]
            models[org_model.id] = {"exchanges": model_util.exchange_list(), "solutions": {},
                                    "name": content["name"], "phenotypes": named_community_members[content["name"]]}
            for pheno, cpds in content['phenotypes'].items():
                # TODO - determine the minimal media a layer above when media can transfer with models
                model = org_model.copy()  # second copy is needed for each phenotype
                model = MSCompatibility.standardize([model], conflicts_file_name="orig_conflicts.json", printing=False)[0]
                pheno_cpds = [model.metabolites.get_by_id(cpd+"_e0") for cpd in cpds]
                ## define the media and uptake fluxes, which are 100 except for O_2, where its inclusion is interpreted as an aerobic model
                if base_media:
                    model_exchanges = [ex.id for ex in model_util.exchange_list()]
                    model.medium = {"EX_" + cpd.id + "_e0": -cpd.minFlux for cpd in base_media.mediacompounds
                                    if "EX_" + cpd.id + "_e0" in model_exchanges}
                    media_conc = {cpd.id: cpd.concentration for cpd in base_media.mediacompounds}
                else:
                    model.medium = minimal_medium(model, minimize_components=True)
                    media_conc.update(list(model.medium.keys()))
                model.medium = {cpd: 100 for cpd, flux in model.medium.items()}
                model.medium.update({cpd: 2 for cpd, flux in model.medium.items() if "cpd00007" in cpd})

                ## constraining compromise of the primary carbon source from extraneous carbon sources
                # ElementUptakePkg(model).build_package({"C": max([cpd.elements["C"] for cpd in pheno_cpds])})
                ## maximize the phenotype influx with a constrained growth
                col = content["name"] + '_' + pheno
                min_growth = .1 ; rel_flux = 0.2
                constraints = {}
                cpd = cpds[0]
                for cpd in cpds:
                    exRXN = model.reactions.get_by_id("EX_"+cpd+"_e0")
                    exRXN.lower_bound = -1000
                    # exRXN.upper_bound = -1  this causes infeasibility
                    for ex in model_util.exchange_list():
                        if "cpd00011" in ex.id or cpd in ex.id:
                            continue
                        ex_met = [ex.reactants + ex.products][0][0]
                        if "C" in ex_met.elements:
                            # print([ex.forward_variable, ex.reverse_variable, exRXN.forward_variable, exRXN.reverse_variable])
                            try:
                                model, constraints[f"{ex.id}_uptakeLimit"] = add_rel_flux_cons(model, ex, exRXN, rel_flux)
                                print(constraints[f"{ex.id}_uptakeLimit"])
                            except KeyError as e:
                                # this error must be investgiated and properly resolved/prevented beyond a temporary try-except block
                                print(f"\nThe exchange {e} is incorrectly defined in the {model.id} model.\n")
                # FBAHelper.add_objective(model, sum([
                #         model.reactions.get_by_id("EX_"+cpd+"_e0").flux_expression for cpd in cpds]), "min")
                # print(model.objective)
                # sol = model.optimize()
                sol, sol_dict = minimizeFlux_withGrowth(model, min_growth=min_growth, obj=sum([
                        model.reactions.get_by_id("EX_"+cpd+"_e0").flux_expression for cpd in cpds]))
                # print(*[f"{cons}\n" for cons in model.constraints])
                # ## prevent the consumption of other phenotype carbon sources
                # for other_pheno, other_cpds in content['phenotypes'].items():
                #     if pheno != other_pheno:
                #         for cpdID in cpds:
                #             rxnID = "EX_" + cpdID + "_e0"
                #             if rxnID in model_rxns:
                #                 model.reactions.get_by_id(rxnID).lb = 0
                # ## add the phenotype-specific constraints
                # for cpdID in cpds:
                #     rxnID = "EX_" + cpdID + "_e0"
                #     if rxnID not in model_rxns:
                #         raise ValueError(f"The cpd phenotype {cpdID} does not exist in the model {model.id}.")
                #         # model.add_boundary(metabolite=model.metabolites.get_by_id(cpdID), reaction_id=rxnID, type="exchange")
                #     model.reactions.get_by_id(rxnID).bounds = (-1, -.98)
                # ## solve the model and store the optimal solution
                # model.solver = solver
                # models[model.id]["solutions"][col] = model.optimize()
                # if models[model.id]["solutions"][col].objective_value == 0:
                #     raise NoFluxError(f"The {model.id} yields zero flux with its media: \n{model.medium}")

                ## normalize the fluxes to -1 for the influx of each phenotype's respective source
                print(sol.fluxes["EX_"+cpd+"_e0"])
                sol.fluxes = sol.fluxes / abs(sol.fluxes["EX_"+cpd+"_e0"])
                models[model.id]["solutions"][col] = sol
                solutions.append(models[model.id]["solutions"][col].objective_value)

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
        phenos_tup = FBAHelper.parse_df(fluxes_df)

        # import and parse the raw CSV data
        data_timestep_hr = []
        dataframes, signal_species = {}, {}
        zipped_output.append(data_paths['path'])
        max_timestep_cols = []
        if min_timesteps:
            for org_sheet, name in data_paths.items():
                if org_sheet == 'path' or "OD" in sheet:
                    continue
                ## define the DataFrame
                sheet = org_sheet.replace(' ', '_')
                dataframes[sheet] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[sheet].columns = dataframes[sheet].iloc[6]
                dataframes[sheet].drop(dataframes[sheet].index[:7], inplace=True)
                ## parse the timesteps from the DataFrame
                drop_timestep_range = GrowthData._min_significant_timesteps(
                    dataframes[sheet], ignore_timesteps, significant_deviation, ignore_trials, sheet, name)
                max_timestep_cols.append(drop_timestep_range)
            max_cols = max(list(map(len, max_timestep_cols)))
            ignore_timesteps = [x for x in max_timestep_cols if len(x) == max_cols][0]

        # remove trials for which the OD has plateaued
        for org_sheet, name in data_paths.items():
            if "OD" not in name:
                continue
            ## load the OD DataFrame
            sheet = org_sheet.replace(' ', '_')
            dataframes[sheet] = _spreadsheet_extension_parse(
                data_paths['path'], raw_data, org_sheet)
            dataframes[sheet].columns = dataframes[sheet].iloc[6]
            dataframes[sheet].drop(dataframes[sheet].index[:7], inplace=True)
            ## process the OD DataFrame
            signal_species[sheet] = name
            data_times_df, data_values_df = _df_construction(
                name, sheet, ignore_trials, ignore_timesteps, significant_deviation, dataframes, row_num)
            plateaued_times = _check_plateau(
                data_values_df, name, name, significant_deviation, 3)
            ## define and store the final DataFrames
            for col in plateaued_times:
                if col in data_times_df.columns:
                    data_times_df.drop(col, axis=1, inplace=True)
                if col in data_values_df.columns:
                    data_values_df.drop(col, axis=1, inplace=True)
            dataframes[sheet] = (data_times_df, data_values_df)
            break

        # refine the non-OD signals
        for org_sheet, name in data_paths.items():
            if org_sheet == 'path' or "OD" in name:
                continue
            sheet = org_sheet.replace(' ', '_')
            if sheet not in dataframes:
                dataframes[sheet] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[sheet].columns = dataframes[sheet].iloc[6]
                dataframes[sheet].drop(dataframes[sheet].index[:7], inplace=True)
            # parse the DataFrame for values
            signal_species[sheet] = name
            simulation_time = dataframes[sheet].iloc[0, -1] / hour
            data_timestep_hr.append(simulation_time / int(dataframes[sheet].columns[-1]))
            # define the times and data
            data_times_df, data_values_df = _df_construction(
                name, sheet, ignore_trials, ignore_timesteps, significant_deviation, dataframes, row_num)
            # display(data_times_df) ; display(data_values_df)
            for col in plateaued_times:
                if col in data_times_df.columns:
                    data_times_df.drop(col, axis=1, inplace=True)
                if col in data_values_df.columns:
                    data_values_df.drop(col, axis=1, inplace=True)
            dataframes[sheet] = (data_times_df, data_values_df)

        # differentiate the phenotypes for each species
        trials = set(chain.from_iterable([list(df.index) for df, times in dataframes.values()]))
        species_phenos_df = DataFrame(columns=phenos_tup.columns, data={pheno: {
            signal:1 if signal_species[signal.replace(" ", "_")] in pheno else 0
            for signal in data_paths if signal != "path" and "OD" not in signal
        } for pheno in phenos_tup.columns})
        return (media_conc, zipped_output, data_timestep_hr, simulation_time, signal_species,
                dataframes, trials, species_phenos_df, fluxes_df)

    @staticmethod
    def _min_significant_timesteps(full_df, ignore_timesteps, significant_deviation, ignore_trials, signal, name):
        # refine the DataFrames
        values_df = _column_reduction(full_df.iloc[1::2])
        values_df = _remove_trials(values_df, ignore_trials, signal, name, significant_deviation)
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
        members = list(mem["name"] for mem in community_members.values())
        species_mets = {mem["name"]: mets for mem in community_members.values() for mets in mem["phenotypes"].values()}

        # define the strains column
        strains, additional_compounds, experiment_ids = [], [], []
        trial_name_conversion = {}
        count = 1
        ## apply universal values to all trials
        base_row_conc = [] if '*' not in carbon_conc else [
            ':'.join([met, str(carbon_conc['*'][met][0]), str(carbon_conc['*'][met][1])])
            for met in carbon_conc['*']]
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
                    for index, mets in enumerate(species_mets.values()):
                        if metID in mets:
                            met_name = list(species_mets.keys())[index]
                            break
                    experiment_id.append(f"{init}_{met_name}")
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
        # experimental_ids = {codeIds[0]:codeIds[1]
        #                     for rowLet, content in trial_name_conversion.items()
        #                     for colNum, codeIds in content.items()}
        short_codes, trials_list = [], []
        values, times = {}, {}  # The times must be capture upstream
        first = True
        for sheet, (times_df, values_df) in dataframes.items():
            times_tup = FBAHelper.parse_df(times_df)
            average_times = np.mean(times_tup.values, axis=0)
            values[sheet], times[sheet] = [], []
            # print(sheet, set(values_df.index))
            for trial_code in values_df.index:
                row_let, col_num = trial_code[0], trial_code[1:]
                # print(trial_code, row_let, col_num)
                for row in trial_contents(trial_code, values_df.index, values_df.values):
                    if first:
                        short_code, experimentalID = trial_name_conversion[row_let][col_num]
                        trials_list.extend([experimentalID] * len(values_df.columns))
                        short_codes.extend([short_code] * len(values_df.columns))
                    values[sheet].extend(row)
                    times[sheet].extend(average_times)
            first = False
        df_data = {"trial_IDs": trials_list, "short_codes": short_codes}
        df_data.update({"Time (s)": np.mean(list(times.values()), axis=0)})  # element-wise average
        df_data.update({f"{sheet}":vals for sheet, vals in values.items()})
        growth_df = DataFrame(df_data)
        growth_df.index = growth_df["short_codes"]
        del growth_df["short_codes"]
        growth_df.to_csv("growth_spectra.csv")

        return growth_df


class BiologData:

    @staticmethod
    def process(data_paths, trial_conditions_path, community_members,
                culture=None, date=None, significant_deviation=None, solver="glpk"):
        row_num = 8 ; column_num = 12
        (fluxes_df, species_phenos_df, zipped_output, data_timestep_hr,
         simulation_time, signal_species, dataframes, trials, culture, date
         ) = BiologData.load_data(
            data_paths, community_members, significant_deviation, row_num, culture, date, solver)
        experimental_metadata, standardized_carbon_conc, trial_name_conversion = BiologData.metadata(
            trial_conditions_path, row_num , column_num, culture, date)
        biolog_df = BiologData.data_process(dataframes, trial_name_conversion, signal_species)
        return (experimental_metadata, biolog_df, fluxes_df, species_phenos_df, standardized_carbon_conc, signal_species,
                trial_name_conversion, np.mean(data_timestep_hr), simulation_time)

    @staticmethod
    def load_data(data_paths, community_members, significant_deviation, row_num, culture, date, solver):
        zipped_output = [data_paths['path']]
        # determine the metabolic fluxes for each member and phenotype
        named_community_members = {content["name"]: list(content["phenotypes"].keys())+["stationary"]
                                   for member, content in community_members.items()}
        models = OrderedDict()
        solutions = []
        for org_model, content in community_members.items():  # community_members excludes the stationary phenotype
            model_util = MSModelUtil(org_model)
            model_rxns = [rxn.id for rxn in org_model.reactions]
            models[org_model.id] = {"exchanges": model_util.exchange_list(), "solutions": {},
                                    "name": content["name"], "phenotypes": named_community_members[content["name"]]}
            for pheno, cpds in content['phenotypes'].items():
                ## copying a new model with each phenotype prevents bleedover
                # TODO - determine the minimal media a layer above when media can transfer with models
                model = org_model.copy()
                model = MSCompatibility.standardize([model], conflicts_file_name="orig_conflicts.json", printing=False)[0]
                ## constraining compromise of the primary carbon source from extraneous carbon sources
                ### The selection of a maximum accommodates diauxic phenotypes

                model.medium = minimal_medium(model, minimize_components=True)  # MSMinimalMedia.minimize_components(model_util.model)  #
                model.medium = {cpd: 100 for cpd, flux in model.medium.items()}
                model.medium.update({cpd: 2 for cpd, flux in model.medium.items() if cpd == "cpd00007"})
                model.solver = solver
                col = content["name"] + '_' + pheno
                for cpdID, bounds in cpds.items():
                    rxnID = "EX_" + cpdID + "_e0"
                    if rxnID not in model_rxns:
                        model.add_boundary(metabolite=model.metabolites.get_by_id(cpdID), reaction_id=rxnID, type="exchange")
                    model.reactions.get_by_id(rxnID).bounds = bounds
                for other_pheno, other_cpds in content['phenotypes'].items():
                    if pheno != other_pheno:
                        for cpdID, bounds in cpds.items():
                            rxnID = "EX_" + cpdID + "_e0"
                            if rxnID in model_rxns:
                                model.reactions.get_by_id(rxnID).lb = 0
                models[model.id]["solutions"][col] = model.optimize()
                if models[model.id]["solutions"][col].objective_value == 0:
                    raise NoFluxError(f"The {model.id} yields zero flux with its media: \n{model.medium}")
                solutions.append(models[model.id]["solutions"][col].objective_value)

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
                    flux = 0 if col not in content["solutions"] or \
                                ex_rxn.id not in list(content["solutions"][col].fluxes.index) \
                                else content["solutions"][col].fluxes[ex_rxn.id]
                    cols[col].append(flux)

        ## construct the DataFrame
        fluxes_df = DataFrame(data=cols)
        fluxes_df.index = fluxes_df['rxn']
        fluxes_df.drop('rxn', axis=1, inplace=True)
        fluxes_df = fluxes_df.groupby(fluxes_df.index).sum()
        fluxes_df = fluxes_df.loc[(fluxes_df != 0).any(axis=1)]
        fluxes_df.astype(str)
        fluxes_df.to_csv("fluxes.csv")
        phenos_tup = FBAHelper.parse_df(fluxes_df)
        zipped_output.append("fluxes.csv")

        # import and parse the raw CSV data
        data_timestep_hr = []
        dataframes, signal_species = {}, {}
        raw_data = _spreadsheet_extension_load(data_paths['path'])
        significant_deviation = significant_deviation or 2
        culture = culture or  _find_culture(data_paths['path'])
        date = date or _findDate(data_paths['path'])
        for org_sheet, name in data_paths.items():
            if org_sheet == 'path':
                continue
            sheet = org_sheet.replace(" ", "_")
            if sheet not in dataframes:
                dataframes[sheet] = _spreadsheet_extension_parse(
                    data_paths['path'], raw_data, org_sheet)
                dataframes[sheet].columns = dataframes[sheet].iloc[8]
                dataframes[sheet].drop(dataframes[sheet].index[:9], inplace=True)
            # parse the DataFrame for values
            dataframes[sheet].columns = [str(x).strip() for x in dataframes[sheet].columns]
            signal_species[sheet] = name
            simulation_time = dataframes[sheet].iloc[0, -1] / hour
            # display(dataframes[sheet])
            data_timestep_hr.append(simulation_time / int(float(dataframes[sheet].columns[-1])))
            # define the times and data
            data_times_df, data_values_df = _df_construction(
                name, sheet, None, None, significant_deviation, dataframes, row_num)
            # display(data_times_df) ; display(data_values_df)
            dataframes[sheet] = (data_times_df, data_values_df)

        # differentiate the phenotypes for each species
        trials = set(chain.from_iterable([list(df.index) for df, times in dataframes.values()]))
        species_phenos_df = DataFrame(columns=phenos_tup.columns, data={pheno: {
            signal:1 if signal_species[signal] in pheno else 0
            for signal in data_paths if signal != "path" and "OD" not in signal
        } for pheno in phenos_tup.columns})
        return (fluxes_df, species_phenos_df, zipped_output, data_timestep_hr,
                simulation_time, signal_species, dataframes, trials, culture, date)

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
    def data_process(dataframes, trial_name_conversion, signal_species):
        short_codes, trials_list = [], []
        values, times = {}, {}  # The times must be capture upstream
        first = True
        for sheet, (times_df, values_df) in dataframes.items():
            times_tup = FBAHelper.parse_df(times_df)
            average_times = np.mean(times_tup.values, axis=0)
            values[sheet], times[sheet] = [], []
            for short_code in values_df.index:
                row_let, col_num = short_code[0], short_code[1:]
                for row in trial_contents(short_code, values_df.index, values_df.values):
                    if first:
                        short_code, experimentalID = trial_name_conversion[row_let][col_num]
                        trials_list.extend([experimentalID] * len(values_df.columns))
                        short_codes.extend([short_code] * len(values_df.columns))
                    values[sheet].extend(row)
                    times[sheet].extend(average_times)
            first = False
        df_data = {"trial_IDs": trials_list, "short_codes": short_codes}
        df_data.update({"Time (s)": np.mean(list(times.values()), axis=0)})  # element-wise average
        df_data.update({f"{signal_species[sheet]}":vals for sheet, vals in values.items()})
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
# experimental_metadata, growth_df, fluxes_df, standardized_carbon_conc, signal_species, trial_name_conversion, species_phenos_df, data_timestep_hr, simulation_timestep, media_conc = GrowthData.process(
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