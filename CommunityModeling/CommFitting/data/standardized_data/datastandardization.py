# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:44:07 2022

@author: Andrew Freiburger
"""
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from cobra.medium import minimal_medium
from modelseedpy.core.fbahelper import FBAHelper
from scipy.constants import hour
from collections import OrderedDict
from zipfile import ZipFile, ZIP_LZMA
from itertools import chain
from typing import Union, Iterable
# from cplex import Cplex
import logging, json, os, re
from pandas import read_csv, DataFrame, ExcelFile, read_excel
import numpy as np


def isnumber(string):
    try:
        float(string)
    except:
        return False
    return True


def findDate(string):
    monthNames = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
    monthNums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    days = list(range(31, 0, -1))  # [f"{num}-" for num in list(range(31,0,-1))]
    years = list(range(2010, 2025))+list(range(10,25))  # [f"-{num}" for num in list(range(2000, 2100))]
    americanDates = [f"{mon}-{day}-{year}" for mon in monthNums for day in days for year in years]

    for date in americanDates:
        if re.search(date, string):
            month, day, year = date.split("-")
            return "-".join([day, month, year])

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


class DataStandardization:

    @staticmethod
    def process_jeffs_data(base_media, community_members: dict, solver: str = 'glpk',
                           signal_csv_paths: dict = None, species_abundances: str = None, carbon_conc_series: dict = None,
                           ignore_trials: Union[dict, list] = None, ignore_timesteps: list = None, species_identities_rows=None,
                           significant_deviation: float = 2, extract_zip_path: str = None, growth_data_path=None):
        (
            media_conc, zipped_output, data_timestep_hr, simulation_time, dataframes,
            data_timesteps, trials, species_phenos_df
        ) = DataStandardization.load_data(
            base_media, community_members, solver, signal_csv_paths, ignore_trials,
            ignore_timesteps, significant_deviation, extract_zip_path
        )
        constructed_experiments, trial_name_conversion = DataStandardization.metadata(
            base_media, community_members, species_abundances, species_identities_rows, findDate(growth_data_path))
        growth_dfs = DataStandardization.growth_data(growth_data_path, trial_name_conversion)
        return (constructed_experiments, growth_dfs, trial_name_conversion, np.mean(data_timestep_hr), simulation_time)

    @staticmethod
    def load_data(base_media, community_members, solver, signal_csv_paths,
                  ignore_trials, ignore_timesteps, significant_deviation, extract_zip_path):
        # define default values
        ignore_timesteps = ignore_timesteps or []
        signal_csv_paths = signal_csv_paths or {}

        named_community_members = {content["name"]: list(content["phenotypes"].keys()) + ["stationary"]
                                   for member, content in community_members.items()}
        media_conc = {cpd.id: cpd.concentration for cpd in base_media.mediacompounds}
        zipped_output = []
        if extract_zip_path:
            with ZipFile(extract_zip_path, 'r') as zp:
                zp.extractall()

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
            models[model] = {"exchanges": FBAHelper.exchange_reactions(model), "solutions": {},
                             "name": content["name"], "phenotypes": named_community_members[content["name"]]}
            for pheno, cpds in content['phenotypes'].items():
                col = content["name"] + '_' + pheno
                for cpdID, bounds in cpds.items():
                    rxnID = "EX_" + cpdID + "_e0"
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
                col = content["name"] + '_' + phenotype
                cols[col] = [0]
                if col in content["solutions"]:
                    bio_rxns = [x for x in content["solutions"][col].fluxes.index if "bio" in x]
                    flux = np.mean(
                        [content["solutions"][col].fluxes[rxn] for rxn in bio_rxns if content["solutions"][col].fluxes[rxn] != 0])
                    cols[col] = [flux]
        ## exchange reactions rows
        looped_cols = cols.copy();
        looped_cols.pop("rxn")
        for model, content in models.items():
            for ex_rxn in content["exchanges"]:
                cols["rxn"].append(ex_rxn.id)
                for col in looped_cols:
                    ### reactions that are not present in the columns are ignored
                    flux = 0 if col not in content["solutions"] or \
                                ex_rxn.id not in list(content["solutions"][col].fluxes.index) else content["solutions"][col].fluxes[
                        ex_rxn.id]
                    cols[col].append(flux)

        ## construct the DataFrame
        fluxes_df = DataFrame(data=cols)
        fluxes_df.index = fluxes_df['rxn']
        fluxes_df.drop('rxn', axis=1, inplace=True)
        fluxes_df = fluxes_df.groupby(fluxes_df.index).sum()
        fluxes_df = fluxes_df.loc[(fluxes_df != 0).any(axis=1)]
        fluxes_df.astype(str)
        fluxes_df.to_csv("fluxes.csv")
        zipped_output.append("fluxes.csv")

        # define only species for which data is defined
        modeled_species = list(v for v in signal_csv_paths.values() if ("OD" not in v and " " not in v))
        removed_phenotypes = [col for col in fluxes_df if not any([species in col for species in modeled_species])]
        for col in removed_phenotypes:
            fluxes_df.drop(col, axis=1, inplace=True)
        if removed_phenotypes:
            print(f'The {removed_phenotypes} phenotypes were removed '
                  f'since their species is not among those with data: {modeled_species}.')
        phenos_tup = FBAHelper.parse_df(fluxes_df)

        # define the set of used trials
        ignore_timesteps = list(map(str, ignore_timesteps))

        # import and parse the raw CSV data
        data_timestep_hr = []
        dataframes, signal_species = {}, {}
        zipped_output.append(signal_csv_paths['path'])
        raw_data = DataStandardization._spreadsheet_extension_load(signal_csv_paths['path'])
        for org_sheet, name in signal_csv_paths.items():
            if org_sheet != 'path':
                sheet = org_sheet.replace(' ', '_')
                dataframes[sheet] = DataStandardization._spreadsheet_extension_parse(
                    signal_csv_paths['path'], raw_data, org_sheet)
                dataframes[sheet].columns = dataframes[sheet].iloc[6]
                dataframes[sheet] = dataframes[sheet].drop(dataframes[sheet].index[:7])
                # parse the DataFrame for values
                signal_species[sheet] = name
                simulation_time = dataframes[sheet].iloc[0, -1] / hour
                data_timestep_hr.append(simulation_time / int(dataframes[sheet].columns[-1]))
                dataframes = DataStandardization._df_construction(
                    name, sheet, ignore_trials, ignore_timesteps, significant_deviation, dataframes)

        data_timestep_hr = np.mean(data_timestep_hr)
        data_timesteps = int(simulation_time / data_timestep_hr)
        trials = set(chain.from_iterable([list(df.index) for df in dataframes.values()]))

        # differentiate the phenotypes for each species
        species_phenos_df = DataFrame(columns=phenos_tup.columns, data={signal: np.array([
            1 if "OD" not in signal and signal != "path" and signal_species[signal] in pheno else 0
            for pheno in phenos_tup.columns]) for signal in signal_csv_paths})

        return (media_conc, zipped_output, data_timestep_hr, simulation_time,
                dataframes, data_timesteps, trials, species_phenos_df)

    @staticmethod
    def _spreadsheet_extension_load(path):
        if ".csv" in path:
            return read_csv(path)
        elif ".xls" in path:
            return ExcelFile(path)

    @staticmethod
    def _spreadsheet_extension_parse(path, raw_data, org_sheet):
        if ".csv" in path:
            return raw_data
        elif ".xls" in path:
            return raw_data.parse(org_sheet)

    @staticmethod
    def metadata(base_media, community_members, species_abundances, carbon_conc, species_identities_rows, date):
        # define carbon concentrations for each trial
        carbon_conc = carbon_conc or {}
        carbon_conc['columns'] = default_dict_values(carbon_conc, "columns", {})
        carbon_conc['rows'] = default_dict_values(carbon_conc, "rows", {})
        column_num = len(species_abundances)
        row_num = len(list(carbon_conc["rows"].values())[0]) or len(species_identities_rows)

        # define the metadata DataFrame and a few columns
        constructed_experiments = DataFrame()
        experiment_prefix = "A"
        constructed_experiments["short_code"] = [f"{experiment_prefix}{x+1}" for x in list(range(column_num*row_num))]
        constructed_experiments["base_media"] = [base_media.path[0]] * (column_num*row_num)

        # define community content
        members = list(content["name"] for content in community_members.values())
        species_mets = {content["name"]: list(v.keys())
                        for content in community_members.values() for v in content["phenotypes"].values()}

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
            print(row_concentration)
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
                trial_name_conversion[trial_letter][str(col+1)] = experiment_prefix + str(count)
                count += 1

        # convert the variable concentrations to short codes
        standardized_carbon_conc = {}
        for met, conc in carbon_conc["rows"].items():
            for row, val in conc.items():
                standardized_carbon_conc[met] = {short_code:val for short_code in trial_name_conversion[row].values()}
        for met, conc in carbon_conc["columns"].items():
            for col, val in conc.items():
                for row in trial_name_conversion:
                    standardized_carbon_conc[met][trial_name_conversion[row][col]] = val

        # add columns to the exported dataframe
        constructed_experiments.insert(0, "trial_IDs", experiment_ids)
        constructed_experiments["additional_compounds"] = additional_compounds
        constructed_experiments["strains"] = strains
        constructed_experiments["date"] = [date] * (column_num*row_num)
        constructed_experiments.to_csv("growth_metadata.csv")
        return constructed_experiments, trial_name_conversion

    def _met_id_parser(self, met):
        met_id = re.sub('(\_\w\d+)', '', met)
        met_id = met_id.replace('EX_', '', 1)
        met_id = met_id.replace('c_', '', 1)
        return met_id

    @staticmethod
    def _df_construction(name, signal, ignore_trials, ignore_timesteps, significant_deviation, dataframes):
        # refine the DataFrame
        dataframes[signal] = dataframes[signal].iloc[1::2]  # excludes the times
        dataframes[signal].columns = map(str, dataframes[signal].columns)
        dataframes[signal].index = dataframes[signal]['Well']
        for col in dataframes[signal].columns:
            if any([x in col for x in ['Plate', 'Well', 'Cycle']]):
                dataframes[signal].drop(col, axis=1, inplace=True)
        dataframes[signal].columns = map(float, dataframes[signal].columns)
        dataframes[signal].columns = map(int, dataframes[signal].columns)

        # filter data contents
        dropped_trials = []
        if isinstance(ignore_trials, dict):
            ignore_trials['columns'] = list(map(str, ignore_trials['columns'])) if 'columns' in ignore_trials else []
            ignore_trials['rows'] = list(map(str, ignore_trials['rows'])) if 'rows' in ignore_trials else []
            ignore_trials['wells'] = ignore_trials['wells'] if 'wells' in ignore_trials else []
        elif isIterable(ignore_trials):
            if ignore_trials[0][0].isalpha() and isnumber(ignore_trials[0][1:]):
                short_code = True  # TODO - drop trials with respect to the short codes, and not the full codes
        for trial in dataframes[signal].index:
            if isinstance(ignore_trials, dict) and any(
                    [trial[0] in ignore_trials['rows'], trial[1:] in ignore_trials['columns'], trial in ignore_trials['wells']]
            ) or isIterable(ignore_trials) and trial in ignore_trials:
                dataframes[signal].drop(trial, axis=0, inplace=True)
                dropped_trials.append(trial)
            elif isIterable(ignore_trials) and trial in ignore_trials:
                dataframes[signal].drop(trial, axis=0, inplace=True)
                dropped_trials.append(trial)
        if dropped_trials:
            print(f'The {dropped_trials} trials were dropped from the {name} measurements.')

        for col in dataframes[signal]:
            if col in ignore_timesteps:
                dataframes[signal].drop(col, axis=1, inplace=True)
        if 'OD' not in signal:
            removed_trials = []
            for trial, row in dataframes[signal].iterrows():
                row_array = np.array(row.to_list())
                ## remove trials for which the biomass growth did not change by the determined minimum deviation
                if row_array[-1] / row_array[0] < significant_deviation:
                    dataframes[signal].drop(trial, axis=0, inplace=True)
                    removed_trials.append(trial)
            if removed_trials:
                print(f'The {removed_trials} trials were removed from the {name} measurements, '
                      f'with their deviation over time being less than the threshold of {significant_deviation}.')

        # process the data for subsequent operations and optimal efficiency
        dataframes[signal].astype(str)
        dataframes[signal] = FBAHelper.parse_df(dataframes[signal])

        return dataframes

    def _process_csv(self, csv_path, index_col):
        self.zipped_output.append(csv_path)
        csv = read_csv(csv_path) ; csv.index = csv[index_col]
        csv.drop(index_col, axis=1, inplace=True)
        csv.astype(str)
        return csv

    @staticmethod
    def growth_data(data_path, trial_name_conversion):
        dataframes = {}
        raw_data = DataStandardization._spreadsheet_extension_load(data_path)
        worksheets = {"Raw OD(590)": "OD", "mNeonGreen": "pf", "mRuby": "ecoli"}
        for org_sheet, name in worksheets.items():
            sheet = org_sheet.replace(' ', '_')
            dataframes[sheet] = DataStandardization._spreadsheet_extension_parse(
                data_path, raw_data, org_sheet)
            dataframes[sheet].columns = dataframes[sheet].iloc[6]
            dataframes[sheet] = dataframes[sheet].drop(dataframes[sheet].index[:7])

        short_codes, trials_list = [], []
        values, times = {}, {}
        first = True
        for sheet, df in dataframes.items():
            values[sheet], times[sheet] = [], []
            for trial in set(df["Well"]):
                for index, row in df[df["Well"] == trial].iterrows():
                    if row["Cycle #"] == "Time (s)":
                        # the IDs are arbitrarily defined once to prevent duplication
                        if first:
                            numerical_columns = [x for x in row.index if isnumber(x)]
                            short_code, experimentalID = trial_name_conversion[trial]
                            trials_list.extend([experimentalID] * len(numerical_columns))
                            short_codes.extend([short_code] * len(numerical_columns))
                        times[sheet].extend([row[x] for x in numerical_columns])
                    elif row["Cycle #"] == "Result":
                        values[sheet].extend([row[x] for x in numerical_columns])
            first = False

        df_data = {"trial_IDs": trials_list, "short_codes": short_codes}
        df_data.update({"Time (s)": np.mean(list(times.values()), axis=0)})  # element-wise average
        df_data.update({f"{sheet}":vals for sheet, vals in values.items()})
        growth_df = DataFrame(df_data)
        growth_df.index = growth_df["short_codes"]
        del growth_df["short_codes"]
        growth_df.to_csv("growth_spectra.csv")

        return growth_df