# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:44:07 2022

@author: Andrew Freiburger
"""
from pandas import read_csv, DataFrame, ExcelFile, read_excel
import numpy as np


class DataStandardization:
    # base_media = date = ""; columns = rows = 0 ; members = [""] ; zipped_output = []
    # species_abundances, carbon_sources, species_identities_rows, row_concentrations = {}, {}, {}, {}

    def process_csv(self, signal_csv_paths, ignore_trials, ignore_timesteps, significant_deviation):
        self.zipped_output.append(signal_csv_paths['path'])
        if "xls" in signal_csv_paths['path']:
            raw_data = ExcelFile(signal_csv_paths['path'])
        elif "csv" in signal_csv_paths['path']:
            raw_data = read_csv(signal_csv_paths['path'])
        for org_sheet, name in signal_csv_paths.items():
            if org_sheet != 'path':
                sheet = org_sheet.replace(' ', '_')
                if "xls" in signal_csv_paths['path']:
                    self.dataframes[sheet] = raw_data.parse(org_sheet)
                elif "csv" in signal_csv_paths['path']:
                    self.dataframes[sheet] = raw_data
                self.dataframes[sheet].columns = self.dataframes[sheet].iloc[6]
                self.dataframes[sheet] = self.dataframes[sheet].drop(self.dataframes[sheet].index[:7])
                self._df_construction(name, sheet, ignore_trials, ignore_timesteps, significant_deviation)

    def _df_construction(self, name, signal, ignore_trials, ignore_timesteps, significant_deviation):
        # parse the DataFrame for values
        self.signal_species[signal] = name
        
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
            ignore_trials = list(map(str, ignore_trials['rows']))
        for trial in self.dataframes[signal].index:
            if isinstance(ignore_trials, dict) and any(
                    [trial[0] in ignore_trials['rows'], trial[1:] in ignore_trials['columns'], trial in ignore_trials['wells']]
                    ) or isinstance(ignore_trials, list) and trial in ignore_trials:
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
        
        self.dataframes[signal].astype(str)
        self.dataframes[signal].to_csv("experiments.csv")