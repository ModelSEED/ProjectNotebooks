# -*- coding: utf-8 -*-
# from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from pandas import DataFrame
from optlang import Model, Objective
from modelseedpy.core.fbahelper import FBAHelper
from scipy.constants import hour, minute
from scipy.optimize import newton
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from itertools import chain
from matplotlib import pyplot
from typing import Union, Iterable
from pprint import pprint
from time import sleep, process_time
from math import inf, isclose
from icecream import ic
import numpy as np
# from cplex import Cplex
import logging, json, os, re

logger = logging.getLogger(__name__)


def isnumber(string):
    try:
        float(string)
    except:
        return False
    return True


def dict_keys_exists(dic, *keys):
    result = keys[0] in dic
    if keys[0] in dic:
        remainingKeys = keys[1:]
        if len(remainingKeys) > 0:
            result = dict_keys_exists(dic[keys[0]], *remainingKeys)
        return result
    return result


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


# define data objects
names = []


def _name(name, suffix, short_code, timestep):
    name = '-'.join(list(map(str, [name + suffix, short_code, timestep])))
    if name not in names:
        names.append(name)
        return name
    else:
        raise ObjectAlreadyDefinedError(f"The object {name} is already defined for the problem.")


class MSCommFitting:

    def __init__(self, fluxes_df, carbon_conc, media_conc, signal_species, species_phenos_df, growth_df=None, experimental_metadata=None):
        self.parameters, self.variables, self.constraints, = {}, {}, {}
        self.zipped_output, self.plots = [], []
        self.signal_species = signal_species; self.species_phenos_df = species_phenos_df
        self.fluxes_tup = FBAHelper.parse_df(fluxes_df)
        self.growth_df = growth_df; self.experimental_metadata = experimental_metadata
        self.carbon_conc = carbon_conc; self.media_conc = media_conc

    #################### FITTING PHASE METHODS ####################
    def fit(self, parameters:dict=None, mets_to_track: list = None, rel_final_conc:dict=None, zero_start:list=None,
            abs_final_conc:dict=None, graphs: list = None, data_timesteps: dict = None, msdb_path:str=None,
            export_zip_name: str = None, export_parameters: bool = True, export_lp: bool = True, figures_zip_name:str=None,
            publishing:bool=False):
        self.define_problem(parameters, mets_to_track, rel_final_conc, zero_start, abs_final_conc, data_timesteps, export_zip_name,
                            export_parameters, export_lp)
        self.compute(graphs, msdb_path, export_zip_name, figures_zip_name, publishing)

    def _export_model_json(self, json_model, path):
        with open(path, 'w') as lp:
            json.dump(json_model, lp, indent=3)

    def _met_id_parser(self, met):
        met_id = re.sub('(\_\w\d+)', '', met)
        met_id = met_id.replace('EX_', '', 1)
        met_id = met_id.replace('c_', '', 1)
        return met_id

    def _update_problem(self, contents: Iterable):
        for content in contents:
            self.problem.add(content)
            self.problem.update()

    def _check_plateau(self):
        # assign the initial OD value

        # iteratively check OD until a relative threshold to the initial value is surpassed
        ## start scanning 2-hour blocks for equivalence between the first and last values

        # remove all timesteps after the first timestep in a block where plateau is observed

        # return the refined OD, which must be used to refine all other datasets and acquire a standard data length

        pass

    def define_problem(self, parameters=None, mets_to_track = None, rel_final_conc=None, zero_start=None, abs_final_conc=None,
                       data_timesteps=None, export_zip_name: str=None, export_parameters: bool=True, export_lp: bool=True):
        # parse the growth data
        growth_tup = FBAHelper.parse_df(self.growth_df)
        num_sorted = np.sort(np.array([int(obj[1:]) for obj in set(growth_tup.index)]))
        # TODO - short_codes must be distinguished for different conditions
        unique_short_codes = [f"{growth_tup.index[0][0]}{num}" for num in map(str, num_sorted)]
        time_column_index = growth_tup.columns.index("Time (s)")
        full_times = growth_tup.values[:, time_column_index]
        self.times = {short_code: trial_contents(short_code, growth_tup.index, full_times) for short_code in unique_short_codes}

        # define default values
        parameters, data_timesteps = parameters or {}, data_timesteps or {}
        self.parameters["data_timestep_hr"] = np.mean(np.diff(np.array(list(self.times.values())).flatten())) / hour
        self.parameters.update({
            "timestep_hr": self.parameters['data_timestep_hr'],  # Simulation timestep magnitude in hours
            "cvct": 1, "cvcf": 1,  # Minimization coefficients of the phenotype conversion to and from the stationary phase.
            "bcv": 1,  # The highest fraction of species biomass that can change phenotypes in a timestep
            "cvmin": 0,  # The lowest value the limit on phenotype conversion goes,
            "v": 0.33,  # The kinetics constant that is externally adjusted
            'diffpos': 1, 'diffneg': 1,
            # diffpos and diffneg coefficients that weight difference between experimental and predicted biomass
        })
        self.parameters.update(parameters)
        default_carbon_sources = ["cpd00076", "cpd00179", "cpd00027"]  # sucrose, maltose, glucose
        self.rel_final_conc = rel_final_conc or {c:1 for c in default_carbon_sources}
        self.abs_final_conc = abs_final_conc or {}
        self.mets_to_track = mets_to_track or self.rel_final_conc
        zero_start = zero_start or []

        # TODO - this must be replaced with the algorithmic assessment of bad timesteps that Chris originally used
        timesteps_to_delete = {}  # {short_code: full_times for short_code in unique_short_codes}
        if data_timesteps:  # {short_code:[times]}
            for short_code, times in data_timesteps.items():
                timesteps_to_delete[short_code] = set(list(range(len(full_times)))) - set(times)
                self.times[short_code] = np.delete(self.times[short_code], list(timesteps_to_delete[short_code]))

        # construct the problem
        objective = tupObjective("minimize variance and phenotypic transitions", [], "min")
        constraints, variables = [], []
        time_1 = process_time()
        for met in self.fluxes_tup.index:
            met_id = self._met_id_parser(met)
            if self.mets_to_track and (met_id == 'cpd00001' or met_id not in self.mets_to_track):
                continue
            self.variables["c_" + met] = {}; self.constraints['dcc_' + met] = {}

            # define the growth rate for each metabolite and
            if "Vmax" and "Km" in self.parameters:
                self.parameters["Vmax"].update(self._universalize(self.parameters["Vmax"], met_id))
                self.parameters["Km"].update(self._universalize(self.parameters["Km"], met_id))
            for short_code in unique_short_codes:
                self.variables["c_" + met][short_code] = {}; self.constraints['dcc_' + met][short_code] = {}
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for index, timestep in enumerate(timesteps):
                    ## define the concentration variables
                    conc_var = tupVariable(_name("c_", met, short_code, timestep))
                    ## constrain initial time concentrations to the media or a large default
                    if index == 0 and not 'bio' in met_id:
                        initial_val = 100 if not self.media_conc or met_id not in self.media_conc else self.media_conc[met_id]
                        initial_val = 0 if met_id in zero_start else initial_val
                        if dict_keys_exists(self.carbon_conc, met_id, short_code):
                            initial_val = self.carbon_conc[met_id][short_code]
                        conc_var = conc_var._replace(bounds=Bounds(initial_val, initial_val))
                    ## mandate complete carbon consumption
                    if index == len(timesteps) - 1:
                        if met_id in self.rel_final_conc:
                            final_bound = self.variables["c_" + met][short_code][1].bounds.lb * self.rel_final_conc[met_id]
                        if met_id in self.abs_final_conc: # this intentionally overwrites rel_final_conc
                            final_bound = self.abs_final_conc[met_id]
                        conc_var = conc_var._replace(bounds=Bounds(0, final_bound))
                    self.variables["c_" + met][short_code][timestep] = conc_var
                    variables.append(self.variables["c_" + met][short_code][timestep])
        for signal in [signal for signal in growth_tup.columns[3:] if 'OD' not in signal]:
            for pheno in self.fluxes_tup.columns:
                if self.signal_species[signal] in pheno:
                    self.constraints['dbc_' + pheno] = {short_code: {} for short_code in unique_short_codes}

        # define growth and biomass variables and constraints
        # self.parameters["v"] = {met_id:{species:self._michaelis_menten()}}
        for pheno in self.fluxes_tup.columns:
            self.variables['cvt_' + pheno] = {}; self.variables['cvf_' + pheno] = {}
            self.variables['b_' + pheno] = {}; self.variables['g_' + pheno] = {}
            # self.variables['v_' + pheno] = {}
            self.constraints['gc_' + pheno] = {}; self.constraints['cvc_' + pheno] = {}
            # self.constraints['vc_' + pheno] = {}
            for short_code in unique_short_codes:
                self.variables['cvt_' + pheno][short_code] = {}; self.variables['cvf_' + pheno][short_code] = {}
                self.variables['b_' + pheno][short_code] = {}; self.variables['g_' + pheno][short_code] = {}
                # self.variables['v_' + pheno][short_code] = {}
                self.constraints['gc_' + pheno][short_code] = {}
                self.constraints['cvc_' + pheno][short_code] = {}
                # self.constraints['vc_' + pheno][short_code] = {}
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for timestep in timesteps:
                    # predicted biomass abundance and biomass growth
                    self.variables['b_' + pheno][short_code][timestep] = tupVariable(
                        _name("b_", pheno, short_code, timestep), Bounds(0, 100))
                    self.variables['g_' + pheno][short_code][timestep] = tupVariable(
                        _name("g_", pheno, short_code, timestep))
                    # self.variables['v_' + pheno][short_code][timestep] = tupVariable(
                    #     _name("v_", pheno, short_code, timestep), Bounds(0, 50))
                    variables.extend([self.variables['b_' + pheno][short_code][timestep],
                                      self.variables['g_' + pheno][short_code][timestep],])
                                      # self.variables['v_' + pheno][short_code][timestep]])

                    if 'stationary' in pheno:
                        continue
                    # the conversion rates to and from the stationary phase
                    self.variables['cvt_' + pheno][short_code][timestep] = tupVariable(
                        _name("cvt_", pheno, short_code, timestep), Bounds(0, 100))
                    self.variables['cvf_' + pheno][short_code][timestep] = tupVariable(
                        _name("cvf_", pheno, short_code, timestep), Bounds(0, 100))
                    variables.extend([self.variables['cvf_' + pheno][short_code][timestep],
                                      self.variables['cvt_' + pheno][short_code][timestep]])

                    # cvt <= bcv*b_{pheno} + cvmin
                    self.constraints['cvc_' + pheno][short_code][timestep] = tupConstraint(
                        _name('cvc_', pheno, short_code, timestep), (0, None), {
                            "elements": [
                                self.parameters['cvmin'],
                                {"elements": [self.parameters['bcv'],
                                              self.variables['b_' + pheno][short_code][timestep].name],
                                 "operation": "Mul"},
                                {"elements": [
                                    -1, self.variables['cvt_' + pheno][short_code][timestep].name],
                                    "operation": "Mul"}],
                            "operation": "Add"
                        })
                    # v_{pheno} = -(Vmax_{met, species} * c_{met}) / (Km_{met, species} + c_{met})
                    # self.constraints['vc_' + pheno][short_code][timestep] = tupConstraint(
                    #     name=_name('vc_', pheno, short_code, timestep),
                    #     expr={
                    #         "elements": [
                    #             self.variables['v_' + pheno][short_code][timestep].name,
                    #             {"elements": [-1, self.parameters['v'],
                    #                           self.variables['b_' + pheno][short_code][timestep].name],
                    #              "operation": "Mul"}],
                    #         "operation": "Add"
                    #     })
                    # g_{pheno} = b_{pheno}*v_{pheno}
                    self.constraints['gc_' + pheno][short_code][timestep] = tupConstraint(
                        name=_name('gc_', pheno, short_code, timestep),
                        expr={
                            "elements": [
                                self.variables['g_' + pheno][short_code][timestep].name,
                                {"elements": [-1, self.parameters['v'],
                                              self.variables['b_' + pheno][short_code][timestep].name],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })
                    constraints.extend([self.constraints['cvc_' + pheno][short_code][timestep],
                                        self.constraints['gc_' + pheno][short_code][timestep]])
                    objective.expr.extend([{
                        "elements": [
                            {"elements": [self.parameters['cvcf'],
                                          self.variables['cvf_' + pheno][short_code][timestep].name],
                             "operation": "Mul"},
                            {"elements": [self.parameters['cvct'],
                                          self.variables['cvt_' + pheno][short_code][timestep].name],
                             "operation": "Mul"}],
                        "operation": "Add"
                    }])

        # define the concentration constraint
        half_dt = self.parameters['data_timestep_hr'] / 2
        time_2 = process_time()
        print(f'Done with concentrations and biomass loops: {(time_2 - time_1) / 60} min')
        for r_index, met in enumerate(self.fluxes_tup.index):
            met_id = self._met_id_parser(met)
            if self.mets_to_track and (met_id == 'cpd00001' or met_id not in self.mets_to_track):
                continue
            for short_code in unique_short_codes:
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for timestep in timesteps[:-1]:
                    # c_{met} + dt/2*sum_k^K(n_{k,met} * (g_{pheno}+g+1_{pheno})) = c+1_{met}
                    next_timestep = timestep + 1
                    growth_phenos = [[self.variables['g_' + pheno][short_code][next_timestep].name,
                                      self.variables['g_' + pheno][short_code][timestep].name] for pheno in self.fluxes_tup.columns]
                    self.constraints['dcc_' + met][short_code][timestep] = tupConstraint(
                        name=_name("dcc_", met, short_code, timestep),
                        expr={
                            "elements": [
                                self.variables["c_" + met][short_code][timestep].name,
                                {"elements": [-1, self.variables["c_" + met][short_code][next_timestep].name],
                                 "operation": "Mul"},
                                *OptlangHelper.dot_product(growth_phenos,
                                                           heuns_coefs=half_dt * self.fluxes_tup.values[r_index])],
                            "operation": "Add"
                        })
                    constraints.append(self.constraints['dcc_' + met][short_code][timestep])

        #   define the conversion variables of every signal for every phenotype
        # for signal in growth_tup.columns[2:]:
        #     for pheno in self.fluxes_tup.columns:
        #         conversion_name = "_".join([signal, pheno, "__conversion"])
        #         self.variables[conversion_name] = tupVariable(conversion_name)
        #         variables.append(self.variables[conversion_name])

        time_3 = process_time()
        print(f'Done with DCC loop: {(time_3 - time_2) / 60} min')
        for index, signal in enumerate(growth_tup.columns[2:]):
            signal_column_index = index + 2
            data_timestep = 1
            # TODO - The conversion must be defined per phenotype
            self.variables[signal + '__conversion'] = tupVariable(signal + '__conversion')
            variables.append(self.variables[signal + '__conversion'])
            self.variables[signal + '__bio'] = {}; self.variables[signal + '__diffpos'] = {}
            self.variables[signal + '__diffneg'] = {}
            self.constraints[signal + '__bioc'] = {}; self.constraints[signal + '__diffc'] = {}
            for short_code in unique_short_codes:
                self.variables[signal + '__bio'][short_code] = {}
                self.variables[signal + '__diffpos'][short_code] = {}
                self.variables[signal + '__diffneg'][short_code] = {}
                self.constraints[signal + '__bioc'][short_code] = {}
                self.constraints[signal + '__diffc'][short_code] = {}
                # the value entries are matched to only the timesteps that are condoned by data_timesteps
                values_slice = trial_contents(short_code, growth_tup.index, growth_tup.values)
                if timesteps_to_delete:
                    values_slice = np.delete(values_slice, list(timesteps_to_delete[short_code]), axis=0)
                for timestep in list(range(1, len(values_slice) + 1))[:-1]:
                    ## the user timestep and data timestep must be synchronized
                    if int(timestep) * self.parameters['timestep_hr'] < data_timestep * self.parameters['data_timestep_hr']:
                        continue
                    data_timestep += 1
                    if data_timestep > int(self.times[short_code][-1] / self.parameters["data_timestep_hr"]):
                        break
                    next_timestep = int(timestep) + 1
                    ## the phenotype transition terms are aggregated
                    total_biomass, signal_sum, from_sum, to_sum = [], [], [], []
                    for pheno_index, pheno in enumerate(self.fluxes_tup.columns):
                        ### define the collections of signal and pheno terms
                        # TODO - The BIOLOG species_pheno_df index seems to misalign with the following calls
                        val = 1 if 'OD' in signal else self.species_phenos_df.loc[signal, pheno]
                        val = val if isnumber(val) else val.values[0]
                        signal_sum.append({"operation": "Mul", "elements": [
                            -1, val, self.variables["b_" + pheno][short_code][timestep].name]})
                        ### total_biomass.append(self.variables["b_"+pheno][short_code][timestep].name)
                        if all(['OD' not in signal, self.signal_species[signal] in pheno, 'stationary' not in pheno]):
                            from_sum.append({"operation": "Mul", "elements": [
                                -val, self.variables["cvf_" + pheno][short_code][timestep].name]})
                            to_sum.append({"operation": "Mul", "elements": [
                                val, self.variables["cvt_" + pheno][short_code][timestep].name]})
                    for pheno in self.fluxes_tup.columns:
                        if 'OD' in signal or self.signal_species[signal] not in pheno:
                            continue
                        if "stationary" in pheno:
                            # b_{phenotype} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) = b+1_{phenotype}
                            self.constraints['dbc_' + pheno][short_code][timestep] = tupConstraint(
                                name=_name("dbc_", pheno, short_code, timestep),
                                expr={
                                    "elements": [
                                        self.variables['b_' + pheno][short_code][timestep].name,
                                        *from_sum, *to_sum,
                                        {"elements": [-1, self.variables["b_" + pheno][short_code][next_timestep].name],
                                         "operation": "Mul"}],
                                    "operation": "Add"
                                })
                        else:
                            # b_{phenotype} + dt/2*(g_{phenotype} + g+1_{phenotype}) + cvf-cvt = b+1_{phenotype}
                            self.constraints['dbc_' + pheno][short_code][timestep] = tupConstraint(
                                name=_name("dbc_", pheno, short_code, timestep),
                                expr={
                                    "elements": [
                                        self.variables['b_' + pheno][short_code][timestep].name,
                                        self.variables['cvf_' + pheno][short_code][timestep].name,
                                        {"elements": [half_dt, self.variables['g_' + pheno][short_code][timestep].name],
                                         "operation": "Mul"},
                                        {"elements": [half_dt, self.variables['g_' + pheno][short_code][next_timestep].name],
                                         "operation": "Mul"},
                                        {"elements": [-1, self.variables['cvt_' + pheno][short_code][timestep].name],
                                         "operation": "Mul"},
                                        {"elements": [-1, self.variables['b_' + pheno][short_code][next_timestep].name],
                                         "operation": "Mul"}],
                                    "operation": "Add"
                                })
                        constraints.append(self.constraints['dbc_' + pheno][short_code][timestep])

                    self.variables[signal + '__bio'][short_code][timestep] = tupVariable(
                        _name(signal, '__bio', short_code, timestep))
                    self.variables[signal + '__diffpos'][short_code][timestep] = tupVariable(
                        _name(signal, '__diffpos', short_code, timestep), Bounds(0, 100))
                    self.variables[signal + '__diffneg'][short_code][timestep] = tupVariable(
                        _name(signal, '__diffneg', short_code, timestep), Bounds(0, 100))
                    variables.extend([self.variables[signal + '__bio'][short_code][timestep],
                                      self.variables[signal + '__diffpos'][short_code][timestep],
                                      self.variables[signal + '__diffneg'][short_code][timestep]])

                    # {signal}__conversion*datum = {signal}__bio
                    self.constraints[signal + '__bioc'][short_code][timestep] = tupConstraint(
                        name=_name(signal, '__bioc', short_code, timestep),
                        expr={
                            "elements": [
                                {"elements": [-1, self.variables[signal + '__bio'][short_code][timestep].name],
                                 "operation": "Mul"},
                                {"elements": [self.variables[signal + '__conversion'].name,
                                              values_slice[timestep, signal_column_index]],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })

                    # {speces}_bio + {signal}_diffneg-{signal}_diffpos = sum_k^K(es_k*b_{phenotype})
                    self.constraints[signal + '__diffc'][short_code][timestep] = tupConstraint(
                        name=_name(signal, '__diffc', short_code, timestep),
                        expr={
                            "elements": [
                                self.variables[signal + '__bio'][short_code][timestep].name, *signal_sum,
                                self.variables[signal + '__diffneg'][short_code][timestep].name,
                                {"elements": [-1, self.variables[signal + '__diffpos'][short_code][timestep].name],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })
                    constraints.extend([self.constraints[signal + '__bioc'][short_code][timestep],
                                        self.constraints[signal + '__diffc'][short_code][timestep]])

                    objective.expr.extend([{
                        "elements": [
                            {"elements": [self.parameters['diffpos'],
                                          self.variables[signal + '__diffpos'][short_code][timestep].name],
                             "operation": "Mul"},
                            {"elements": [self.parameters['diffneg'],
                                          self.variables[signal + '__diffneg'][short_code][timestep].name],
                             "operation": "Mul"}],
                        "operation": "Add"
                    }])

        time_4 = process_time()
        print(f'Done with the dbc & diffc loop: {(time_4 - time_3) / 60} min')

        # construct the problem
        # self.dict_problem = OptlangHelper.define_model("MSCommFitting model", variables, constraints, objective)
        self.problem = OptlangHelper.define_model("MSCommFitting model", variables, constraints, objective, True)
        print("Solver:", type(self.problem))
        time_5 = process_time()
        print(f'Done with loading the variables, constraints, and objective: {(time_5 - time_4) / 60} min')

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
        print(f'Done exporting the content: {(time_6 - time_5) / 60} min')

    def compute(self, graphs: list = None, msdb_path:str=None, export_zip_name=None, figures_zip_name=None, publishing=False):
        self.values = {}
        solution = self.problem.optimize()
        # categorize the primal values by trial and time
        if all(np.array(list(self.problem.primal_values.values())) == 0):
            raise NoFluxError("The simulation lacks any flux.")
        for variable, value in self.problem.primal_values.items():
            if 'conversion' not in variable:
                basename, short_code, timestep = variable.split('-')
                time_hr = int(timestep) * self.parameters['data_timestep_hr']
                self.values[short_code] = default_dict_values(self.values, short_code, {})
                self.values[short_code][basename] = default_dict_values(self.values[short_code], basename, {})
                self.values[short_code][basename][time_hr] = value

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
            self.graph(graphs, msdb_path=msdb_path, export_zip_name=figures_zip_name or export_zip_name, publishing=publishing)

    def load_model(self, mscomfit_json_path: str = None, zip_name: str = None, model_to_load: dict = None):
        if zip_name:
            with ZipFile(zip_name, 'r') as zp:
                zp.extract(mscomfit_json_path)
        if mscomfit_json_path:
            with open(mscomfit_json_path, 'r') as mscmft:
                return json.load(mscmft)
        if model_to_load:
            self.problem = Model.from_json(model_to_load)

    def _change_param(self, param, param_time, param_trial):
        if not isinstance(param, dict):
            return param
        if param_time in param:
            if param_trial in param[param_time]:
                return param[param_time][param_trial]
            return param[param_time]
        return param['default']

    def _change_v(self, new_v, mscomfit_json):
        for v_arg in mscomfit_json['constraints']:  # TODO - specify as phenotype-specific, as well as the Km
            v_name, v_time, v_trial = v_arg['name'].split('-')
            if 'gc' in v_name:  # gc = growth constraint
                v_arg['expression']['args'][1]['args'][0]['value'] = self._change_param(new_v, v_time, v_trial)

    def _universalize(self, param, met_id): # , variable):
        if not isinstance(param[met_id], (float, int)):
            return {}
        return {met_id: {species: find_dic_number(param)} for species in self.signal_species.values() if "OD" not in species}
                    # {short_code: {list(timestep_info.keys())[0]: find_dic_number(param)} for short_code, timestep_info in variable.items()}}

    def _michaelis_menten(self, conc, vmax, km):
        return -(conc*vmax)/(km+conc)

    def _align_concentrations(self, met_name, met_id, vmax, km, graphs, mscomfit_json, convergence_tol):
        v, primal_conc = vmax.copy(), {}
        count = 0
        error = convergence_tol + 1
        ### optimizing the new growth rate terms
        while (error > convergence_tol):
            error = 0
            for short_code in self.growth_df.index:
                primal_conc[short_code] = default_dict_values(primal_conc, short_code, {})
                for timestep in self.variables[met_name]:
                    time_hr = int(timestep) * self.parameters['data_timestep_hr']
                    if short_code in primal_conc[short_code]:
                        error += (primal_conc[short_code][timestep] - self.values[short_code][met_name][time_hr]) ** 2
                    #### concentrations from the last simulation calculate a new growth rate
                    primal_conc[short_code][timestep] = self.values[short_code][met_name][time_hr]
                    print("new concentration", primal_conc[short_code][timestep])
                    v[met_id][short_code][timestep] = self._michaelis_menten(
                        primal_conc[short_code][timestep], vmax[met_id][short_code][timestep], km[met_id][short_code][timestep])
                    if v[met_id][short_code][timestep] > 0:
                        logger.critical(f"The growth rate of {v[met_id][short_code][timestep]} will cause "
                                        "an infeasible solution.")
                    print('new growth rate: ', v[met_id][short_code][timestep])
                    count += 1
            self._change_v(v[met_id], mscomfit_json)
            # self._export_model_json(mscomfit_json, mscomfit_json_path)
            self.load_model(model_to_load=mscomfit_json)
            self.compute(graphs)  # , export_zip_name)
            # TODO - the primal values dictionary must be updated with each loop to allow errors to change
            error = (error / count) ** 0.5 if error > 0 else 0
            print("Error:", error)

    def change_parameters(self, cvt=None, cvf=None, diff=None, vmax=None, km=None, graphs: list = None,
                          mscomfit_json_path='mscommfitting.json', primal_values_filename: str = None,
                          export_zip_name=None, extract_zip_name=None, previous_relative_conc: float = None, convergence_tol=0.1):

        # load the model JSON
        vmax, km = vmax or {}, km or {}
        time_1 = process_time()
        if not os.path.exists(mscomfit_json_path):
            extract_zip_name = extract_zip_name or self.zip_name
            mscomfit_json = self.load_model(mscomfit_json_path, zip_name=extract_zip_name)
        else:
            mscomfit_json = self.load_model(mscomfit_json_path)
        time_2 = process_time()
        print(f'Done loading the JSON: {(time_2 - time_1) / 60} min')

        # change objective coefficients
        if any([cvf, cvt, diff]):
            for arg in mscomfit_json['objective']['expression']['args']:
                name, timestep, trial = arg['args'][1]['name'].split('-')
                if cvf and 'cvf' in name:
                    arg['args'][0]['value'] = self._change_param(cvf, timestep, trial)
                if cvt and 'cvt' in name:
                    arg['args'][0]['value'] = self._change_param(cvt, timestep, trial)
                if diff and 'diff' in name:
                    arg['args'][0]['value'] = self._change_param(diff, timestep, trial)

        if km and not vmax:
            raise ParameterError(f'A Vmax must be defined when Km is defined (here {km}).')
        if any([hasattr(self,"rel_final_conc"), hasattr(self,"abs_final_conc"), vmax]):
            # uploads primal values when they are not in RAM
            if not hasattr(self, 'values'):
                with open(primal_values_filename, 'r') as pv:
                    self.values = json.load(pv)
            initial_concentrations = {}; already_constrained = []
            for var in mscomfit_json['variables']:
                if 'cpd' not in var['name']:
                    continue
                met = var.copy()
                met_name, timestep, trial = met['name'].split('-')
                # assign initial concentration
                if timestep == self.timesteps[0]:
                    initial_concentrations[met_name] = met["ub"]
                # assign final concentration
                elif timestep == self.timesteps[-1]:
                    if self.abs_final_conc and dict_keys_exists(self.abs_final_conc, met_name):
                        met['lb'] = met['ub'] = self.abs_final_conc[met_name]
                    elif any([x in met_name for x in self.rel_final_conc]):
                        print("ub 1", met['ub'])
                        met['lb'] = met['ub'] = initial_concentrations[met_name] * self.rel_final_conc[met_name]
                        if previous_relative_conc:
                            met['ub'] /= previous_relative_conc
                            print("ub 2", met['ub'])
                            met['lb'] /= previous_relative_conc
                            print("ub 3", met['lb'])

                if met_name in already_constrained:
                    continue
                already_constrained.append(met_name)
                # confirm that the metabolite was produced during the simulation
                met_id = self._met_id_parser(met_name)
                if met_id not in list(chain(*self.phenotype_met.values())):
                    continue
                if any([isclose(max(list(self.values[trial][met_name].values())), 0)
                        for trial in self.values]):  # TODO - perhaps this should only skip affected trials?
                    print(f"The {met_id} metabolite of interest was not produced "
                          "during the simulation; hence, its does not contribute to growth kinetics.")
                    continue
                # change growth kinetics
                ## defines the Vmax for each metabolite, or distributes a constant Vmax
                vmax.update(self._universalize(vmax, met_id, self.variables[met_name]))
                if km:
                    ## calculate the Michaelis-Menten kinetic rate: vmax*[maltose] / (km+[maltose])
                    km.update(self._universalize(km, met_id, self.variables[met_name]))
                    self._align_concentrations(
                        met_name, met_id, vmax, km, graphs, mscomfit_json, convergence_tol)
                else:
                    self._change_v(vmax[met_id], mscomfit_json)

        # export and load the edited model
        self._export_model_json(mscomfit_json, mscomfit_json_path)
        export_zip_name = export_zip_name or self.zip_name
        with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
            zp.write(mscomfit_json_path)
            os.remove(mscomfit_json_path)
        time_3 = process_time()
        print(f'Done exporting the model: {(time_3 - time_2) / 60} min')
        self.problem = Model.from_json(mscomfit_json)
        time_4 = process_time()
        print(f'Done loading the model: {(time_4 - time_3) / 60} min')

    def parameter_optimization(self, ):
        with ZipFile(self.zip_name, 'r') as zp:
            zp.extract('mscommfitting.json')

        # leverage the newton module for Newton's optimization algorithm

    def _add_plot(self, ax, labels, label, basename, trial, x_axis_split, linestyle="solid"):
        labels.append(label or basename.split('-')[-1])
        ax.plot(list(self.values[trial][basename].keys()),
                list(self.values[trial][basename].values()),
                label=labels[-1], linestyle=linestyle)
        x_ticks = np.around(np.array(list(self.values[trial][basename].keys())), 0)
        ax.set_xticks(x_ticks[::x_axis_split])
        return ax, labels

    def graph(self, graphs, primal_values_filename: str = None, primal_values_zip_path: str = None, msdb_path:str=None,
              export_zip_name: str = None, data_timestep_hr: float = 0.163, publishing: bool = False, title: str = None):
        # define the default timestep ratio as 1
        data_timestep_hr = self.parameters.get('data_timestep_hr', data_timestep_hr)
        timestep_ratio = data_timestep_hr / self.parameters.get('timestep_hr', data_timestep_hr)
        if primal_values_filename:
            if primal_values_zip_path:
                with ZipFile(primal_values_zip_path, 'r') as zp:
                    zp.extract(primal_values_filename)
            with open(primal_values_filename, 'r', encoding='utf-8') as primal:
                self.values = json.load(primal)

        # plot the content for desired trials
        if msdb_path and not hasattr(self, "msdb"):
            from modelseedpy.biochem import from_local
            self.msdb = from_local(msdb_path)
        x_axis_split = int(2 / data_timestep_hr / timestep_ratio)
        self.plots = []
        contents = {"biomass": 'b_', "all_biomass": 'b_', "growth": 'g_', "conc": "c_"}
        linestyle = {"OD":"solid", "ecoli":"dashed", "pf":"dotted"}
        mM_threshold = 1e-3
        for graph_index, graph in enumerate(graphs):
            content = contents.get(graph['content'], graph['content'])
            y_label = 'Variable value'; x_label = 'Time (hr)'
            if any([x in graph['content'] for x in ['biomass', 'OD']]):
                ys = {name: [] for name in self.signal_species.values()}
                if "species" not in graph:
                    graph['species'] = list(self.signal_species.values())
            if "biomass" in graph['content']:
                y_label = 'Biomass concentration (g/L)'
            elif 'growth' in graph['content']:
                y_label = 'Biomass growth (g/hr)'
            # elif 'stress-test' in graph['content']:
            #     content = graph['content'].split('_')[1]
            #     y_label = graph['species']+' coculture %'
            #     x_label = content+' (mM)'
            graph["experimental_data"] = default_dict_values(graph, "experimental_data", False)
            if 'phenotype' in graph and graph['phenotype'] == '*':
                if "species" not in graph:
                    graph['species'] = list(self.signal_species.values())
                graph['phenotype'] = set([col.split('_')[1] for col in self.fluxes_tup.columns
                                          if col.split('_')[0] in graph["species"]])
            if 'species' in graph and graph['species'] == '*':
                graph['species'] = list(self.signal_species.values())
            elif content == "c_" and 'mets' not in graph:
                graph["mets"] = self.mets_to_track
            elif not any(["species" in graph, "mets" in graph]):
                raise ValueError(f"The specified graph {graph} must define species for which data will be plotted.")
            print(f"graph_{graph_index}"); pprint(graph)

            # define figure specifications
            if publishing:
                pyplot.rc('axes', titlesize=20, labelsize=20)
                pyplot.rc('xtick', labelsize=20)
                pyplot.rc('ytick', labelsize=20)
                pyplot.rc('legend', fontsize=18)
            fig, ax = pyplot.subplots(dpi=200, figsize=(11, 7))
            x_ticks = None
            yscale = "linear"

            # define the figure contents
            for trial, basenames in self.values.items():
                if trial not in graph['trial']:
                    continue
                labels = []
                for basename, values in basenames.items():
                    # graph comprehensive overlaid figures of biomass plots
                    if any([x in graph['content'] for x in ['biomass', 'OD']]):
                        if 'b_' in basename:
                            var_name, species, phenotype = basename.split('_')
                            label = f'{species}_biomass (model)'
                            if publishing:
                                # if any([species == species_name for species_name in self.signal_species.values()]):
                                #     break
                                if species == 'ecoli':
                                    species_name = 'E. coli'
                                elif species == 'pf':
                                    species_name = 'P. fluorescens'
                                elif species == 'OD':
                                    species_name = 'Total'
                                label = f'{species_name} total (model)'
                            labels.append({species: label})
                            xs = np.array(list(values.keys()))
                            vals = np.array(list(values.values()))
                            ax.set_xticks(xs[::int(3 / data_timestep_hr / timestep_ratio)])
                            if (any([x in graph['content'] for x in ["total", 'OD']]) or
                                    graph['species'] == list(self.signal_species.values())
                            ):
                                ys['OD'].append(vals)
                                if "OD" not in graph['content']:
                                    ys[species].append(vals)
                        if all([graph['experimental_data'], '__bio' in basename, ]):
                            # any([content in basename])]):  # TODO - any() must include all_biomass and total
                            signal = "_".join([x for x in basename.split('_')[:-1] if x])
                            label = basename
                            if publishing:
                                if self.signal_species[signal] == 'ecoli':
                                    species = 'E. coli'
                                elif self.signal_species[signal] == 'pf':
                                    species = 'P. fluorescens'
                                label = f'Experimental {species} (from {signal})'
                                if 'OD' in signal:
                                    label = 'Experimental total biomass (from OD)'
                            ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split)
                    # graph an aspect of a specific species across all phenotypes
                    if content not in basename:
                        continue
                    if "phenotype" in graph:
                        for specie in graph["species"]:
                            if specie not in basename:
                                continue
                            label = basename.split("_")[-1]
                            style = "solid"
                            # multi-species figures
                            if len(graph["species"]) > 1:
                                label = re.sub(r"(^[a-b]+\_)", "", basename)
                                style = linestyle[specie]
                            if graph['phenotype'] == '*':
                                if 'total' in graph["content"]:  # TODO - this logic appears erroneous by not using _add_plot()
                                    labels = [label]
                                    xs = np.array(list(values.keys()))
                                    ys.append(np.array(list(values.values())))
                                    ax.set_xticks(x_ticks[::int(3 / data_timestep_hr / timestep_ratio)])
                                else:
                                    ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split, style)
                                # print('species content of all phenotypes')
                            # graph all phenotypes
                            elif any([x in basename for x in graph['phenotype']]):
                                ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split, style)
                                # print('all content over all phenotypes')
                            break
                    # graph media concentration plots
                    elif "mets" in graph and all([
                        any([x in basename for x in graph["mets"]]), 'EX_' in basename]
                    ):
                        if not any(np.array(list(self.values[trial][basename].values())) > mM_threshold):
                            continue
                        label=self.msdb.compounds.get_by_id(re.search(r"(cpd\d+)", basename).group()).name
                        ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split)
                        yscale = "log"
                        y_label = 'Concentration (mM)'
                        # print('media concentration')

                if labels:  # this assesses whether a graph was constructed
                    if any([x in graph['content'] for x in ['OD', 'biomass', 'total']]):
                        labeled_species = [label for label in labels if isinstance(label, dict)]
                        for name, vals in ys.items():
                            if not vals:
                                continue
                            label = f'{name}_biomass (model)'
                            if labeled_species:
                                for label_specie in labeled_species:
                                    if name in label_specie:
                                        label = label_specie[name]
                                        break
                            style = "solid" if (len(graph["species"]) < 1 or name not in linestyle
                                                ) else linestyle[name]
                            style = "dashdot" if "model" in label else style
                            style = "solid" if ("OD" in name and not graph["experimental_data"]) else style
                            ax.plot(xs.astype(np.float32), sum(vals), label=label, linestyle=style)

                    phenotype_id = "" if "phenotype" not in graph else graph['phenotype']
                    if "phenotype" in graph and not isinstance(graph['phenotype'], str):
                        phenotype_id = f"{','.join(graph['phenotype'])} phenotypes"

                    if "mets" not in graph and content != "c_":
                        species_id = graph["species"] if isinstance(graph["species"], str) else ",".join(graph["species"])
                        if "species" in graph and graph['species'] == list(self.signal_species.values()):
                            species_id = 'all species'
                        else:
                            phenotype_id = f"{','.join(graph['phenotype'])} phenotypes"
                        if species_id == "all species" and not phenotype_id:
                            phenotype_id = ','.join(graph['species'])

                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_yscale(yscale)
                    if "mets" in graph:
                        ax.set_ylim(mM_threshold)
                    ax.grid(axis="y")
                    if len(labels) > 1:
                        ax.legend()
                    if not publishing:
                        if not title:
                            org_content = content if content not in contents.values() else list(
                                contents.keys())[list(contents.values()).index(content)]
                            this_title = f'{org_content} of {species_id} ({phenotype_id}) in the {trial} trial'
                            if content == "c_":
                                this_title = f"{org_content} in the {trial} trial"
                            ax.set_title(this_title)
                        else:
                            ax.set_title(title)
                    fig_name = f'{"_".join([trial, species_id, phenotype_id, content])}.jpg'
                    if "mets" in graph:
                        fig_name = f"{trial}_{','.join(graph['mets'])}_c.jpg"
                    fig.savefig(fig_name)
                    self.plots.append(fig_name)

        # export the figures with other simulation content
        if export_zip_name:
            with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
                for plot in self.plots:
                    zp.write(plot)
                    os.remove(plot)


    #################### ENGINEERING PHASE METHODS ####################

    def engineering(self):
        if not hasattr(self, "problem"):
            self.fit() # TODO - accommodate both fitting a new model and loading an existing model

        # This will capture biomass variables at all times and trials, which seems undesirable
        self.problem.objective = Objective(sum([x for x in self.problem.variables if "bio" in x.name]))

        # Use a community COBRA model and CommKinetics with the fitted kinetic parameters?

    def _add_phenotypes(self):
        pass



    def _change_obj(self):
        pass


