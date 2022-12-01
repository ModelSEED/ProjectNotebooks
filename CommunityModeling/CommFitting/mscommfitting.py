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
from deepdiff import DeepDiff
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

def dic_keys(dic):
    keys = []
    if isinstance(dic, dict):
        for key, value in dic.items():
            keys.append(key)
            keys.extend(dic_keys(value))
    return keys

# define data objects
def _name(name, suffix, short_code, timestep, names):
    name = '-'.join([x for x in list(map(str, [name + suffix, short_code, timestep])) if x])
    if name not in names:
        names.append(name)
        return name
    else:
        raise ObjectAlreadyDefinedError(f"The object {name} is already defined for the problem.")

def _export_model_json(json_model, path):
    with open(path, 'w') as lp:
        json.dump(json_model, lp, indent=3)

def _met_id_parser(met):
    met_id = re.sub('(\_\w\d+)', '', met)
    met_id = met_id.replace('EX_', '', 1)
    met_id = met_id.replace('c_', '', 1)
    return met_id

# define an entity as a variable or a constant
def _obj_val(primal, name, pheno, short_code, timestep, bounds, data_timestep_hr, names):
    time_hr = int(timestep) * data_timestep_hr
    return tupVariable(_name(name, pheno, short_code, timestep, names),
                       Bounds=bounds) if not primal else primal[short_code][name+pheno][time_hr]

def _michaelis_menten(conc, vmax, km):
    return -(conc*vmax)/(km+conc)

# parse primal values for use in the optimization loops
def parse_primals(primal_values, entity_label):
    distinguished_primals = {}
    for trial, entities in primal_values.items():
        distinguished_primals[trial] = {}
        for entity, times in entities.items():
            if entity_label in entity:
                distinguished_primals[trial][entity] = {time:value for time, value in times.items()}
    return distinguished_primals

def signal_species(signal):
    return signal.split(":")[0].replace(" ", "_")

class CommPhitting:

    def __init__(self, fluxes_df, carbon_conc, media_conc, growth_df=None, experimental_metadata=None):
        self.parameters, self.variables, self.constraints, = {}, {}, {}
        self.zipped_output, self.plots = [], []
        self.fluxes_tup = FBAHelper.parse_df(fluxes_df)
        self.growth_df = growth_df; self.experimental_metadata = experimental_metadata
        self.carbon_conc = carbon_conc; self.media_conc = media_conc
        self.names = []

    #################### FITTING PHASE METHODS ####################
    def fit_kinetics(self, parameters:dict=None, mets_to_track: list = None, rel_final_conc:dict=None, zero_start:list=None,
                     abs_final_conc:dict=None, graphs: list = None, data_timesteps: dict = None, msdb_path:str=None,
                     export_zip_name: str = None, export_parameters: bool = True, export_lp: str = 'CommPhitting.lp',
                     figures_zip_name:str=None, publishing:bool=False):
        # solve for biomass b with parameterized growth rate
        simulation = new_simulation = CommPhitting(
            self.fluxes_tup, self.carbon_conc, self.media_conc, self.growth_df, self.experimental_metadata)
        new_simulation.define_problem(parameters, mets_to_track, rel_final_conc, zero_start,
                                      abs_final_conc, data_timesteps, export_zip_name, export_parameters,
                                      'solveBiomass0.lp', None)
        new_simulation.compute(primals_export_path="b_primals0.json")
        b_values = parse_primals(new_simulation.values, "b_")
        ## create a ghost b_param dictionary to deviate from the primals and enter the while loop
        b_param = {}
        for trial, entities in b_values.items():
            b_param[trial] = {}
            for entity, times in entities.items():
                b_param[trial][entity] = {time:2*float(value) for time, value in times.items()}
        # iteratively optimize growth rate and biomass until the biomass converges
        count = 1
        while not all([isclose(b_param[trial][entity][time], b_values[trial][entity][time])
                       for trial, entities in b_param.items()
                       for entity, times in entities.items() for time in times
                       if "b_" in entity]) and count < 11:
            print(f"\nFirst kinetics optimization phase, iteration: {count}")
            ## solve for growth rate v with solved b
            b_param = b_values.copy()
            print(DeepDiff(b_param, b_values))
            simulation = new_simulation1 = CommPhitting(
                self.fluxes_tup, self.carbon_conc, self.media_conc, self.growth_df, self.experimental_metadata)
            new_simulation1.define_problem(parameters, mets_to_track, rel_final_conc, zero_start, abs_final_conc,
                                           data_timesteps, export_zip_name, export_parameters,
                                           f'solveGrowthRates{count}.lp', b_param)
            try:
                new_simulation1.compute(primals_export_path=f"v_primals{count}.json")
                ### parse and homogenize the growth rates
                v_param = {k:val for k,val in new_simulation1.values.items() if "v_" in k}
            except FeasibilityError as e:
                print(e)
                if "new_simulation2" not in locals():
                    raise ValueError("The kinetic optimization immediately failed with the first optimized biomasses.")
                simulation = new_simulation2
                break

            ## solve for biomass b with parameterized growth rate
            print("\nComplete growth rate optimization")
            print(v_param)
            simulation = new_simulation2 = CommPhitting(
                self.fluxes_tup, self.carbon_conc, self.media_conc, self.growth_df, self.experimental_metadata)
            new_simulation2.define_problem(parameters, mets_to_track, rel_final_conc, zero_start, abs_final_conc,
                                           data_timesteps, export_zip_name, export_parameters,
                                           f'solveBiomass{count}.lp', v_param)
            try:
                new_simulation2.compute(primals_export_path=f"b_primals{count}.json")
                b_values = parse_primals(new_simulation2.values, "b_")
            except FeasibilityError as e:
                print(e)
                simulation = new_simulation1
                break
            ## track iteration progress
            print("\nComplete biomass optimization")
            print(DeepDiff(b_param, b_values))
            count += 1

        # simulate the last problem with the converged parameters to render the figures
        if count == 11:
            print("The kinetics optimization reached the iteration limit and exited.")
        simulation.compute(graphs, msdb_path, export_zip_name, figures_zip_name, publishing)
        # export the converged growth rates
        if "v_param" not in locals():
            raise ValueError(f"The b_param value was not properly overwritten; "
                             f"hence, the while loop was never initiated.")
        return v_param

    def fit(self, parameters:dict=None, mets_to_track: list = None, rel_final_conc:dict=None, zero_start:list=None,
            abs_final_conc:dict=None, graphs: list = None, data_timesteps: dict = None, msdb_path:str=None,
            export_zip_name: str = None, export_parameters: bool = True, export_lp: str = 'CommPhitting.lp', figures_zip_name:str=None,
            publishing:bool=False):
        self.define_problem(parameters, mets_to_track, rel_final_conc, zero_start, abs_final_conc, data_timesteps, export_zip_name,
                            export_parameters, export_lp)
        self.compute(graphs, msdb_path, export_zip_name, figures_zip_name, publishing)

    def _update_problem(self, contents: Iterable):
        for content in contents:
            self.problem.add(content)
            self.problem.update()

    def define_problem(self, parameters=None, mets_to_track = None, rel_final_conc=None, zero_start=None, abs_final_conc=None,
                       data_timesteps=None, export_zip_name: str=None, export_parameters: bool=True, export_lp: str='CommPhitting.lp',
                       primal_values=None):
        # parse the growth data
        growth_tup = FBAHelper.parse_df(self.growth_df)
        self.species_list = [signal_species(signal) for signal in growth_tup.columns[3:]]
        num_sorted = np.sort(np.array([int(obj[1:]) for obj in set(growth_tup.index)]))
        # TODO - short_codes must be distinguished for different conditions
        unique_short_codes = [f"{growth_tup.index[0][0]}{num}" for num in map(str, num_sorted)]
        time_column_index = growth_tup.columns.index("Time (s)")
        full_times = growth_tup.values[:, time_column_index]
        self.times = {short_code: trial_contents(short_code, growth_tup.index, full_times) for short_code in unique_short_codes}

        # define default values
        # TODO render bcv and cvmin dependent upon temperature, and possibly trained on Carlson's data
        # TODO find the lowest cvmin and bcv that fit the experimental data
        parameters, data_timesteps = parameters or {}, data_timesteps or {}
        self.parameters["data_timestep_hr"] = np.mean(np.diff(np.array(list(self.times.values())).flatten())) / hour
        self.parameters.update({
            "timestep_hr": self.parameters['data_timestep_hr'],  # Simulation timestep magnitude in hours
            "cvct": 1, "cvcf": 1,  # Minimization coefficients of the phenotype conversion to and from the stationary phase.
            "bcv": 0.1,  # The highest fraction of species biomass that can change phenotypes in a timestep
            "cvmin": 0,  # The lowest value the limit on phenotype conversion goes,
            "v": 0.33,  # The kinetics constant that is externally adjusted
            'diffpos': 1, 'diffneg': 1,
            # diffpos and diffneg coefficients that weight difference between experimental and predicted biomass
        })
        self.parameters.update(parameters)
        self.parameters.update(self._universalize(self.parameters,"v"))
        default_carbon_sources = ["cpd00076", "cpd00179", "cpd00027"]  # sucrose, maltose, glucose
        self.rel_final_conc = rel_final_conc or {c:1 for c in default_carbon_sources}
        self.abs_final_conc = abs_final_conc or {}
        self.mets_to_track = mets_to_track or self.fluxes_tup.index if not isinstance(
            rel_final_conc, dict) else list(self.rel_final_conc.keys())
        zero_start = zero_start or []
        # define the varying entities
        # b_values = {}
        v_values = []

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
            met_id = _met_id_parser(met)
            if self.mets_to_track and (met_id == 'cpd00001' or met_id not in self.mets_to_track):
                continue
            self.variables["c_" + met] = {}; self.constraints['dcc_' + met] = {}

            # define the growth rate for each metabolite and concentrations
            if "Vmax" and "Km" in self.parameters:
                self.parameters["Vmax"].update(self._universalize(self.parameters["Vmax"], met_id))
                self.parameters["Km"].update(self._universalize(self.parameters["Km"], met_id))
            for short_code in unique_short_codes:
                self.variables["c_" + met][short_code] = {}; self.constraints['dcc_' + met][short_code] = {}
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for index, timestep in enumerate(timesteps):
                    ## define the concentration variables
                    conc_var = tupVariable(_name("c_", met, short_code, timestep, self.names))
                    ## constrain initial time concentrations to the media or a large default
                    if index == 0 and not 'bio' in met_id:
                        initial_val = 100 if not self.media_conc or met_id not in self.media_conc else self.media_conc[met_id]
                        initial_val = 0 if met_id in zero_start else initial_val
                        if dict_keys_exists(self.carbon_conc, met_id, short_code):
                            initial_val = self.carbon_conc[met_id][short_code]
                        conc_var = conc_var._replace(bounds=Bounds(initial_val, initial_val))
                    ## mandate complete carbon consumption
                    if timestep == timesteps[-1] and any([
                        met_id in self.rel_final_conc, met_id in self.abs_final_conc
                    ]):
                        if met_id in self.rel_final_conc:
                            final_bound = self.variables["c_" + met][short_code][1].bounds.lb * self.rel_final_conc[met_id]
                        if met_id in self.abs_final_conc: # this intentionally overwrites rel_final_conc
                            final_bound = self.abs_final_conc[met_id]
                        conc_var = conc_var._replace(bounds=Bounds(0, final_bound))
                        if met_id in zero_start:
                            conc_var = conc_var._replace(bounds=Bounds(final_bound, final_bound))
                    self.variables["c_" + met][short_code][timestep] = conc_var
                    variables.append(self.variables["c_" + met][short_code][timestep])
        for signal in growth_tup.columns[3:]:
            for pheno in self.fluxes_tup.columns:
                if signal_species(signal) in pheno:
                    self.constraints['dbc_' + pheno] = {short_code: {} for short_code in unique_short_codes}

        # define growth and biomass variables and constraints
        # self.parameters["v"] = {met_id:{species:_michaelis_menten()}}
        for pheno in self.fluxes_tup.columns:
            # print(f"\n\n{pheno}\n==================\n")
            # b_values[pheno], v_values[pheno] = {}, {}

            self.variables['cvt_' + pheno] = {}; self.variables['cvf_' + pheno] = {}
            self.variables['b_' + pheno] = {}; self.variables['g_' + pheno] = {}
            # self.variables['v_' + pheno] = {}
            self.constraints['gc_' + pheno] = {}; self.constraints['cvc_' + pheno] = {}
            # self.constraints['vc_' + pheno] = {}
            for short_code in unique_short_codes:
                # b_values[pheno][short_code], v_values[pheno][short_code] = {}, {}

                self.variables['cvt_' + pheno][short_code] = {}; self.variables['cvf_' + pheno][short_code] = {}
                self.variables['b_' + pheno][short_code] = {}; self.variables['g_' + pheno][short_code] = {}
                # self.variables['v_' + pheno][short_code] = {}
                self.constraints['gc_' + pheno][short_code] = {}
                self.constraints['cvc_' + pheno][short_code] = {}
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for timestep in timesteps:
                    timestep = int(timestep)
                    # predicted biomass abundance and biomass growth
                    ## define the biomass variable or primal value
                    self.variables['b_' + pheno][short_code][timestep] = tupVariable(
                        _name("b_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
                    variables.append(self.variables['b_' + pheno][short_code][timestep])
                    time_hr = timestep * self.parameters['data_timestep_hr']
                    b_value = self.variables['b_' + pheno][short_code][timestep].name
                    # print(b_value)
                    # b_values[pheno][short_code][timestep] = b_value

                    ## define the growth rate variable or primal value
                    species, phenotype = pheno.split("_")
                    v_value = self.parameters["v"][species][phenotype]
                    if primal_values:
                        if 'v_' + pheno in primal_values:
                            ## universalize the phenotype growth rates for all codes and timesteps
                            v_value = primal_values['v_' + pheno]
                        elif 'b_' + pheno in primal_values[short_code]:
                            if 'v_' + pheno not in self.variables:
                                self.variables['v_' + pheno] = tupVariable(
                                    _name("v_", pheno, "", "", self.names), Bounds(0, 1000))
                                variables.append(self.variables['v_' + pheno])
                            v_value = self.variables['v_' + pheno].name
                            # v_values.append(v_value)
                            b_value = primal_values[short_code]['b_' + pheno][time_hr]

                    self.variables['g_' + pheno][short_code][timestep] = tupVariable(
                        _name("g_", pheno, short_code, timestep, self.names))
                    variables.append(self.variables['g_' + pheno][short_code][timestep])

                    if 'stationary' in pheno:
                        continue
                    # if "pf" and "4HB" in pheno:
                        # ic(pheno, v_value, b_value)
                    # the conversion rates to and from the stationary phase
                    self.variables['cvt_' + pheno][short_code][timestep] = tupVariable(
                        _name("cvt_", pheno, short_code, timestep, self.names), Bounds(0, 100))
                    self.variables['cvf_' + pheno][short_code][timestep] = tupVariable(
                        _name("cvf_", pheno, short_code, timestep, self.names), Bounds(0, 100))
                    variables.extend([self.variables['cvf_' + pheno][short_code][timestep],
                                      self.variables['cvt_' + pheno][short_code][timestep]])

                    # cvt <= bcv*b_{pheno} + cvmin
                    self.constraints['cvc_' + pheno][short_code][timestep] = tupConstraint(
                        _name('cvc_', pheno, short_code, timestep, self.names), (0, None), {
                            "elements": [
                                {"elements": [
                                    -1, self.variables['cvt_' + pheno][short_code][timestep].name],
                                    "operation": "Mul"}],
                            "operation": "Add"
                        })
                    # biomass_term = [self.parameters['bcv']*b_value + self.parameters['cvmin']] if isnumber(b_value) else [
                    biomass_term = [
                        self.parameters['cvmin'], {"elements": [self.parameters['bcv'], b_value], "operation": "Mul"}]
                    self.constraints['cvc_' + pheno][short_code][timestep].expr["elements"].extend(biomass_term)

                    # v_{pheno} = -(Vmax_{met, species} * c_{met}) / (Km_{met, species} + c_{met})
                    # self.constraints['vc_' + pheno][short_code][timestep] = tupConstraint(
                    #     name=_name('vc_', pheno, short_code, timestep, self.names),
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
                        name=_name('gc_', pheno, short_code, timestep, self.names),
                        expr={
                            "elements": [
                                self.variables['g_' + pheno][short_code][timestep].name,
                                {"elements": [-1, v_value, b_value],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })

                    # v_{pheno}_t,i = v_{pheno}_(t+1),(i+1), for all t in T and i in I
                    # self.constraints['vc_' + pheno][short_code][timestep] = tupConstraint(
                    #     name=_name('vc_', pheno, short_code, timestep, self.names),
                    #     expr={
                    #         "elements": [v_values],
                    #         "operation": "Add"
                    #     })

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
            met_id = _met_id_parser(met)
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
                        name=_name("dcc_", met, short_code, timestep, self.names),
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
                        val = 0
                        if signal.split(":")[0].replace(" ", "_") in pheno:
                            val = 1
                            # if not isnumber(b_values[pheno][short_code][timestep]):
                            signal_sum.append({"operation": "Mul", "elements": [
                                -1, self.variables['b_' + pheno][short_code][timestep].name]})
                            # else:
                            #     signal_sum.append(-b_values[pheno][short_code][timestep])
                        ### total_biomass.append(self.variables["b_"+pheno][short_code][timestep].name)
                        if all(['OD' not in signal,signal_species(signal) in pheno, 'stationary' not in pheno]):
                            from_sum.append({"operation": "Mul", "elements": [
                                -val, self.variables["cvf_" + pheno][short_code][timestep].name]})
                            to_sum.append({"operation": "Mul", "elements": [
                                val, self.variables["cvt_" + pheno][short_code][timestep].name]})
                    for pheno in self.fluxes_tup.columns:
                        if 'OD' in signal or signal_species(signal) not in pheno:
                            continue
                        # print(pheno, timestep, b_values[pheno][short_code][timestep], b_values[pheno][short_code][next_timestep])
                        if "stationary" in pheno:
                            # b_{phenotype} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) = b+1_{phenotype}
                            self.constraints['dbc_' + pheno][short_code][timestep] = tupConstraint(
                                name=_name("dbc_", pheno, short_code, timestep, self.names),
                                expr={
                                    "elements": [
                                        {"elements": [-1, self.variables['b_' + pheno][short_code][next_timestep].name],
                                         "operation": "Mul"}, *from_sum, *to_sum],
                                    "operation": "Add"
                                })
                        else:
                            # b_{phenotype} + dt/2*(g_{phenotype} + g+1_{phenotype}) + cvf-cvt = b+1_{phenotype}
                            self.constraints['dbc_' + pheno][short_code][timestep] = tupConstraint(
                                name=_name("dbc_", pheno, short_code, timestep, self.names),
                                expr={
                                    "elements": [
                                        self.variables['cvf_' + pheno][short_code][timestep].name,
                                        {"elements": [half_dt, self.variables['g_' + pheno][short_code][timestep].name],
                                         "operation": "Mul"},
                                        {"elements": [half_dt, self.variables['g_' + pheno][short_code][next_timestep].name],
                                         "operation": "Mul"},
                                        {"elements": [-1, self.variables['cvt_' + pheno][short_code][timestep].name],
                                         "operation": "Mul"}],
                                    "operation": "Add"
                                })
                        # if not isnumber(self.variables['b_' + pheno][short_code][timestep]):
                        biomass_term = [self.variables['b_' + pheno][short_code][timestep].name, {
                            "elements": [-1, self.variables['b_' + pheno][short_code][next_timestep].name], "operation": "Mul"}]
                        # else:
                        #     biomass_term = [b_values[pheno][short_code][timestep]-b_values[pheno][short_code][next_timestep]]
                        # print(biomass_term)
                        self.constraints['dbc_' + pheno][short_code][timestep].expr["elements"].extend(biomass_term)
                        constraints.append(self.constraints['dbc_' + pheno][short_code][timestep])

                    self.variables[signal + '__bio'][short_code][timestep] = tupVariable(
                        _name(signal, '__bio', short_code, timestep, self.names))
                    self.variables[signal + '__diffpos'][short_code][timestep] = tupVariable(
                        _name(signal, '__diffpos', short_code, timestep, self.names), Bounds(0, 100))
                    self.variables[signal + '__diffneg'][short_code][timestep] = tupVariable(
                        _name(signal, '__diffneg', short_code, timestep, self.names), Bounds(0, 100))
                    variables.extend([self.variables[signal + '__bio'][short_code][timestep],
                                      self.variables[signal + '__diffpos'][short_code][timestep],
                                      self.variables[signal + '__diffneg'][short_code][timestep]])

                    # {signal}__conversion*datum = {signal}__bio
                    self.constraints[signal + '__bioc'][short_code][timestep] = tupConstraint(
                        name=_name(signal, '__bioc', short_code, timestep, self.names),
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
                        name=_name(signal, '__diffc', short_code, timestep, self.names),
                        expr={
                            "elements": [
                                self.variables[signal + '__bio'][short_code][timestep].name,
                                self.variables[signal + '__diffneg'][short_code][timestep].name,
                                {"elements": [-1, self.variables[signal + '__diffpos'][short_code][timestep].name],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })
                    if all([not isnumber(val) for val in signal_sum]):
                        self.constraints[signal + "__diffc"][short_code][timestep].expr["elements"].extend(signal_sum)
                    elif all([isnumber(val) for val in signal_sum]):
                        self.constraints[signal + "__diffc"][short_code][timestep].expr["elements"].append(sum(signal_sum))
                    else:
                        raise ValueError(f"The {signal_sum} value has unexpected contents.")
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
        self.problem = OptlangHelper.define_model("CommPhitting model", variables, constraints, objective, True)
        print("Solver:", type(self.problem))
        time_5 = process_time()
        print(f'Done with loading the variables, constraints, and objective: {(time_5 - time_4) / 60} min')

        # print contents
        if export_parameters:
            self.zipped_output.append('parameters.csv')
            DataFrame(data=list(self.parameters.values()), index=list(self.parameters.keys()), columns=['values']).to_csv('parameters.csv')
        if export_lp:
            self.zipped_output.extend([export_lp, 'CommPhitting.json'])
            with open(export_lp, 'w') as lp:
                lp.write(self.problem.to_lp())
            _export_model_json(self.problem.to_json(), 'CommPhitting.json')
        if export_zip_name:
            self.zip_name = export_zip_name
            sleep(2)
            with ZipFile(self.zip_name, 'w', compression=ZIP_LZMA) as zp:
                for file in self.zipped_output:
                    zp.write(file)
                    os.remove(file)

        time_6 = process_time()
        print(f'Done exporting the content: {(time_6 - time_5) / 60} min')

    def compute(self, graphs: list = None, msdb_path:str=None, export_zip_name=None, figures_zip_name=None, publishing=False,
                primals_export_path:str = "primal_values.json"):
        self.values = {}
        solution = self.problem.optimize()
        # categorize the primal values by trial and time
        if all(np.array(list(self.problem.primal_values.values())) == 0):
            raise NoFluxError("The simulation lacks any flux.")
        for variable, value in self.problem.primal_values.items():
            if "v_" in variable:
                self.values[variable] = value
            elif 'conversion' not in variable:
                basename, short_code, timestep = variable.split('-')
                time_hr = int(timestep) * self.parameters['data_timestep_hr']
                self.values[short_code] = default_dict_values(self.values, short_code, {})
                self.values[short_code][basename] = default_dict_values(self.values[short_code], basename, {})
                self.values[short_code][basename][time_hr] = value

        # export the processed primal values for graphing
        with open(primals_export_path, 'w') as out:
            json.dump(self.values, out, indent=3)
        if not export_zip_name:
            if hasattr(self, 'zip_name'):
                export_zip_name = self.zip_name
        if export_zip_name:
            with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
                zp.write(primals_export_path)
                os.remove(primals_export_path)
        if "optimal" not in solution:
            raise FeasibilityError(f'The solution is sub-optimal, with a(n) {solution} status.')

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

    # def _change_param(self, param, param_time, param_trial):
    #     if not isinstance(param, dict):
    #         return param
    #     if param_time in param:
    #         if param_trial in param[param_time]:
    #             return param[param_time][param_trial]
    #         return param[param_time]
    #     return param['default']

    # def _change_v(self, new_v, mscomfit_json):
    #     for v_arg in mscomfit_json['constraints']:  # TODO - specify as phenotype-specific, as well as the Km
    #         v_name, v_time, v_trial = v_arg['name'].split('-')
    #         if 'gc' in v_name:  # gc = growth constraint
    #             v_arg['expression']['args'][1]['args'][0]['value'] = self._change_param(new_v, v_time, v_trial)

    @staticmethod
    def assign_values(param, var, next_dimension):
        dic = {var: {}}
        for dim1, dim2_list in next_dimension.items():
            dic[var][dim1] = {dim2: param for dim2 in dim2_list}
        return dic

    def _universalize(self, param, var, next_dimension=None):
        if not next_dimension:
            next_dimension = {}
            for organism in self.fluxes_tup.columns:
                species, pheno = organism.split("_")
                if species in next_dimension:
                    next_dimension[species].append(pheno)
                else:
                    next_dimension[species] = [pheno]
        if isnumber(param):
            return CommPhitting.assign_values(param, var, next_dimension)
        elif isnumber(param[var]):
            return CommPhitting.assign_values(param[var], var, next_dimension)
        elif isinstance(param[var], dict):
            dic = {var:{}}
            for dim1, dim2_list in next_dimension.items():
                dic[var][dim1] = {dim2: param[var][dim1] for dim2 in dim2_list}
            return dic
        else:
            logger.critical(f"The param (with keys {dic_keys(param)}) and var {var} are not amenable"
                            f" with the parameterizing a universal value.")
                    # {short_code: {list(timestep_info.keys())[0]: find_dic_number(param)} for short_code, timestep_info in variable.items()}}

    # def _align_concentrations(self, met_name, met_id, vmax, km, graphs, mscomfit_json, convergence_tol):
    #     v, primal_conc = vmax.copy(), {}
    #     count = 0
    #     error = convergence_tol + 1
    #     ### optimizing the new growth rate terms
    #     while (error > convergence_tol):
    #         error = 0
    #         for short_code in self.growth_df.index:
    #             primal_conc[short_code] = default_dict_values(primal_conc, short_code, {})
    #             for timestep in self.variables[met_name]:
    #                 time_hr = int(timestep) * self.parameters['data_timestep_hr']
    #                 if short_code in primal_conc[short_code]:
    #                     error += (primal_conc[short_code][timestep] - self.values[short_code][met_name][time_hr]) ** 2
    #                 #### concentrations from the last simulation calculate a new growth rate
    #                 primal_conc[short_code][timestep] = self.values[short_code][met_name][time_hr]
    #                 print("new concentration", primal_conc[short_code][timestep])
    #                 v[met_id][short_code][timestep] = _michaelis_menten(
    #                     primal_conc[short_code][timestep], vmax[met_id][short_code][timestep], km[met_id][short_code][timestep])
    #                 if v[met_id][short_code][timestep] > 0:
    #                     logger.critical(f"The growth rate of {v[met_id][short_code][timestep]} will cause "
    #                                     "an infeasible solution.")
    #                 print('new growth rate: ', v[met_id][short_code][timestep])
    #                 count += 1
    #         self._change_v(v[met_id], mscomfit_json)
    #         # _export_model_json(mscomfit_json, mscomfit_json_path)
    #         self.load_model(model_to_load=mscomfit_json)
    #         self.compute(graphs)  # , export_zip_name)
    #         # TODO - the primal values dictionary must be updated with each loop to allow errors to change
    #         error = (error / count) ** 0.5 if error > 0 else 0
    #         print("Error:", error)

    # def change_parameters(self, cvt=None, cvf=None, diff=None, vmax=None, km=None, graphs: list = None,
    #                       mscomfit_json_path='CommPhitting.json', primal_values_filename: str = None,
    #                       export_zip_name=None, extract_zip_name=None, previous_relative_conc: float = None, convergence_tol=0.1):
    #
    #     # load the model JSON
    #     vmax, km = vmax or {}, km or {}
    #     time_1 = process_time()
    #     if not os.path.exists(mscomfit_json_path):
    #         extract_zip_name = extract_zip_name or self.zip_name
    #         mscomfit_json = self.load_model(mscomfit_json_path, zip_name=extract_zip_name)
    #     else:
    #         mscomfit_json = self.load_model(mscomfit_json_path)
    #     time_2 = process_time()
    #     print(f'Done loading the JSON: {(time_2 - time_1) / 60} min')
    #
    #     # change objective coefficients
    #     if any([cvf, cvt, diff]):
    #         for arg in mscomfit_json['objective']['expression']['args']:
    #             name, timestep, trial = arg['args'][1]['name'].split('-')
    #             if cvf and 'cvf' in name:
    #                 arg['args'][0]['value'] = self._change_param(cvf, timestep, trial)
    #             if cvt and 'cvt' in name:
    #                 arg['args'][0]['value'] = self._change_param(cvt, timestep, trial)
    #             if diff and 'diff' in name:
    #                 arg['args'][0]['value'] = self._change_param(diff, timestep, trial)
    #
    #     if km and not vmax:
    #         raise ParameterError(f'A Vmax must be defined when Km is defined (here {km}).')
    #     if any([hasattr(self,"rel_final_conc"), hasattr(self,"abs_final_conc"), vmax]):
    #         # uploads primal values when they are not in RAM
    #         if not hasattr(self, 'values'):
    #             with open(primal_values_filename, 'r') as pv:
    #                 self.values = json.load(pv)
    #         initial_concentrations = {}; already_constrained = []
    #         for var in mscomfit_json['variables']:
    #             if 'cpd' not in var['name']:
    #                 continue
    #             met = var.copy()
    #             met_name, timestep, trial = met['name'].split('-')
    #             # assign initial concentration
    #             if timestep == self.timesteps[0]:
    #                 initial_concentrations[met_name] = met["ub"]
    #             # assign final concentration
    #             elif timestep == self.timesteps[-1]:
    #                 if self.abs_final_conc and dict_keys_exists(self.abs_final_conc, met_name):
    #                     met['lb'] = met['ub'] = self.abs_final_conc[met_name]
    #                 elif any([x in met_name for x in self.rel_final_conc]):
    #                     print("ub 1", met['ub'])
    #                     met['lb'] = met['ub'] = initial_concentrations[met_name] * self.rel_final_conc[met_name]
    #                     if previous_relative_conc:
    #                         met['ub'] /= previous_relative_conc
    #                         print("ub 2", met['ub'])
    #                         met['lb'] /= previous_relative_conc
    #                         print("ub 3", met['lb'])
    #
    #             if met_name in already_constrained:
    #                 continue
    #             already_constrained.append(met_name)
    #             # confirm that the metabolite was produced during the simulation
    #             met_id = _met_id_parser(met_name)
    #             if met_id not in list(chain(*self.phenotype_met.values())):
    #                 continue
    #             if any([isclose(max(list(self.values[trial][met_name].values())), 0)
    #                     for trial in self.values]):  # TODO - perhaps this should only skip affected trials?
    #                 print(f"The {met_id} metabolite of interest was not produced "
    #                       "during the simulation; hence, its does not contribute to growth kinetics.")
    #                 continue
    #             # change growth kinetics
    #             ## defines the Vmax for each metabolite, or distributes a constant Vmax
    #             vmax.update(self._universalize(vmax, met_id, self.variables[met_name]))
    #             if km:
    #                 ## calculate the Michaelis-Menten kinetic rate: vmax*[maltose] / (km+[maltose])
    #                 km.update(self._universalize(km, met_id, self.variables[met_name]))
    #                 self._align_concentrations(
    #                     met_name, met_id, vmax, km, graphs, mscomfit_json, convergence_tol)
    #             else:
    #                 self._change_v(vmax[met_id], mscomfit_json)
    #
    #     # export and load the edited model
    #     _export_model_json(mscomfit_json, mscomfit_json_path)
    #     export_zip_name = export_zip_name or self.zip_name
    #     with ZipFile(export_zip_name, 'a', compression=ZIP_LZMA) as zp:
    #         zp.write(mscomfit_json_path)
    #         os.remove(mscomfit_json_path)
    #     time_3 = process_time()
    #     print(f'Done exporting the model: {(time_3 - time_2) / 60} min')
    #     self.problem = Model.from_json(mscomfit_json)
    #     time_4 = process_time()
    #     print(f'Done loading the model: {(time_4 - time_3) / 60} min')

    # def parameter_optimization(self, ):
    #     with ZipFile(self.zip_name, 'r') as zp:
    #         zp.extract('CommPhitting.json')

        # leverage the newton module for Newton's optimization algorithm

    def _add_plot(self, ax, labels, label, basename, trial, x_axis_split, linestyle="solid", scatter=False):
        labels.append(label or basename.split('-')[-1])
        if scatter:
            ax.scatter(list(self.values[trial][basename].keys()),
                       list(self.values[trial][basename].values()),
                       s=10, label=labels[-1])
        else:
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
                ys = {name: [] for name in self.species_list}
                ys.update({"OD":[]})
                if "species" not in graph:
                    graph['species'] = self.species_list
            if "biomass" in graph['content']:
                y_label = 'Biomass concentration (g/L)'
            elif 'growth' in graph['content']:
                y_label = 'Biomass growth (g/hr)'
            graph["experimental_data"] = default_dict_values(graph, "experimental_data", False)
            if 'phenotype' in graph and graph['phenotype'] == '*':
                if "species" not in graph:
                    graph['species'] = self.species_list
                graph['phenotype'] = set([col.split('_')[1] for col in self.fluxes_tup.columns
                                          if col.split('_')[0] in graph["species"]])
            if 'species' in graph and graph['species'] == '*':   # TODO - a species-resolved option must be developed for the paper figure
                graph['species'] = self.species_list
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
                                # if any([species == species_name for species_name in self.species_list]):
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
                                    graph['species'] == self.species_list
                            ):
                                ys['OD'].append(vals)
                                if "OD" not in graph['content']:
                                    ys[species].append(vals)
                        if all([graph['experimental_data'], '__bio' in basename, ]):
                            # any([content in basename])]):  # TODO - any() must include all_biomass and total
                            signal = "_".join([x for x in basename.split('_')[:-1] if x])
                            label = basename
                            if publishing:
                                if signal_species(signal) == 'ecoli':
                                    species = 'E. coli'
                                elif signal_species(signal) == 'pf':
                                    species = 'P. fluorescens'
                                label = f'Experimental {species} (from {signal})'
                                if 'OD' in signal:
                                    label = 'Experimental total biomass (from OD)'
                            ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split, scatter=True)
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
                            # TODO possibly express the modeled conversions of experimental data discretely, reflecting the original data
                            style = "solid" if (len(graph["species"]) < 1 or name not in linestyle) else linestyle[name]
                            style = "dashdot" if "model" in label else style
                            style = "solid" if ("OD" in name and not graph["experimental_data"]
                                                or "total" in graph["content"]) else style
                            ax.plot(xs.astype(np.float32), sum(vals), label=label, linestyle=style)

                    phenotype_id = "" if "phenotype" not in graph else graph['phenotype']
                    if "phenotype" in graph and not isinstance(graph['phenotype'], str):
                        phenotype_id = f"{','.join(graph['phenotype'])} phenotypes"

                    if "mets" not in graph and content != "c_":
                        species_id = graph["species"] if isinstance(graph["species"], str) else ",".join(graph["species"])
                        if "species" in graph and graph['species'] == self.species_list:
                            species_id = 'all species'
                        else:
                            phenotype_id = f"{','.join(graph['species'])} species"
                        if species_id == "all species" and not phenotype_id:
                            phenotype_id = ','.join(graph['species'])

                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    if "mets" in graph:
                        ax.set_ylim(mM_threshold)
                    ax.grid(axis="y")
                    if len(labels) > 1:
                        ax.legend()
                    else:
                        yscale = "linear"
                    ax.set_yscale(yscale)
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
                    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
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


class BIOLOGPhitting(CommPhitting):
    def __init__(self, fluxes_df, carbon_conc, media_conc, biolog_df, experimental_metadata, msdb_path):
        self.fluxes_df = fluxes_df; self.biolog_df = biolog_df; self.experimental_metadata = experimental_metadata
        self.carbon_conc = carbon_conc; self.media_conc = media_conc
        # import os
        from modelseedpy.biochem import from_local
        self.msdb = from_local(msdb_path)

    def fitAll(self, parameters: dict = None, rel_final_conc: float = None,
            abs_final_conc: dict = None, graphs: list = None, data_timesteps: dict = None,
            export_zip_name: str = None, export_parameters: bool = True, figures_zip_name: str = None, publishing: bool = False):
        # simulate each condition
        org_rel_final_conc = rel_final_conc
        for index, experiment in self.experimental_metadata.iterrows():
            if not any([re.search(experiment["ModelSEED_ID"], met.id) for met in model.metabolites]):
                continue
            print(index)
            display(experiment)
            # TODO - define the fluxes_df and phenotype(s) for each condition in this loop
            ## define the parameters for each experiment
            mets_to_track = zero_start = [experiment["ModelSEED_ID"]] if experiment["ModelSEED_ID"] else None
            rel_final_conc = {experiment["ModelSEED_ID"]: org_rel_final_conc} if experiment["ModelSEED_ID"] else None
            export_path = os.path.join(os.getcwd(), f"BIOLOG_LPs", f"{index}_{mets_to_track}.lp")
            ## define the CommPhitting object and simulate the experiment
            CommPhitting.__init__(self, self.fluxes_df, self.carbon_conc, self.media_conc,
                                  self.biolog_df.loc[index,:], self.experimental_metadata)
            CommPhitting.define_problem(self, parameters, mets_to_track, rel_final_conc, zero_start,
                                        abs_final_conc, data_timesteps, export_zip_name, export_parameters, export_path)
            try:
                CommPhitting.compute(self, graphs, None, export_zip_name, figures_zip_name, publishing)
            except (NoFluxError) as e:
                print(e)
            print("\n\n\n")
