# -*- coding: utf-8 -*-
# from modelseedpy.fbapkg.mspackagemanager import MSPackageManager
from modelseedpy.core.exceptions import FeasibilityError, ParameterError, ObjectAlreadyDefinedError, NoFluxError
from modelseedpy.core.optlanghelper import OptlangHelper, Bounds, tupVariable, tupConstraint, tupObjective, isIterable, define_term
from data.standardized_data.datastandardization import GrowthData
from modelseedpy.biochem import from_local
from pandas import DataFrame
from optlang import Model, Objective
from modelseedpy.core.fbahelper import FBAHelper
from scipy.constants import hour, minute
from scipy.optimize import newton
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from matplotlib import pyplot
from typing import Union, Iterable
from pprint import pprint
from time import sleep, process_time
from math import inf, isclose
from deepdiff import DeepDiff
from icecream import ic
import numpy as np
import cobra.io
# from cplex import Cplex
import warnings, logging, json, os, re

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
        pprint(names)
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
    return (conc*vmax)/(km+conc)

# parse primal values for use in the optimization loops
def parse_primals(primal_values, entity_labels, coefs=None, kcat_vals=None):
    if entity_labels == "kcat":
        kcat_primal = {}
        for trial, content in primal_values.items():
            for primal, value in content.items():
                if "bin" not in primal:
                    continue
                name, trial = primal.split("-")
                number = re.search(r"(\d)", name).group()
                species, pheno = re.sub(r"(bin\d_)", "", name).split("_")
                if "stationary" in pheno:
                    continue
                if species not in kcat_primal:
                    kcat_primal[species] = {}
                if pheno not in kcat_primal[species]:
                    kcat_primal[species][pheno] = 0
                if value == 0:
                    kcat_primal[species][pheno] += coefs[int(number)-1]*kcat_vals[species][pheno]

        return kcat_primal
    distinguished_primals = {}
    for trial, entities in primal_values.items():
        distinguished_primals[trial] = {}
        for entity, times in entities.items():
            if any([label in entity for label in entity_labels]):
                distinguished_primals[trial][entity] = {time:value for time, value in times.items()}
    return distinguished_primals

def signal_species(signal):
    return signal.split(":")[0].replace(" ", "_")

def _partition_coefs(initial_val, divisor):
    return (initial_val, initial_val/divisor, initial_val/divisor**2, initial_val/divisor**3, initial_val/divisor**4)

class CommPhitting:

    def __init__(self, fluxes_df, carbon_conc, media_conc, msdb_path, growth_df=None, experimental_metadata=None):
        self.parameters, self.variables, self.constraints, = {}, {}, {}
        self.zipped_output, self.plots = [], []
        self.fluxes_tup = FBAHelper.parse_df(fluxes_df)
        self.growth_df = growth_df; self.experimental_metadata = experimental_metadata
        self.carbon_conc = carbon_conc; self.media_conc = media_conc
        self.names = []
        self.msdb = from_local(msdb_path) ; self.msdb_path = msdb_path

    #################### FITTING PHASE METHODS ####################

    def fit_kcat(self, parameters: dict = None, mets_to_track: list = None, rel_final_conc: dict = None, zero_start: list = None,
                 abs_final_conc: dict = None, graphs: list = None, data_timesteps: dict = None,
                 export_zip_name: str = None, export_parameters: bool = True, requisite_biomass: dict = None,
                 export_lp: str = f'solveKcat.lp', publishing=True, primals_export_path=None):
        if export_zip_name and os.path.exists(export_zip_name):
            os.remove(export_zip_name)
        biomass_partition_coefs = [_partition_coefs(100, 10), _partition_coefs(4, 2), _partition_coefs(0.5, 2)]
        kcat_primal = None
        for index, coefs in enumerate(biomass_partition_coefs):
            # solve for growth rate constants with the previously solved biomasses
            new_simulation = CommPhitting(self.fluxes_tup, self.carbon_conc, self.media_conc,
                                          self.msdb_path, self.growth_df, self.experimental_metadata)
            new_simulation.define_problem(
                parameters, mets_to_track, rel_final_conc, zero_start, abs_final_conc,
                data_timesteps, export_zip_name, export_parameters, export_lp, kcat_primal, coefs, requisite_biomass)
            time1 = process_time()
            primals_export_path = primals_export_path or re.sub(r"(.lp)", ".json", export_lp)
            new_simulation.compute(graphs, export_zip_name, None, publishing, primals_export_path)
            kcat_primal = parse_primals(new_simulation.values, "kcat", coefs, new_simulation.parameters["kcat"])
            time2 = process_time()
            print(f"Done simulating with the coefficients for biomass partitions: {index}"
                  f"\n{(time2-time1)/60} minutes")
            pprint(kcat_primal)
        return {k: val for k, val in new_simulation.values.items() if "kcat" in k}

    def fit(self, parameters:dict=None, mets_to_track: list = None, rel_final_conc:dict=None, zero_start:list=None,
            abs_final_conc:dict=None, graphs: list = None, data_timesteps: dict = None, msdb_path:str=None,
            export_zip_name: str = None, export_parameters: bool = True, export_lp: str = 'CommPhitting.lp', figures_zip_name:str=None,
            publishing:bool=False):
        self.define_problem(parameters, mets_to_track, rel_final_conc, zero_start, abs_final_conc, data_timesteps, export_zip_name,
                            export_parameters, export_lp)
        self.compute(graphs, export_zip_name, figures_zip_name, publishing)

    def _update_problem(self, contents: Iterable):
        for content in contents:
            self.problem.add(content)
            self.problem.update()

    def define_b_vars(self, pheno, short_code, timestep, variables):
        self.variables['b_' + pheno][short_code][timestep] = tupVariable(
            _name("b_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
        self.variables['b1_' + pheno][short_code][timestep] = tupVariable(
            _name("b1_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
        self.variables['b2_' + pheno][short_code][timestep] = tupVariable(
            _name("b2_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
        self.variables['b3_' + pheno][short_code][timestep] = tupVariable(
            _name("b3_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
        self.variables['b4_' + pheno][short_code][timestep] = tupVariable(
            _name("b4_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
        self.variables['b5_' + pheno][short_code][timestep] = tupVariable(
            _name("b5_", pheno, short_code, timestep, self.names), Bounds(0, 1000))
        variables.extend([self.variables['b_' + pheno][short_code][timestep],
                          self.variables['b1_' + pheno][short_code][timestep],
                          self.variables['b2_' + pheno][short_code][timestep],
                          self.variables['b3_' + pheno][short_code][timestep],
                          self.variables['b4_' + pheno][short_code][timestep],
                          self.variables['b5_' + pheno][short_code][timestep]])
        if short_code not in self.variables["bin1_" + pheno]:
            self.variables['bin1_' + pheno][short_code] = tupVariable(
                _name("bin1_", pheno, short_code, "", self.names), Bounds(0, 1), "binary")
            self.variables['bin2_' + pheno][short_code] = tupVariable(
                _name("bin2_", pheno, short_code, "", self.names), Bounds(0, 1), "binary")
            self.variables['bin3_' + pheno][short_code] = tupVariable(
                _name("bin3_", pheno, short_code, "", self.names), Bounds(0, 1), "binary")
            self.variables['bin4_' + pheno][short_code] = tupVariable(
                _name("bin4_", pheno, short_code, "", self.names), Bounds(0, 1), "binary")
            self.variables['bin5_' + pheno][short_code] = tupVariable(
                _name("bin5_", pheno, short_code, "", self.names), Bounds(0, 1), "binary")
            variables.extend([self.variables['bin1_' + pheno][short_code], self.variables['bin2_' + pheno][short_code],
                              self.variables['bin3_' + pheno][short_code], self.variables['bin4_' + pheno][short_code],
                              self.variables['bin5_' + pheno][short_code]])
        return variables

    def define_b_cons(self, pheno, short_code, timestep, constraints, biomass_coefs):
        # define the partitioned biomass groups
        self.constraints['b1c_' + pheno][short_code][timestep] = tupConstraint(
            _name("b1c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    {"elements": [biomass_coefs[0], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b1_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b2c_' + pheno][short_code][timestep] = tupConstraint(
            _name("b2c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    {"elements": [biomass_coefs[1], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b2_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b3c_' + pheno][short_code][timestep] = tupConstraint(
            _name("b3c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    {"elements": [biomass_coefs[2], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b3_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b4c_' + pheno][short_code][timestep] = tupConstraint(
            _name("b4c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    {"elements": [biomass_coefs[3], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b4_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b5c_' + pheno][short_code][timestep] = tupConstraint(
            _name("b5c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    {"elements": [biomass_coefs[4], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b5_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })

        ## define the comprehensive biomass constraints
        self.constraints['b1c_control_' + pheno][short_code][timestep] = tupConstraint(
            _name("b1c_control_", pheno, short_code, timestep, self.names), Bounds(None, 0), {
                "elements": [
                    {"elements": [biomass_coefs[0], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b1_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin1_' + pheno][short_code].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b2c_control_' + pheno][short_code][timestep] = tupConstraint(
            _name("b2c_control_", pheno, short_code, timestep, self.names), Bounds(None, 0), {
                "elements": [
                    {"elements": [biomass_coefs[1], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b2_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin2_' + pheno][short_code].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b3c_control_' + pheno][short_code][timestep] = tupConstraint(
            _name("b3c_control_", pheno, short_code, timestep, self.names), Bounds(None, 0), {
                "elements": [
                    {"elements": [biomass_coefs[2], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b3_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin3_' + pheno][short_code].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b4c_control_' + pheno][short_code][timestep] = tupConstraint(
            _name("b4c_control_", pheno, short_code, timestep, self.names), Bounds(None, 0), {
                "elements": [
                    {"elements": [biomass_coefs[3], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b4_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin4_' + pheno][short_code].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })
        self.constraints['b5c_control_' + pheno][short_code][timestep] = tupConstraint(
            _name("b5c_control_", pheno, short_code, timestep, self.names), Bounds(None, 0), {
                "elements": [
                    {"elements": [biomass_coefs[4], self.variables['b_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1, self.variables['b5_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin5_' + pheno][short_code].name],
                     "operation": "Mul"},
                ],
                "operation": "Add"
            })

        # define the binary constraints
        self.constraints['bin1c_' + pheno][short_code][timestep] = tupConstraint(
            _name("bin1c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    1000,
                    {"elements": [-1, self.variables['b1_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin1_' + pheno][short_code].name],
                     "operation": "Mul"}
                ],
                "operation": "Add"
            })
        self.constraints['bin2c_' + pheno][short_code][timestep] = tupConstraint(
            _name("bin2c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    1000,
                    {"elements": [-1, self.variables['b2_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin2_' + pheno][short_code].name],
                     "operation": "Mul"}
                ],
                "operation": "Add"
            })
        self.constraints['bin3c_' + pheno][short_code][timestep] = tupConstraint(
            _name("bin3c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    1000,
                    {"elements": [-1, self.variables['b3_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin3_' + pheno][short_code].name],
                     "operation": "Mul"}
                ],
                "operation": "Add"
            })
        self.constraints['bin4c_' + pheno][short_code][timestep] = tupConstraint(
            _name("bin4c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    1000,
                    {"elements": [-1, self.variables['b4_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin4_' + pheno][short_code].name],
                     "operation": "Mul"}
                ],
                "operation": "Add"
            })
        self.constraints['bin5c_' + pheno][short_code][timestep] = tupConstraint(
            _name("bin5c_", pheno, short_code, timestep, self.names), Bounds(0, None), {
                "elements": [
                    1000,
                    {"elements": [-1, self.variables['b5_' + pheno][short_code][timestep].name],
                     "operation": "Mul"},
                    {"elements": [-1000, self.variables['bin5_' + pheno][short_code].name],
                     "operation": "Mul"}
                ],
                "operation": "Add"
            })

        # load the constraints to the model
        constraints.extend([self.constraints['b1c_' + pheno][short_code][timestep],
                            self.constraints['b2c_' + pheno][short_code][timestep],
                            self.constraints['b3c_' + pheno][short_code][timestep],
                            self.constraints['b4c_' + pheno][short_code][timestep],
                            self.constraints['b5c_' + pheno][short_code][timestep],
                            self.constraints['b1c_control_' + pheno][short_code][timestep],
                            self.constraints['b2c_control_' + pheno][short_code][timestep],
                            self.constraints['b3c_control_' + pheno][short_code][timestep],
                            self.constraints['b4c_control_' + pheno][short_code][timestep],
                            self.constraints['b5c_control_' + pheno][short_code][timestep],
                            self.constraints['bin1c_' + pheno][short_code][timestep],
                            self.constraints['bin2c_' + pheno][short_code][timestep],
                            self.constraints['bin3c_' + pheno][short_code][timestep],
                            self.constraints['bin4c_' + pheno][short_code][timestep],
                            self.constraints['bin5c_' + pheno][short_code][timestep]])
        return constraints

    def initialize_vars_cons(self, pheno, short_code):
        # cvt and cvf
        self.variables['cvt_' + pheno] = {}; self.variables['cvf_' + pheno] = {}
        self.variables['cvt_' + pheno][short_code] = {}; self.variables['cvf_' + pheno][short_code] = {}
        # total biomass and growth
        self.variables['b_' + pheno] = {}; self.variables['g_' + pheno] = {}
        self.variables['b_' + pheno][short_code] = {}; self.variables['g_' + pheno][short_code] = {}
        self.constraints['gc_' + pheno] = {}; self.constraints['cvc_' + pheno] = {}
        self.constraints['gc_' + pheno][short_code] = {}; self.constraints['cvc_' + pheno][short_code] = {}
        # partitioned biomasses
        self.variables['b1_' + pheno] = {}; self.variables['b2_' + pheno] = {};  self.variables['b3_' + pheno] = {}
        self.variables['b4_' + pheno] = {}; self.variables['b5_' + pheno] = {}
        self.variables['b1_' + pheno][short_code] = {}; self.variables['b2_' + pheno][short_code] = {}
        self.variables['b3_' + pheno][short_code] = {}; self.variables['b4_' + pheno][short_code] = {}
        self.variables['b5_' + pheno][short_code] = {}
        ## biomass binary variables
        self.variables['bin1_' + pheno] = {}; self.variables['bin2_' + pheno] = {};  self.variables['bin3_' + pheno] = {}
        self.variables['bin4_' + pheno] = {}; self.variables['bin5_' + pheno] = {}
        ## biomass partition constraints
        self.constraints['b1c_' + pheno] = {}; self.constraints['b2c_' + pheno] = {}; self.constraints['b3c_' + pheno] = {}
        self.constraints['b4c_' + pheno] = {}; self.constraints['b5c_' + pheno] = {}
        self.constraints['b1c_' + pheno][short_code] = {}; self.constraints['b2c_' + pheno][short_code] = {}
        self.constraints['b3c_' + pheno][short_code] = {}; self.constraints['b4c_' + pheno][short_code] = {}
        self.constraints['b5c_' + pheno][short_code] = {}
        self.constraints['b1c_control_' + pheno] = {}; self.constraints['b2c_control_' + pheno] = {}
        self.constraints['b3c_control_' + pheno] = {}; self.constraints['b4c_control_' + pheno] = {}
        self.constraints['b5c_control_' + pheno] = {}
        self.constraints['b1c_control_' + pheno][short_code] = {}; self.constraints['b2c_control_' + pheno][short_code] = {}
        self.constraints['b3c_control_' + pheno][short_code] = {}; self.constraints['b4c_control_' + pheno][short_code] = {}
        self.constraints['b5c_control_' + pheno][short_code] = {}
        self.constraints['binc_' + pheno] = {}; self.constraints['bin1c_' + pheno] = {}; self.constraints['bin2c_' + pheno] = {}
        self.constraints['bin3c_' + pheno] = {}; self.constraints['bin4c_' + pheno] = {}; self.constraints['bin5c_' + pheno] = {}
        self.constraints['bin1c_' + pheno][short_code] = {}; self.constraints['bin2c_' + pheno][short_code] = {}
        self.constraints['bin3c_' + pheno][short_code] = {}; self.constraints['bin4c_' + pheno][short_code] = {}
        self.constraints['bin5c_' + pheno][short_code] = {}

    def define_problem(self, parameters=None, mets_to_track = None, rel_final_conc=None, zero_start=None, abs_final_conc=None,
                       data_timesteps=None, export_zip_name: str=None, export_parameters: bool=True, export_lp: str='CommPhitting.lp',
                       primal_values=None, biomass_coefs=None, requisite_biomass:dict=None):
        # parse the growth data
        growth_tup = FBAHelper.parse_df(self.growth_df)
        self.species_list = [signal_species(signal) for signal in growth_tup.columns if ":" in signal]
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
            "cvct": 0.01, "cvcf": 0.01,  # Minimization coefficients of the phenotype conversion to and from the stationary phase.
            "bcv": 0.1,  # The highest fraction of species biomass that can change phenotypes in a timestep
            "cvmin": 0,  # The lowest value the limit on phenotype conversion goes,
            "kcat": 0.33,  # The kinetics constant that is externally adjusted
            'diffpos': 1, 'diffneg': 1,  # coefficients that weight difference between experimental and predicted biomass
            "stationary": 0.075,  # the penalty coefficient for the stationary phenotype
        })
        self.parameters.update(parameters)
        # define the appropriate kcat values
        self.parameters.update(self._universalize(self.parameters,"kcat", exclude=["stationary"]))
        if primal_values:
            for species, content in self.parameters["kcat"].items():
                if species not in primal_values:
                    continue
                for pheno in content:
                    if pheno not in primal_values[species]:
                        continue
                    self.parameters["kcat"][species][pheno] = primal_values[species][pheno]
        default_carbon_sources = ["cpd00076", "cpd00179", "cpd00027"]  # sucrose, maltose, glucose
        self.rel_final_conc = rel_final_conc or {c:1 for c in default_carbon_sources}
        self.abs_final_conc = abs_final_conc or {}
        if mets_to_track:
            self.mets_to_track = mets_to_track
        elif not isinstance(rel_final_conc, dict):
            self.mets_to_track = self.fluxes_tup.index
        else:
            self.mets_to_track = list(self.rel_final_conc.keys())
        zero_start = zero_start or []

        timesteps_to_delete = {}  # {short_code: full_times for short_code in unique_short_codes}
        if data_timesteps:  # {short_code:[times]}
            for short_code, times in data_timesteps.items():
                timesteps_to_delete[short_code] = set(list(range(len(full_times)))) - set(times)
                self.times[short_code] = np.delete(self.times[short_code], list(timesteps_to_delete[short_code]))

        # construct the problem
        objective = tupObjective("minimize variance and phenotypic transitions", [], "min")
        constraints, variables, simulated_mets = [], [], []
        time_1 = process_time()
        for exID in self.fluxes_tup.index:
            if exID == "bio":
                continue
            met_id = re.search(r"(cpd\d{5})", exID).group()
            met = self.msdb.compounds.get_by_id(met_id)
            if "C" not in met.elements:
                continue
            concID = f"c_{met_id}_e0"
            simulated_mets.append(met_id)
            self.variables[concID] = {}; self.constraints['dcc_' + met_id] = {}

            # define the growth rate for each metabolite and concentrations
            if "Vmax" and "Km" in self.parameters:
                self.parameters["Vmax"].update(self._universalize(self.parameters["Vmax"], met_id))
                self.parameters["Km"].update(self._universalize(self.parameters["Km"], met_id))
            for short_code in unique_short_codes:
                self.variables[concID][short_code] = {}; self.constraints['dcc_' + met_id][short_code] = {}
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for timestep in timesteps:
                    ## define the concentration variables
                    conc_var = tupVariable(_name(concID, "", short_code, timestep, self.names))
                    ## constrain initial time concentrations to the media or a large default
                    if timestep == timesteps[0]:
                        initial_val = None
                        if met_id in self.media_conc:
                            initial_val = self.media_conc[met_id]
                        if met_id in zero_start:
                            initial_val = 0
                        if dict_keys_exists(self.carbon_conc, met_id, short_code):
                            initial_val = self.carbon_conc[met_id][short_code]
                        if initial_val is not None:
                            conc_var = conc_var._replace(bounds=Bounds(initial_val, initial_val))
                    ## mandate complete carbon consumption
                    elif timestep == timesteps[-1] and (met_id in self.rel_final_conc or met_id in self.abs_final_conc):
                        if met_id in self.rel_final_conc:
                            final_bound = self.variables[concID][short_code][1].bounds.lb * self.rel_final_conc[met_id]
                        if met_id in self.abs_final_conc:  # this intentionally overwrites rel_final_conc
                            final_bound = self.abs_final_conc[met_id]
                        conc_var = conc_var._replace(bounds=Bounds(0, final_bound))
                        if met_id in zero_start:
                            conc_var = conc_var._replace(bounds=Bounds(final_bound, final_bound))
                    self.variables[concID][short_code][timestep] = conc_var
                    variables.append(self.variables[concID][short_code][timestep])
                    # print(variables[-1])
        for signal in growth_tup.columns:
            if ":" in signal:
                for pheno in self.fluxes_tup.columns:
                    if signal_species(signal) in pheno:
                        self.constraints['dbc_' + pheno] = {
                            short_code: {} for short_code in unique_short_codes}

        # define growth and biomass variables and constraints
        # self.parameters["kcat"] = {met_id:{species:_michaelis_menten()}}
        for pheno in self.fluxes_tup.columns:
            for short_code in unique_short_codes:
                self.initialize_vars_cons(pheno, short_code)
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                ninetieth_percentile_timestep = timesteps[int(0.9*len(timesteps))]
                penalty_range = np.linspace(self.parameters['stationary'], 0, len(timesteps[ninetieth_percentile_timestep:]))
                timestep_excess_count = 0
                for timestep in timesteps:
                    timestep = int(timestep)
                    variables = self.define_b_vars(pheno, short_code, timestep, variables)
                    if short_code not in self.constraints["binc_" + pheno]:
                        self.constraints['binc_' + pheno][short_code] = tupConstraint(
                            _name("binc_", pheno, short_code, "", self.names), Bounds(0, 4), {
                                "elements": [self.variables['bin1_' + pheno][short_code].name,
                                             self.variables['bin2_' + pheno][short_code].name,
                                             self.variables['bin3_' + pheno][short_code].name,
                                             self.variables['bin4_' + pheno][short_code].name,
                                             self.variables['bin5_' + pheno][short_code].name],
                                "operation": "Add"
                            })
                        constraints.append(self.constraints['binc_' + pheno][short_code])
                    constraints = self.define_b_cons(pheno, short_code, timestep, constraints, biomass_coefs)

                    ## define the growth rate variable or primal value
                    species, phenotype = pheno.split("_")
                    self.variables['g_' + pheno][short_code][timestep] = tupVariable(
                        _name("g_", pheno, short_code, timestep, self.names))
                    variables.append(self.variables['g_' + pheno][short_code][timestep])

                    if 'stationary' in pheno:
                        weight = self.parameters['stationary']
                        if timestep > ninetieth_percentile_timestep:
                            weight = penalty_range[timestep_excess_count]
                            timestep_excess_count += 1
                        objective.expr.extend([{
                            "elements": [
                                {"elements": [weight, self.variables['b_' + pheno][short_code][timestep].name],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        }])
                        continue
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
                    biomass_term = [self.parameters['cvmin'],
                                    {"elements": [self.parameters['bcv'],
                                                  self.variables["b_"+pheno][short_code][timestep].name],
                                     "operation": "Mul"}]
                    self.constraints['cvc_' + pheno][short_code][timestep].expr["elements"].extend(biomass_term)

                    # g_{pheno} = b_{pheno}*v_{pheno}
                    b_values = [self.variables['b1_' + pheno][short_code][timestep].name,
                                self.variables['b2_' + pheno][short_code][timestep].name,
                                self.variables['b3_' + pheno][short_code][timestep].name,
                                self.variables['b4_' + pheno][short_code][timestep].name,
                                self.variables['b5_' + pheno][short_code][timestep].name]
                    b_terms = [{"elements": [-self.parameters["kcat"][species][phenotype], b], "operation": "Mul"} for b in b_values]
                    self.constraints['gc_' + pheno][short_code][timestep] = tupConstraint(
                        name=_name('gc_', pheno, short_code, timestep, self.names),
                        expr={
                            "elements": [*b_terms, self.variables['g_' + pheno][short_code][timestep].name],
                            "operation": "Add"
                        })

                    constraints.extend([self.constraints['cvc_' + pheno][short_code][timestep],
                                        self.constraints['gc_' + pheno][short_code][timestep],])
                                        # self.constraints["binTot_" + pheno][short_code]])

        # define the concentration constraint
        half_dt = self.parameters['data_timestep_hr'] / 2
        time_2 = process_time()
        print(f'Done with concentrations and biomass loops: {(time_2 - time_1) / 60} min')
        for r_index, met in enumerate(self.fluxes_tup.index):
            met_id = _met_id_parser(met)
            if met_id not in simulated_mets:
                continue
            concID = f"c_{met_id}_e0"
            for short_code in unique_short_codes:
                timesteps = list(range(1, len(self.times[short_code]) + 1))
                for timestep in timesteps[:-1]:
                    # c_{met} + dt/2*sum_k^K(n_{k,met} * (g_{pheno}+g+1_{pheno})) = c+1_{met}
                    next_timestep = timestep + 1
                    growth_phenos = [[self.variables['g_' + pheno][short_code][next_timestep].name,
                                      self.variables['g_' + pheno][short_code][timestep].name] for pheno in self.fluxes_tup.columns]
                    self.constraints['dcc_' + met_id][short_code][timestep] = tupConstraint(
                        name=_name("dcc_", met_id, short_code, timestep, self.names),
                        expr={
                            "elements": [
                                self.variables[concID][short_code][timestep].name,
                                {"elements": [-1, self.variables[concID][short_code][next_timestep].name],
                                 "operation": "Mul"},
                                *OptlangHelper.dot_product(growth_phenos,
                                                           heuns_coefs=half_dt * self.fluxes_tup.values[r_index])],
                            "operation": "Add"
                        })
                    constraints.append(self.constraints['dcc_' + met_id][short_code][timestep])

        #   define the conversion variables of every signal for every phenotype
        # for signal in growth_tup.columns[2:]:
        #     for pheno in self.fluxes_tup.columns:
        #         conversion_name = "_".join([signal, pheno, "__conversion"])
        #         self.variables[conversion_name] = tupVariable(conversion_name)
        #         variables.append(self.variables[conversion_name])

        time_3 = process_time()
        print(f'Done with DCC loop: {(time_3 - time_2) / 60} min')
        species_phenos = {}
        self.conversion_bounds = [5e-6, 50]
        for index, org_signal in enumerate(growth_tup.columns[2:]):
            # signal = org_signal.split(":")[1]
            signal = org_signal.replace(":", "|")
            species = signal_species(org_signal)
            species_phenos[species] = {None if "OD" in species else f"{species}_stationary"}
            signal_column_index = index + 2
            data_timestep = 1
            # TODO - The conversion must be defined per phenotype
            self.variables[signal + '|conversion'] = tupVariable(
                signal + '|conversion', bounds=Bounds(*self.conversion_bounds))
            variables.append(self.variables[signal + '|conversion'])

            self.variables[signal + '|bio'] = {}; self.variables[signal + '|diffpos'] = {}
            self.variables[signal + '|diffneg'] = {}; self.variables['g_' + species] = {}
            self.constraints[signal + '|bioc'] = {}; self.constraints[signal + '|diffc'] = {}
            self.constraints["gc_" + species] = {}; self.constraints["totVc_" + species] = {}
            self.constraints["totGc_" + species] = {}; self.constraints[signal + '|bio_finalc'] = {}
            for short_code in unique_short_codes:
                self.variables[signal + '|bio'][short_code] = {}
                self.variables[signal + '|diffpos'][short_code] = {}
                self.variables[signal + '|diffneg'][short_code] = {}
                self.variables['g_' + species][short_code] = {}
                self.constraints[signal + '|bioc'][short_code] = {}
                self.constraints[signal + '|diffc'][short_code] = {}
                self.constraints["gc_" + species][short_code] = {}
                self.constraints["totVc_" + species][short_code] = {}
                self.constraints["totGc_" + species][short_code] = {}
                # self.constraints[signal + '|bio_finalc'][short_code] = {}
                # the value entries are matched to only the timesteps that are condoned by data_timesteps
                values_slice = trial_contents(short_code, growth_tup.index, growth_tup.values)
                if timesteps_to_delete:
                    values_slice = np.delete(values_slice, list(timesteps_to_delete[short_code]), axis=0)
                timesteps = list(range(1, len(values_slice) + 1))
                for timestep in timesteps[:-1]:
                    ## the user timestep and data timestep must be synchronized
                    if int(timestep) * self.parameters['timestep_hr'] < data_timestep * self.parameters['data_timestep_hr']:
                        continue
                    data_timestep += 1
                    if data_timestep > int(self.times[short_code][-1] / self.parameters["data_timestep_hr"]):
                        print(f"The time from the user-defined exceeds the simulation time, so the DBC & diff loop os terminating.")
                        break
                    next_timestep = int(timestep) + 1
                    ## the phenotype transition terms are aggregated
                    total_biomass, signal_sum, from_sum, to_sum = [], [], [], []
                    for pheno_index, pheno in enumerate(self.fluxes_tup.columns):
                        ### define the collections of signal and pheno terms
                        if species in pheno or "OD" in signal:
                            # if not isnumber(b_values[pheno][short_code][timestep]):
                            signal_sum.append({"operation": "Mul", "elements": [
                                -1, self.variables['b_' + pheno][short_code][timestep].name]})
                            # else:
                            #     signal_sum.append(-b_values[pheno][short_code][timestep])
                            ### total_biomass.append(self.variables["b_"+pheno][short_code][timestep].name)
                            if all(['OD' not in signal, species in pheno, 'stationary' not in pheno]):
                                species_phenos[species].add(pheno)
                                from_sum.append({"operation": "Mul", "elements": [
                                    -1, self.variables["cvf_" + pheno][short_code][timestep].name]})
                                to_sum.append(self.variables["cvt_" + pheno][short_code][timestep].name)
                    for pheno in species_phenos[species]:
                        if "OD" in signal:
                            continue
                        # print(pheno, timestep, b_values[pheno][short_code][timestep], b_values[pheno][short_code][next_timestep])
                        if "stationary" in pheno:
                            # b_{phenotype} - sum_k^K(es_k*cvf) + sum_k^K(pheno_bool*cvt) = b+1_{phenotype}
                            self.constraints['dbc_' + pheno][short_code][timestep] = tupConstraint(
                                name=_name("dbc_", pheno, short_code, timestep, self.names),
                                expr={"elements": [*from_sum, *to_sum], "operation": "Add"})
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
                            "elements": [-1, self.variables['b_' + pheno][short_code][next_timestep].name],
                            "operation": "Mul"}]
                        # else:
                        #     biomass_term = [b_values[pheno][short_code][timestep]-b_values[pheno][short_code][next_timestep]]
                        # print(biomass_term)
                        self.constraints['dbc_' + pheno][short_code][timestep].expr["elements"].extend(biomass_term)
                        constraints.append(self.constraints['dbc_' + pheno][short_code][timestep])

                    if timestep != timesteps[-2] or signal not in requisite_biomass[short_code]:
                        self.variables[signal + '|bio'][short_code][timestep] = tupVariable(
                            _name(signal, '|bio', short_code, timestep, self.names))
                    else:
                        biomass_flux = requisite_biomass[short_code][signal]["bio"]
                        estimated_biomass = biomass_flux * int(timestep)*self.parameters['data_timestep_hr']
                        self.variables[signal + '|bio'][short_code][timestep] = tupVariable(
                            _name(signal, '|bio', short_code, timestep, self.names),
                            Bounds(estimated_biomass * 0.7, estimated_biomass * 1.3))
                    self.variables[signal + '|diffpos'][short_code][timestep] = tupVariable(
                        _name(signal, '|diffpos', short_code, timestep, self.names), Bounds(0, 100))
                    self.variables[signal + '|diffneg'][short_code][timestep] = tupVariable(
                        _name(signal, '|diffneg', short_code, timestep, self.names), Bounds(0, 100))
                    variables.extend([self.variables[signal + '|bio'][short_code][timestep],
                                      self.variables[signal + '|diffpos'][short_code][timestep],
                                      self.variables[signal + '|diffneg'][short_code][timestep]])

                    # {signal}__conversion*datum = {signal}__bio
                    self.constraints[signal + '|bioc'][short_code][timestep] = tupConstraint(
                        name=_name(signal, '|bioc', short_code, timestep, self.names),
                        expr={
                            "elements": [
                                {"elements": [-1, self.variables[signal + '|bio'][short_code][timestep].name],
                                 "operation": "Mul"},
                                {"elements": [self.variables[signal + '|conversion'].name,
                                              values_slice[timestep, signal_column_index]],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })
                    constraints.append(self.constraints[signal + '|bioc'][short_code][timestep])

                    # {speces}_bio + {signal}_diffneg-{signal}_diffpos = sum_k^K(es_k*b_{phenotype})
                    self.constraints[signal + '|diffc'][short_code][timestep] = tupConstraint(
                        name=_name(signal, '|diffc', short_code, timestep, self.names),
                        expr={
                            "elements": [
                                self.variables[signal + '|bio'][short_code][timestep].name,
                                self.variables[signal + '|diffneg'][short_code][timestep].name,
                                {"elements": [-1, self.variables[signal + '|diffpos'][short_code][timestep].name],
                                 "operation": "Mul"}],
                            "operation": "Add"
                        })
                    if all([isinstance(val, dict) for val in signal_sum]):
                        self.constraints[signal + "|diffc"][short_code][timestep].expr["elements"].extend(signal_sum)
                    elif all([isnumber(val) for val in signal_sum]):
                        self.constraints[signal + "|diffc"][short_code][timestep].expr["elements"].append(sum(signal_sum))
                    else:
                        raise ValueError(f"The {signal_sum} value has unexpected contents.")
                    constraints.append(self.constraints[signal + '|diffc'][short_code][timestep])

                    objective.expr.extend([{
                        "elements": [
                            {"elements": [self.parameters['diffpos'],
                                          self.variables[signal + '|diffpos'][short_code][timestep].name],
                             "operation": "Mul"},
                            {"elements": [self.parameters['diffneg'],
                                          self.variables[signal + '|diffneg'][short_code][timestep].name],
                             "operation": "Mul"}],
                        "operation": "Add"
                    }])

        time_4 = process_time()
        print(f'Done with the DBC & diffc loop: {(time_4 - time_3) / 60} min')

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
            if re.search(r"(\\\\/)", export_lp):
                os.makedirs(os.path.dirname(export_lp), exist_ok=True)
            with open(export_lp, 'w') as lp:
                lp.write(self.problem.to_lp())
            _export_model_json(self.problem.to_json(), 'CommPhitting.json')
            self.zipped_output.extend([export_lp, 'CommPhitting.json'])
        if export_zip_name:
            self.zip_name = export_zip_name
            sleep(2)
            with ZipFile(self.zip_name, 'a', compression=ZIP_LZMA) as zp:
                for file in self.zipped_output:
                    zp.write(file)
                    os.remove(file)

        time_6 = process_time()
        print(f'Done exporting the content: {(time_6 - time_5) / 60} min')

    def compute(self, graphs: list = None, export_zip_name=None, figures_zip_name=None, publishing=False,
                primals_export_path:str = "primal_values.json", remove_empty_plots=False):
        print("starting optimization")
        self.values = {}
        solution = self.problem.optimize()
        # categorize the primal values by trial and time
        if "optimal" not in solution:
            raise FeasibilityError(f'The solution is sub-optimal, with a(n) {solution} status.')
        if all(np.array(list(self.problem.primal_values.values())) == 0):
            raise NoFluxError("The simulation lacks any flux.")
        for variable, value in self.problem.primal_values.items():
            if "v_" in variable:
                self.values[variable] = value
            elif 'conversion' in variable or re.search(r"(bin\d)", variable):
                self.values[short_code].update({variable: value})
                if value in self.conversion_bounds:
                    warnings.warn(f"The conversion factor {value} optimized to a bound, which may be "
                                  f"indicative of an error, such as improper kinetic rates.")
            else:
                # print(variable, value)
                basename, short_code, timestep = variable.split('-')
                time_hr = int(timestep) * self.parameters['data_timestep_hr']
                self.values[short_code] = self.values.get(short_code, {})
                self.values[short_code][basename] = self.values[short_code].get(basename, {})
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

        # visualize the specified information
        if graphs:
            self.graph(graphs, export_zip_name=figures_zip_name or export_zip_name, publishing=publishing,
                       remove_empty_plots=remove_empty_plots)

    def load_model(self, mscomfit_json_path: str = None, zip_name: str = None, model_to_load: dict = None):
        if zip_name:
            with ZipFile(zip_name, 'r') as zp:
                zp.extract(mscomfit_json_path)
        if mscomfit_json_path:
            with open(mscomfit_json_path, 'r') as mscmft:
                return json.load(mscmft)
        if model_to_load:
            self.problem = Model.from_json(model_to_load)

    @staticmethod
    def assign_values(param, var, next_dimension):
        dic = {var: {}}
        for dim1, dim2_list in next_dimension.items():
            dic[var][dim1] = {dim2: param for dim2 in dim2_list}
        return dic

    def _universalize(self, param, var, next_dimension=None, exclude=None):
        if not next_dimension:
            next_dimension = {}
            for organism in self.fluxes_tup.columns:
                species, pheno = organism.split("_")
                if pheno in exclude:
                    continue
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

    def adjust_color(self, color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        import colorsys
        try:
            import matplotlib.colors as mc
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def _add_plot(self, ax, labels, label, basename, trial, x_axis_split, linestyle="solid", scatter=False, color=None):
        labels.append(label or basename.split('-')[-1])
        if scatter:
            # print(label)
            # pprint(list(self.values[trial][basename].keys()))
            # pprint(list(self.values[trial][basename].values()))
            ax.scatter(list(self.values[trial][basename].keys()),
                       list(self.values[trial][basename].values()),
                       s=10, label=labels[-1], color=color or None)
        else:
            ax.plot(list(self.values[trial][basename].keys()),
                    list(self.values[trial][basename].values()),
                    label=labels[-1], linestyle=linestyle)
        x_ticks = np.around(np.array(list(self.values[trial][basename].keys())), 0)
        ax.set_xticks(x_ticks[::x_axis_split])
        return ax, labels

    def graph(self, graphs, primal_values_filename: str = None, primal_values_zip_path: str = None,
              export_zip_name: str = None, data_timestep_hr: float = 0.163, publishing: bool = False, title: str = None,
              remove_empty_plots:bool = False):
        print(export_zip_name)
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
        x_axis_split = int(2 / data_timestep_hr / timestep_ratio)
        self.plots = set()
        contents = {"biomass": 'b_', "all_biomass": 'b_', "growth": 'g_', "conc": "c_"}
        mM_threshold = 1e-3
        for graph_index, graph in enumerate(graphs):
            content = contents.get(graph['content'], graph['content'])
            y_label = 'Variable value'; x_label = r'Time ($hr$)'
            if any([x in graph['content'] for x in ['biomass', 'OD']]):
                ys = {name: [] for name in self.species_list}
                ys.update({"OD":[]})
                if "species" not in graph:
                    graph['species'] = self.species_list
            if "biomass" in graph['content']:
                y_label = r'Biomass ($\frac{g}{L}$)'
            elif 'growth' in graph['content']:
                y_label = r'Biomass growth ($\frac{g}{hr}$)'
            graph["experimental_data"] = graph.get("experimental_data", False)
            if "painting" not in graph:
                graph["painting"] = {
                    "OD": {
                        "color": "blue",
                        "linestyle": "solid",
                        "name": "Total biomass"
                    },
                    "ecoli": {
                        "color": "red",
                        "linestyle": "dashed",
                        "name": "E. coli"
                    },
                    "pf": {
                        "color": "green",
                        "linestyle": "dotted",
                        "name": "P. fluorescens"
                    }}
            graph["parsed"] = graph.get("parsed", False)
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
            print(f"graph_{graph_index}") ; pprint(graph)

            # define figure specifications
            if publishing:
                pyplot.rc('axes', titlesize=22, labelsize=28)
                pyplot.rc('xtick', labelsize=24)
                pyplot.rc('ytick', labelsize=24)
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
                                species_name = graph["painting"][species]["name"]
                                label = f'{species_name} total (model)'
                            labels.append({species: label})
                            xs = np.array(list(values.keys()))
                            vals = np.array(list(values.values()))
                            if remove_empty_plots and all(vals == 0):
                                continue
                            # print(basename, values.values())
                            ax.set_xticks(xs[::int(3 / data_timestep_hr / timestep_ratio)])
                            if (any([x in graph['content'] for x in ["total", "biomass", 'OD']]) or
                                    graph['species'] == self.species_list):
                                ys['OD'].append(vals)
                                if "OD" not in graph['content']:
                                    ys[species].append(vals)
                        if all([graph['experimental_data'], '|bio' in basename, ]):
                            # any([content in basename])]):  # TODO - any() must include all_biomass and total
                            species, signal, phenotype = basename.split('|')
                            # signal = "_".join([x for x in basename.split('_')[:-1] if x])
                            label = basename
                            if publishing:
                                species_name = graph["painting"][species]["name"]
                                label = f'Experimental {species_name} (from {signal})'
                                if 'OD' in signal:
                                    label = f'Experimental total (from {signal})'
                            # print(basename, label, self.values[trial][basename].values())
                            if remove_empty_plots and all(self.values[trial][basename].values() == 0):
                                continue
                            ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split, scatter=True,
                                                        color=self.adjust_color(graph["painting"][species]["color"], 1.5))
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
                                style = graph["painting"][specie]["linestyle"]
                            if graph['phenotype'] == '*':
                                if 'total' in graph["content"]:
                                    labels = [label]
                                    xs = np.array(list(values.keys()))
                                    ys.append(np.array(list(values.values())))
                                    if remove_empty_plots and all(ys == 0):
                                        continue
                                    ax.set_xticks(x_ticks[::int(3 / data_timestep_hr / timestep_ratio)])
                                else:
                                    if remove_empty_plots and all(self.values[trial][basename].values() == 0):
                                        continue
                                    ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split, style)
                                # print('species content of all phenotypes')
                            # graph all phenotypes
                            elif any([x in basename for x in graph['phenotype']]):
                                if remove_empty_plots and all(self.values[trial][basename].values() == 0):
                                    continue
                                ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split, style)
                                # print('all content over all phenotypes')
                            break
                    # graph media concentration plots
                    elif "mets" in graph and all([any([x in basename for x in graph["mets"]]), 'c_cpd' in basename]):
                        if not any(np.array(list(self.values[trial][basename].values())) > mM_threshold):
                            continue
                        if remove_empty_plots and all(self.values[trial][basename].values() == 0):
                            continue
                        label=self.msdb.compounds.get_by_id(re.search(r"(cpd\d+)", basename).group()).name
                        ax, labels = self._add_plot(ax, labels, label, basename, trial, x_axis_split)
                        yscale = "log"
                        y_label = r'Concentration ($mM$)'

                if labels:  # this assesses whether a graph was constructed
                    if any([x in graph['content'] for x in ['OD', 'biomass', 'total']]):
                        labeled_species = [label for label in labels if isinstance(label, dict)]
                        for name, vals in ys.items():
                            if not vals or (len(ys) == 2 and "OD" not in name):
                                continue
                            if len(ys) == 2:
                                specie_label = [graph["painting"][name]["name"] for name in ys if "OD" not in name][0]
                                label = f"{graph['painting'][name]['name']} ({specie_label})"
                            else:
                                label = f'{name}_biomass (model)'
                                if labeled_species:
                                    for label_specie in labeled_species:
                                        if name in label_specie:
                                            label = label_specie[name]
                                            break
                            style = "solid" if (len(graph["species"]) < 1 or name not in graph["painting"]
                                                ) else graph["painting"][name]["linestyle"]
                            style = "dashdot" if "model" in label else style
                            style = "solid" if ("OD" in name and not graph["experimental_data"]
                                                or "total" in graph["content"]) else style
                            if not graph["parsed"]:
                                ax.plot(xs.astype(np.float32), sum(vals), label=label, linestyle=style,
                                        color=self.adjust_color(graph["painting"][name].get("color", None), 1.5))
                            else:
                                # TODO - the phenotypes of each respective species must be added to these developed plots.
                                fig, ax = pyplot.subplots(dpi=200, figsize=(11, 7))
                                ax.set_xlabel(x_label) ; ax.set_ylabel(y_label) ; ax.grid(axis="y") ; ax.legend()
                                ax.plot(xs.astype(np.float32), sum(vals), label=label, linestyle=style,
                                        color=self.adjust_color(graph["painting"][name].get("color", None), 1.5))
                                phenotype_id = graph.get('phenotype', "")
                                if "phenotype" in graph and not isinstance(graph['phenotype'], str):
                                    phenotype_id = f"{','.join(graph['phenotype'])} phenotypes"
                                fig_name = f'{"_".join([trial, name, phenotype_id, content])}.jpg'
                                if "mets" in graph:
                                    fig_name = f"{trial}_{','.join(graph['mets'])}_c.jpg"
                                fig.savefig(fig_name, bbox_inches="tight", transparent=True)
                    if graph["parsed"]:
                        break

                    phenotype_id = graph.get('phenotype', "")
                    if "phenotype" in graph and not isinstance(graph['phenotype'], str):
                        phenotype_id = f"{','.join(graph['phenotype'])} phenotypes"

                    species_id = ""
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

                    self.plots.add(fig_name)

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
    def __init__(self, carbon_conc, media_conc, biolog_df, fluxes_df, experimental_metadata, msdb_path):
        self.biolog_df = biolog_df; self.experimental_metadata = experimental_metadata
        self.carbon_conc = carbon_conc; self.media_conc = media_conc; self.fluxes_df = fluxes_df
        # import os
        from modelseedpy.biochem import from_local
        self.msdb = from_local(msdb_path)

    def fitAll(self, models_list:list, parameters: dict = None, rel_final_conc: float = None,
               abs_final_conc: dict = None, graphs: list = None, data_timesteps: dict = None,
               export_zip_name: str = None, export_parameters: bool = True,
               figures_zip_name: str = None, publishing: bool = False):
        # simulate each condition
        org_rel_final_conc = rel_final_conc
        # total_reactions = set(list(chain.from_iterable([model.reactions for model in models_dict.values()])))
        community_members = {model: {"name": name} for name, model in models_list.items()}
        model_abbreviations = ','.join([model_abbrev for model_abbrev in models_list])
        for index, experiment in self.experimental_metadata.iterrows():
            print(f"\n{index}")
            display(experiment)
            if not experiment["ModelSEED_ID"]:
                print("The BIOLOG condition is not defined.")
                continue
            for model in models_list.values():
                cpd = self.msdb.compounds.get_by_id(experiment["ModelSEED_ID"])
                if ("C" not in cpd.elements or not any([
                        re.search(experiment["ModelSEED_ID"], rxn.id) for rxn in model.reactions])):
                    if "valid_condition" not in locals():
                        valid_condition = False
                    continue
                exp_list = [experiment["ModelSEED_ID"]] if isinstance(
                    experiment["ModelSEED_ID"], str) else experiment["ModelSEED_ID"]
                community_members[model].update({"phenotypes": {
                    re.sub(r"(-|\s)", "", experiment["condition"]): {"consumed": exp_list} }})
                valid_condition = True
            if not valid_condition:
                print(f"The BIOLOG condition with {experiment['ModelSEED_ID']} is not"
                      f" absorbed by the {model_abbreviations} model(s).")
                continue
            print(f"The {experiment['ModelSEED_ID']} ({cpd.formula}) metabolite of the "
                  f"{experiment['condition']} condition may feed the {model_abbreviations} model(s).")
            if not any([experiment["ModelSEED_ID"] in pheno for pheno in self.fluxes_df]):
                print(e)
                print(f"The {experiment['ModelSEED_ID']} ({cpd.formula}) metabolite of the "
                      f"{experiment['condition']} condition is not a suitable phenotype for "
                      f"the {model_abbreviations} model(s).")
                continue
            mets_to_track = exp_list
            rel_final_conc = {experiment["ModelSEED_ID"]: org_rel_final_conc}
            export_path = os.path.join(os.getcwd(), f"BIOLOG_LPs", f"{index}_{','.join(mets_to_track)}.lp")
            ## define the CommPhitting object and simulate the experiment
            CommPhitting.__init__(self, self.fluxes_df, self.carbon_conc, self.media_conc,
                                  self.biolog_df.loc[index,:], self.experimental_metadata)
            CommPhitting.define_problem(self, parameters, mets_to_track, rel_final_conc, None,
                                        abs_final_conc, data_timesteps, export_zip_name, export_parameters, export_path)
            new_graphs = []
            for graph in graphs:
                graph["trial"] = index
                new_graphs.append(graph)
            try:
                CommPhitting.compute(self, new_graphs, export_zip_name, figures_zip_name, publishing,
                                     f"BIOLOG_{experiment['ModelSEED_ID']}.json", remove_empty_plots=True)
                break
            except (NoFluxError) as e:
                print(e)
            print("\n\n\n")
