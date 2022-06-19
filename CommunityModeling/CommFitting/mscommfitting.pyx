# -*- coding: utf-8 -*-
# cython: language_level=3
from pandas import read_table, read_csv, DataFrame
from optlang import Variable, Constraint, Objective, Model
from modelseedpy.core.fbahelper import FBAHelper
from zipfile import ZipFile, ZIP_LZMA
from optlang.symbolics import Zero
from sympy.core.add import Add
from datetime import datetime
# from pprint import pprint
from time import sleep, process_time
import numpy as np
# from cplex import Cplex

import cython
import json, os

def _variable_name(name, suffix, col, index):
    return '-'.join([name+suffix, col, index])

def _constraint_name(name, suffix, col, index):
    return '_'.join([name+suffix, col, index])

@cython.cclass
class MSCommFitting():   # explicit typing for cython
    parameters: dict = {}; variables: dict = {}; constraints: dict = {}; dataframes: dict = {}; signal_species: dict = {}
    phenotypes_parsed_df: np.ndarray; problem: object; species_phenotypes_bool_df: object
        
    def __init__(self, 
                 phenotypes_csv_path: str = None,      # a custom CSV of media phenotypic data
                 permanent_media_id: str = None,       # the permanent KBase ID of the media (e.g. 93465/3/1) that describes the experimental conditions 
                 kbase_token: str = None,              # the KBase user token that must be provided to access permanent_media_id
                 signal_tsv_paths: dict = {},          # the dictionary of index names for each paths to signal TSV data that will be fitted
                 ):
        if phenotypes_csv_path:
            phenotypes_df = read_csv(phenotypes_csv_path)
            phenotypes_df.index = phenotypes_df['rxn']
            to_drop = [col for col in phenotypes_df.columns if ' ' in col]
            for col in to_drop+['rxn']:
                phenotypes_df.drop(col, axis=1, inplace=True)
            print(f'The {to_drop+["rxn"]} columns were dropped from the phenotypes CSV.')
            phenotypes_df.astype(str)
        elif permanent_media_id:
            import cobrakbase
            if kbase_token:
                kbase = cobrakbase.KBaseAPI(kbase_token)
            elif not kbase_token and not os.exists('~/.kbase/token'):
                raise ValueError(f'The kbase_token {kbase_token} must be provided or stored in ".kbase/token" to access KBase content.')
            
            media = kbase.get_from_ws(permanent_media_id)
            for info, content in media.data['mediacompounds'].items():
                pass  # !!! convert the media into a DataFrame of equivalent structure to the phenotypes CSV
        self.phenotypes_parsed_df = FBAHelper.parse_df(phenotypes_df)
        
        self.species_phenotypes_bool_df = DataFrame(columns=self.phenotypes_parsed_df[1])
        for path, name in signal_tsv_paths.items():
            signal = os.path.splitext(path)[0].split("_")[0]
            # define the signal dataframe
            self.signal_species[signal] = name
            self.dataframes[signal] = read_table(path).iloc[1::2]  # !!! is this the proper slice of data, or should the other set of values at each well/time be used?
            self.dataframes[signal].index = self.dataframes[signal]['Well']
            for col in ['Plate', 'Cycle', 'Well']:
                self.dataframes[signal].drop(col, axis=1, inplace=True)
            self.dataframes[signal].astype(str)
            # convert the dataframe to a numpy array for greater efficiency
            self.dataframes[signal]: np.ndarray = FBAHelper.parse_df(self.dataframes[signal])
        
            # differentiate the phenotypes for each species
            self.parameters[signal]: dict = {}
            self.variables[signal+'__coef'], self.variables[signal+'__bio'], self.variables[signal+'__diff'] = {}, {}, {}
            self.constraints[signal+'__bioc'], self.constraints[signal+'__diffc'] = {}, {}
            if "OD" not in signal:
                self.species_phenotypes_bool_df.loc[signal]: np.ndarray[cython.int] = np.array([
                    1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_parsed_df[1]])
                
    @cython.ccall # cfunc
    def define_problem(self, conversion_rates={'cvt': 0, 'cvf': 0}, constraints={'cvc': {}}): 
        self.problem = Model()
        self.parameters.update({
            "timestep_s": 600,      # Size of time step used in simulation in seconds
            "cvct": 1,              # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf": 1,              # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv": 1,               # This is the highest fraction of biomass for a given strain that can change phenotypes in a single time step
            "cvmin": 0.1,           # This is the lowest value the limit on phenotype conversion goes, 
            "y": 1,                 # Stoichiometry for interaction of strain k with metabolite i
            "v": 1,
        })
        self.parameters.update(conversion_rates)
        index: str; col: str; name: str; strain: str; met: str; growth_stoich: cython.float = 0; 
        obj_coef:dict = {}; constraints: np.ndarray = np.array([]); variables: np.ndarray = np.array([])

        # 3.3E4 loops => 67 minutes
        time_1 = process_time()
        for name, parsed_df in self.dataframes.items():  # 3 loops
            for strain in self.phenotypes_parsed_df[1]:  # 6
                self.variables['cvt_'+strain]:dict = {}; self.variables['cvf_'+strain]:dict = {}
                self.variables['b_'+strain]:dict = {}; self.variables['g_'+strain]:dict = {}
                self.variables['v_'+strain]:dict = {}; self.variables['b+1_'+strain]:dict = {}
                self.constraints['gc_'+strain]:dict = {}; self.constraints['cvc_'+strain]:dict = {}
                self.constraints['dbc_'+strain]:dict = {}
                for r_index, index in enumerate(parsed_df[0]):  # 66
                    self.variables['b_'+strain][index]:dict = {}; self.variables['g_'+strain][index]:dict = {}
                    self.variables['cvt_'+strain][index]:dict = {}; self.variables['cvf_'+strain][index]:dict = {}
                    self.variables['v_'+strain][index]:dict = {};  self.variables['b+1_'+strain][index]:dict = {}
                    self.constraints['gc_'+strain][index]:dict = {}; self.constraints['cvc_'+strain][index]:dict = {}
                    self.constraints['dbc_'+strain][index]:dict = {}
                    last_column = False
                    for col in parsed_df[1]:  # 27
                        next_col = str(int(col)+1)
                        if next_col == len(parsed_df[1]):
                            last_column = True   
                        self.variables['b_'+strain][index][col] = Variable(    # predicted biomass abundance
                            _variable_name("b_", strain, col, index), lb=0, ub=1000)  
                        self.variables['g_'+strain][index][col] = Variable(    # biomass growth
                            _variable_name("g_", strain, col, index), lb=0, ub=1000)   
                        # self.variables['v_'+strain][index][col] = Variable(    # predicted kinetic coefficient
                        #     _variable_name("v_", strain, index, col), lb=0, ub=1000)                          
                        self.variables['b+1_'+strain][index][next_col] = Variable(  # predicted biomass abundance
                            _variable_name("b+1_", strain, index, next_col), lb=0, ub=1000)      
                        self.variables['cvt_'+strain][index][col] = Variable(  # conversion rate to the stationary phase
                            _variable_name("cvt_", strain, index, col), lb=0, ub=100)   
                        self.variables['cvf_'+strain][index][col] = Variable(  # conversion from to the stationary phase
                            _variable_name("cvf_", strain, index, col), lb=0, ub=100)
                        
                        # g_{strain} - b_{strain}*v = 0
                        self.constraints['gc_'+strain][index][col] = Constraint(
                            self.variables['g_'+strain][index][col] 
                            - self.parameters['v']*self.variables['b_'+strain][index][col],  
                            _constraint_name("gc_", strain, index, col), lb=0, ub=0)
                        
                        # 0 <= -cvt + bcv*b_{strain} + cvmin
                        self.constraints['cvc_'+strain][index][col] = Constraint(
                            -self.variables['cvt_'+strain][index][col] 
                            + self.parameters['bcv']*self.variables['b_'+strain][index][col] 
                            + self.parameters['cvmin'],
                            _constraint_name('cvc_', strain, index, col), lb=0, ub=None)
                    
                        obj_coef.update({
                            self.variables['cvt_'+strain][index][col]: self.parameters['cvct'],
                            self.variables['cvf_'+strain][index][col]: self.parameters['cvcf'],
                            })
                        variables = np.hstack((variables, [
                            self.variables['b_'+strain][index][col], self.variables['g_'+strain][index][col],
                            self.variables['b+1_'+strain][index][next_col], self.variables['cvt_'+strain][index][col],
                            self.variables['cvt_'+strain][index][col], self.variables['cvf_'+strain][index][col]]))
                        constraints = np.hstack((constraints, 
                            [self.constraints['gc_'+strain][index][col], self.constraints['cvc_'+strain][index][col]]))
    
                        if last_column:
                            break
                    if last_column:
                        break
                    
        # define non-concentration variables
        # 6.7E4 total loops
        time_2 = process_time()
        print(f'Done with predictory loops: {(time_2-time_1)/60} min')
        for r_index, met in enumerate(self.phenotypes_parsed_df[0]):  # 25
            self.variables["c_"+met]:dict = {}; self.variables["c+1_"+met]:dict = {}; self.constraints['dcc_'+met]:dict = {}
            for name, parsed_df in self.dataframes.items():  # 3
                for index in parsed_df[0]:  # 66
                    self.variables["c_"+met][index]:dict = {}; self.variables["c+1_"+met][index]:dict = {}
                    self.constraints['dcc_'+met][index]:dict = {}
                    for col in parsed_df[1]:  # 27             
                        # define biomass measurement conversion variables 
                        self.variables["c_"+met][index][col] = Variable(
                            _variable_name("c_", met, index, col), lb=0, ub=1000)    
                        self.variables["c+1_"+met][index][col] = Variable(
                            _variable_name("c+1_", met, index, col), lb=0, ub=1000)    
                        
                        # c_{met} + dt*sum_k^K() - c+1_{met} = 0
                        self.constraints['dcc_'+met][index][col] = Constraint(
                            self.variables["c_"+met][index][col] 
                            + np.dot(self.phenotypes_parsed_df[2][r_index]*self.parameters['timestep_s'],
                                     np.array([self.variables['g_'+strain][index][col] for strain in self.phenotypes_parsed_df[1]])) 
                            - self.variables["c+1_"+met][index][col],
                            ub=0, lb=0, name=_constraint_name("dcc_", met, index, col))
                        
                        variables = np.hstack((variables, [self.variables["c_"+met][index][col], self.variables["c+1_"+met][index][col]]))
                        constraints = np.concatenate((constraints, [self.constraints['dcc_'+met][index][col]]))
                    
        # 5.3E3 loops
        time_3 = process_time()
        print(f'Done with metabolites loops: {(time_3-time_2)/60} min')
        for name, parsed_df in self.dataframes.items():  # 3
            self.variables[name+'__bio']:dict = {}; self.variables[name+'__diff']:dict = {}
            self.variables[name+'__conversion'] = Variable(name+'__conversion', lb=0, ub=1000)
            self.constraints[name+'__bioc']:dict = {}; self.constraints[name+'__diffc']:dict = {}  # diffc is defined latter
            for r_index, index in enumerate(parsed_df[0]):  # 66
                self.variables[name+'__bio'][index]:dict = {}; self.variables[name+'__diff'][index]:dict = {}
                self.constraints[name+'__bioc'][index]:dict = {}; self.constraints[name+'__diffc'][index]:dict = {}
                for col in parsed_df[1]:  # 27
                    self.variables[name+'__bio'][index][col] = Variable(      
                        _variable_name(name, '__bio', index, col), lb=0, ub=1000)
                    self.variables[name+'__diff'][index][col] = Variable(       
                        _variable_name(name, '__diff', index, col), lb=-100, ub=100)
                    # {name}__conversion*datum = {name}__bio
                    self.constraints[name+'__bioc'][index][col] = Constraint(
                         self.variables[name+'__conversion']*parsed_df[2][r_index, int(col)-1] - self.variables[name+'__bio'][index][col], 
                         name=_constraint_name(name, '__bioc', index, col), lb=0, ub=0)
                    
                    obj_coef[self.variables[name+'__diff'][index][col]] = 1
                    variables = np.hstack((variables, 
                        [self.variables[name+'__bio'][index][col], self.variables[name+'__diff'][index][col],
                         self.variables[name+'__conversion']]))
                    constraints = np.concatenate((constraints, [self.constraints[name+'__bioc'][index][col]]))
                
        # 3.2E4 loops
        time_4 = process_time()
        print(f'Done with experimental loops: {(time_4-time_3)/60} min')
        for name, parsed_df in self.dataframes.items():  # 3
            for r_index, index in enumerate(parsed_df[0]):  # 66
                for col in parsed_df[1]:  # 27 
                    total_biomass: Add = 0; signal_sum: Add = 0; from_sum: Add = 0; to_sum: Add = 0
                    for strain in self.phenotypes_parsed_df[1]:  # 6
                        total_biomass += self.variables["b_"+strain][index][col]
                        from_sum += self.species_phenotypes_bool_df.loc[name, strain]*self.variables['cvf_'+strain][index][col]
                        to_sum += self.species_phenotypes_bool_df.loc[name, strain]*self.variables['cvt_'+strain][index][col]
                        if 'OD' not in name:  # the OD strain has a different constraint
                            signal_sum += self.species_phenotypes_bool_df.loc[name, strain]*self.variables["b_"+strain][index][col]

                    self.constraints[name+'__diffc'][index][col] = Constraint(   
                         self.variables[name+'__bio'][index][col]-signal_sum
                         - self.variables[name+'__diff'][index][col], 
                         name=_constraint_name(name, '__diffc', index, col), lb=0, ub=0)
                    
                    if "stationary" in strain:
                        # b_{strain} - sum_k^K(pheno_bool*cvf) + sum_k^K(pheno_bool*cvt) - b+1_{strain} = 0
                        self.constraints['dbc_'+strain][index][col] = Constraint(
                            self.variables['b_'+strain][index][col] - from_sum + to_sum 
                            - self.variables['b+1_'+strain][index][next_col],
                            ub=0, lb=0, name=_constraint_name("dbc_", strain, index, col))
                    else:
                        # -b_{strain} + dt*g_{strain} + cvf - cvt - b+1_{strain} = 0
                        self.constraints['dbc_'+strain][index][col] = Constraint(
                            self.variables['b_'+strain][index][col]
                            + self.parameters['timestep_s']*self.variables['g_'+strain][index][col]
                            + self.variables['cvf_'+strain][index][col] - self.variables['cvt_'+strain][index][col]
                            - self.variables['b+1_'+strain][index][next_col],
                            ub=0, lb=0, name=_constraint_name("dbc_", strain, index, col))
                    
                    np.hstack((constraints, [
                        self.constraints[name+'__diffc'][index][col], self.constraints['dbc_'+strain][index][col]]))
                            
        time_5 = process_time()
        print(f'Done with the dbc & diffc loop: {(time_5-time_4)/60} min')
        # construct the problem
        self.problem.add(np.concatenate((constraints, variables)))
        self.problem.objective = Objective(Zero, direction="min")
        self.problem.objective.set_linear_coefficients(obj_coef)
                
    def _export(self, solution, print_conditions, print_lp):
        with open('solution.json', 'w') as sol_out:
            json.dump(solution, sol_out, index = 4)
        zipped_output = np.array(['solution.json'])
        if print_conditions:
            DataFrame(self.parameters).to_csv('parameters.csv')
            zipped_output = np.concatenate((zipped_output, 'constraints.csv'))
        if print_lp:
            zipped_output = np.concatenate((zipped_output, 'mscommfitting.lp'))
            with open('mscommfitting.lp', 'w') as lp:
                lp.write(self.problem.solver)
                
        # zip the results
        # if len(zipped_output) == 0:
        #     raise ValueError(f'One of the arguments < print_conditions ({print_conditions}), print_lp ({print_lp}) > must be True.')
        sleep(2)
        with ZipFile('msComFit.zip', 'w', compression = ZIP_LZMA) as zp:
            for file in zipped_output:
                zp.write(file)
                os.remove(file)
                
    def compute(self, print_conditions: bool = True, print_lp: bool = True):
        solution = self.problem.optimize()
        self._export(solution, print_conditions, print_lp)       
        