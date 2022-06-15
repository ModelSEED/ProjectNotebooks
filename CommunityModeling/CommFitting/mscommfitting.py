# -*- coding: utf-8 -*-
from pandas import read_table, read_csv, DataFrame
from optlang import Variable, Constraint, Objective, Model
from numpy import array
# from cplex import Cplex
import os

def _variable_name(name, suffix, col, index):
    return '-'.join([name+suffix, col, index])

def _constraint_name(name, suffix, col, index):
    return '_'.join([name+suffix, col, index])

def _variance():
    pass

class MSCommFitting():
    def __init__(self):
        self.parameters, self.variables, self.constraints, self.dataframes = {}, {}, {}, {}
        
    def load_data(self, phenotypes_csv_paths: dict = {},  # the dictionary of index names for each paths to signal CSV data that will be fitted
                  signal_tsv_paths: list = {},       # the dictionary of index names for each paths to signal TSV data that will be fitted
                  ):
        self.num_species = len(signal_tsv_paths)-1  # !!! is this a good estimation for the number of species?
        for path, content in phenotypes_csv_paths.items():
            self.phenotypes_df = read_csv(path)
            self.phenotypes_df.index = self.phenotypes_df[content['index_col']]
            for col in content['drops']+[content['index_col']]:
                self.phenotypes_df.drop(col, axis=1, inplace=True)
                
        self.signal_species = {}
        self.species_phenotypes_bool_df = DataFrame(cols=self.phenotypes_df.cols)
        for path, name in signal_tsv_paths.items():
            signal = os.path.splitext(path)[0].split("_")[0]
            # define the signal dataframe
            self.signal_species[signal] = name
            self.dataframes[signal] = read_table(path)
            self.dataframes[signal].index = self.dataframes[signal]['Well']
            for col in ['Plate', 'Cycle', 'Well']:
                self.dataframes[signal].drop(col, axis=1, inplace=True)
        
            # differentiate the phenotypes for each species
            self.parameters[signal] = {}
            self.variables[signal+'__coef'], self.variables[signal+'__bio'], self.variables[signal+'__diff'] = {}, {}, {}
            self.constraints[signal+'__bioc'], self.constraints[signal+'__diffc'] = {}, {}
            if "OD" not in signal:
                self.species_phenotypes_bool_df.loc[signal] = [
                    1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_df.cols]  # !!! This must still be applied
                
    def define_problem(self, conversion_rates={'cvt': 0, 'cvf': 0}, constraints={'cvc': {}}):  # !!! the arguments are never used
        self.variables['bio_abundance'], self.variables['growth_rate'], self.variables['conc_met'], obj_coef = {}, {}, {}, {}
        constraints, variables = [], []
        self.problem = Model()
        
        self.parameters.update({
            "timestep_s": 600,      # Size of time step used in simulation in seconds
            "cvct": 1,      # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf": 1,      # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv": 1,       # This is the highest fraction of biomass for a given strain that can change phenotypes in a single time step
            "cvmin": 0.1,     # This is the lowest value the limit on phenotype conversion goes, 
            "y": 1,         # Stoichiometry for interaction of strain k with metabolite i
            "v": 1,
        })
        self.parameters.update(conversion_rates)
        
        # define predicted variables for each strain, time, and trial
        self.constraints['dbc'], self.constraints['gc'], self.constraints['dcc'] = {}, {}, {}
        strain_independent = True
        for strain in self.phenotypes_df:
            for name, df in self.dataframes.items(): 
                # define biomass measurement conversion variables 
                self.variables[name+'__conversion'] = Variable(name+'__conversion', type='continuous', lb=0, ub=1000)
                if strain_independent:
                    # define biomass measurement conversion variables 
                    self.variables[name+'__conversion'] = Variable(name+'__conversion', type='continuous', lb=0, ub=1000)
                    signal_bool = self.species_phenotypes_bool_df[name]
                    for index in df.index:
                        self.variables[name+'__bio'][index], self.variables[name+'__diff'][index] = {}, {}
                        self.variables[name+'__diff'][index], self.constraints[name+'__bioc'][index] = {}, {}
                        for col in df:
                            total_biomass = [self.variables["b_"+strain][col][index] for strain in self.phenotypes_df] 
                            # measured biomass abundance
                            self.variables[name+'__bio'][index][col] = Variable(      
                                _variable_name(name, '__bio', col, index), lb=0, ub=1000)
                            
                            # biomass growth 
                            # self.variables[name+'__growth'][index][col] = Variable(     
                            #     _variable_name(name, '__growth', col, index), lb=-100, ub=100)
                            
                            # differential between measured and predicted biomass abundace 
                            self.variables[name+'__diff'][index][col] = Variable(       
                                _variable_name(name, '__diff', col, index), lb=-100, ub=100)
                            variables.extend([self.variables[name+'__bio'][index][col], self.variables[name+'__diff'][index][col]])
                            
                            # {name}__conversion*datum = {name}__bio
                            self.constraints[name+'__bioc'][index][col] = Constraint(
                                 self.variables[name+'__conversion']*experimental_signal - self.variables[name+'__bio'][index][col], 
                                 name=_constraint_name(name, '__bioc', col, index), 
                                 lb=0, ub=0)
                            constraints.extend(self.constraints[name+'__bioc'][index][col])
                    strain_independent = False
                for index in df.index: 
                    self.variables['b_'+strain][index], self.constraints['g_'+strain][index] = {}, {}
                    self.variables['cvt_'+strain][index], self.constraints['cvf_'+strain][index] = {}, {}
                    for col in df:
                        # define variables
                        self.variables['v_'+strain][index][col] = Variable(    # predicted kinetic coefficient
                            _variable_name("v_", strain, col, index), lb=0, ub=1000)                          
                        self.variables['b_'+strain][index][col] = Variable(    # predicted biomass abundance
                            _variable_name("b_", strain, col, index), lb=0, ub=1000)   
                        self.variables['g_'+strain][index][col] = Variable(    # biomass growth
                            _variable_name("g_", strain, col, index), lb=0, ub=1000)    
                        self.variables['cvt_'+strain][index][col] = Variable(  # conversion rate to the stationary phase
                            _variable_name("cvt_", strain, col, index), lb=0, ub=100)   
                        self.variables['cvf_'+strain][index][col] = Variable(  # conversion from to the stationary phase
                            _variable_name("cvf_", strain, col, index), lb=0, ub=100)
                        variables.extend([
                            self.variables['g_'+strain][index][col], self.variables['b_'+strain][index][col],
                            self.variables['cvt_'+strain][index][col], self.variables['cvf_'+strain][index][col]])
                            
                        # !!! t+1 constraints that are still under development
                        # self.constraints["dbc"][index][col] = Constraint(
                        #     _constraint_name("dbc_", strain, col, index), lb=0, ub=0)
                        self.constraints['gc_'+strain][index][col] = Constraint(
                            self.variables['g_'+strain][index][col] 
                            - self.parameters['v']*self.variables['b_'+strain][index][col],   # !!! The v kinetic coefficient may need to be an Optlang variable to be optimized
                            _constraint_name("gc_", strain, col, index), lb=0, ub=0)
                        self.constraints["cvc_"+strain][index][col] = Constraint(
                            self.parameters['bcv']*self.variables['b_'+strain][index][col] - self.parameters['cvmin'] 
                            - self.variables['cvt_'+strain][index][col],
                            _constraint_name("cvc_", strain, col, index), lb=0, ub=None)
                        # {name}__bio - sum_k^K(signal_bool*{name}__conversion*datum) = {name}__diff
                        signal_sum = [self.variables["b_"+strain][col][index]*self.species_phenotypes_bool_df.loc[name, strain] for strain in self.phenotypes_df] 
                        self.constraints[name+'__diffc'][index][col] = Constraint(   
                             (self.variables[name+'__bio'][index][col]-signal_sum)/total_biomass
                             - self.variables[name+'__diff'][index][col], 
                             name=_constraint_name(name, '__diffc', col, index), lb=0, ub=0)
                        if "stationary" in strain:  # not done coding this constraint
                            b1 = _constraint_name("b1_", strain, col, index)
                            col2 = int(col)+1
                            b2 = _constraint_name("b_", strain, col, index)  # need to do something about the variables that are out of time range
                            c = _constraint_name("cvt_", strain, col, index)
                            f = _constraint_name("cvf_", strain, col, index)
                            const = Constraint(
                                self.variables["b"][b2]-self.variables["b"][b1],
                                ub=0, lb=0, name=_constraint_name("dbc_", strain, col, index))
                        else:  # !!! The non-stationary constraint still must be developed
                            pass
                            
                        constraints.extend([
                            self.constraints["gc"][index][col], self.constraints["cvc"][index][col],
                            self.constraints['gc_'+strain][index][col], self.constraints["cvc"][index][col],
                            self.constraints[name+'__diffc'][index][col]])
                    
        # define metabolite variable/constraint
        for met, col in self.phenotypes_df.to_dict().items():
            for index in df.index:
                # define biomass measurement conversion variables 
                self.variables[name+'__conversion'] = Variable(name+'__conversion', type='continuous', lb=0, ub=1000)
                self.variables["c_"+met][index][col] = Variable(
                    _variable_name("c_", met, col, index), lb=0, ub=1000)
                variables.extend([self.variables[name+'__conversion'], self.variables["c_"+met][index][col]])
                # growth_stoich = sum([self.phenotypes_df.loc[met, col]*self.variables['g_'+strain][index][col] for strain in self.phenotypes_df])
                # self.constraints["dcc"][index][col] = Constraint(   # !!! t+1 constraints that are still under development
                #     self.variables["c_"+met][index][col] - self.parameters['timestep_s']*growth_stoich,
                #     _constraint_name('dcc_', met, col, index),
                #     ub=0, lb=0)
            
        # construct the objective function
        for name, df in self.dataframes.items():
            coef[_variable_name(name, '__diff', col, index)] = _variable_name(name, '__diff', col, index)
            if "OD" in name:
                for strain in self.phenotypes_df:
                    coef.update({
                        sum(items for items in self.parameters["cvct"] * "cvt_"+strain+ "_"+col+"_"+index): -1,
                        sum(items for items in self.parameters["cvcf"] * "cvf_"+strain+ "_"+col+"_"+index): -1})
                    
        # construct the problem
        self.problem.add(constraints+variables)
        self.problem.objective = Objective(obj_coef, direction="min")
                
    def _export(self, filename):
        with open(filename, 'w') as out:
            out.writelines(
                ['|'.join(['parameter', param, content+": "+str(self.parameters[param][content])]) for param, content in self.parameters.items()]
                + ['|'.join(['variable', var, content+": "+str(self.parameters[var][content])]) for var, content in self.variables.items()]
                + ['|'.join(['constant', const, content+": "+str(self.parameters[const][content])]) for const, content in self.constants.items()])