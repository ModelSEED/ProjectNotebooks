# -*- coding: utf-8 -*-
from pandas import read_table, read_csv, DataFrame
from optlang import Variable, Constraint
from numpy import array
# from cplex import Cplex
import sys, os

def _variable_name(name, suffix, column, index):
    return '-'.join([name+suffix, column, index])

def _constraint_name(name, suffix, column, index):
    return '_'.join([name+suffix, column, index])

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
        self.species_phenotypes_bool_df = DataFrame(columns=self.phenotypes_df.columns)
        for path, content in signal_tsv_paths.items():
            signal = os.path.splitext(path)[0].split("_")[0]
            # define the signal dataframe
            self.signal_species[signal] = content['name']
            self.dataframes[signal] = read_table(path)
            self.dataframes[signal].index = self.dataframes[signal][content['index_col']]
            for col in content['drops']+[content['index_col']]:
                self.dataframes[signal].drop(col, axis=1, inplace=True)
        
            # differentiate the phenotypes for each species
            self.parameters[signal] = {}
            self.variables[signal+'__coef'], self.variables[signal+'__bio'], self.variables[signal+'__diff'] = {}, {}, {}
            self.constraints[signal+'__bioc'], self.constraints[signal+'__diffc'] = {}, {}
            if "OD" not in signal:
                self.species_phenotypes_bool_df.loc[signal] = [
                    1 if self.signal_species[signal] in pheno else 0 for pheno in self.phenotypes_df.columns]  # !!! This must still be applied
                
    def define_problem(self, conversion_rates={'cvt': 0, 'cvf': 0}, constraints={'cvc': {}}):  # !!! the arguments are never used
        # define biomass abundance, growth rate, and concentration dictionaries
        self.variables['bio_abundance'], self.variables['growth_rate'], self.variables['conc_met'] = {}, {}, {}
        
        # define data variables for each respective signal, time, and trial
        for name, df in self.dataframes.items():
            # define biomass measurement conversion variables 
            self.variables[name+'__conversion'] = Variable(name+'__conversion', type='continuous', lb=0, ub=1000)
            signal_bool = self.species_phenotypes_bool_df[name]
            for index in df.index:
                for col in df:
                    experimental_signal = df.loc[index, col]
                    b = _constraint_name(name, '__diffc', column, index)  # 
                    
                    # define variables for each species 
                    self.variables[name+'__bio'][(column, index)] = Variable(        # biomass abundance
                        _variable_name(name, '__bio', column, index), lb=0, ub=1000)
                    self.variables[name+'__diff'][(column, index)] = Variable(       # biomass growth 
                        _variable_name(name, '__growth', column, index), lb=-100, ub=100)
                    self.variables[name+'__diff'][(column, index)] = Variable(       # differential between measured and predicted biomass abundace 
                        _variable_name(name, '__diff', column, index), lb=-100, ub=100)
                    
                    # construct constraints
                    self.constraints[name+'__bioc'][(column, index)] = Constraint(   # {name}__conversion*datum = {name}__bio
                         self.variables[name+'__conversion']*experimental_signal, 
                         name=_constraint_name(name, '__bioc', column, index), 
                         lb=self.variables[name+'__bio'][(column, index)],
                         ub=self.variables[name+'__bio'][(column, index)]
                        )
                    
                    
                    coef = {self.variables[name+'__bio'][(column, index)]: 1   # !!! must be summed over all species
                            }
                    self.constraints[name+'__bioc'][(column, index)] = Constraint(   # {name}__bio - sum_k^K(signal_bool*{name}__conversion*datum) = {name}__diff
                         self.variables[name+'__conversion']*experimental_signal, 
                         name=_constraint_name(name, '__bioc', column, index), 
                         lb=self.variables[name+'__bio'][(column, index)],
                         ub=self.variables[name+'__bio'][(column, index)]
                        )
            
                    # define 
                    if 'OD' in name:              
                        for strain in self.phenotypes:
                            z = _variable_name("b_", strain, column, index) 
                            a = _variable_name("g_", strain, column, index)
                            b = _variable_name("cvt_", strain, column, index)
                            c = _variable_name("cvf_", strain, column, index) 
                            d = _variable_name("dbc_", strain, column, index)
                            e = _variable_name("gc_", strain, column, index) 
                            f = _variable_name("cvc_", strain, column, index)
                            
                            self.variables["b"][z] = Variable(z, lb=0, ub=1000)
                            self.variables["g"][a] = Variable(a, lb=0, ub=1000)
                            self.variables["cvt"][b] = Variable(b, lb=0, ub=100)
                            self.variables["cvf"][c] = Variable(c, lb=0, ub=100)
                            self.constraints["dbc"][d] = d
                            self.constraints["gc"][e] = e
                            self.constraints["cvc"][f] = f
            
            for met, col in self.phenotypes_df.to_dict().items():
                x = _variable_name("c_", met, column, index)
                y = _variable_name("dcc_", met, column, index)
                self.variables["c"][x] = Variable(x, lb=0, ub=1000)
                self.constraints["dcc"][y] = y

        # define variables
        for var in self.varables:  
            if '__coef' in var:
                self.varables[var] = Variable(var, type="continuous", lb=0, ub=1000)

        
        # define changes in concentrations, growth rate, and biomass values
        self.constraints['dbc'], self.constraints['gc'], self.constraints['dcc'] = {}, {}, {}
        self.constraints.update(constraints)
        
        self.parameters.update({
            "dt": 600,      # Size of time step used in simulation in seconds
            "cvct": 1,      # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf": 1,      # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv": 1,       # This is the highest fraction of biomass for a given strain that can change phenotypes in a single time step
            "cvmin": 0.1,     # This is the lowest value the limit on phenotype conversion goes, 
            "y": 1,         # Stoichiometry for interaction of strain k with metabolite i
            "v": 1,
        })
        self.parameters.update(conversion_rates)
        
       
        
     
    def _parameterize(self,):
        for name, df in self.dataframes.items():
            for index in df.index:
                for col in df:
                    a = '-'.join([name, col, index])                    
                    self.parameters[name][a]= df.loc[index, col]
                
        for met in self.phenotypes_df.index:
            for col in self.phenotypes_df:
                d = '-'.join(["PS_", col, met])
                z = '-'.join(["V_", met, col, index])
                self.parameters["y"][d] = self.phenotypes_df.loc[met, col]
                self.parameters["v"][z] = z
                
    def _set_objective(self,):  # !!! the numerical and variable values must be distinguished
        coef = {}
        for name, df in self.dataframes.items():
            # env_content = array(['-'.join([name + '__diff', col, index]) for col in df for index in df.index])
            coef[_variable_name(name, '__diff', col, index)] = _variable_name(name, '__diff', col, index)
            if "OD" in name:
                for strain in self.phenotypes_df:
                    coef.update({
                        sum(items for items in parameters["cvct"] * "cvt_"+strain+ "_"+col+"_"+index): -1,
                        sum(items for items in parameters["cvcf"] * "cvf_"+strain+ "_"+col+"_"+index): -1})
        obj = Objective(coef, direction="min")
        
                
    def _export(self, filename):
        with open(filename, 'w') as out:
            out.writelines(
                ['|'.join(['parameter', param, content+": "+str(self.parameters[param][content])]) for param, content in self.parameters.items()]
                + ['|'.join(['variable', var, content+": "+str(self.parameters[var][content])]) for var, content in self.variables.items()]
                + ['|'.join(['constant', const, content+": "+str(self.parameters[const][content])]) for const, content in self.constants.items()])
                
    def calculate(self,):
        const_list = []
        for nme, df in self.dataframes.items():
            for index in df.index:
                for col in df:
                    x = _variable_name(nme,'__bio', col, index)
                    a = _variable_name(nme, '', col, index])
                    const = Constraint(
                        self.variables[nme+'__bio'][x]/ self.variables[nme+'__coef']["C_"+nme+'__coef'],
                        ub=self.parameters[nme][a], lb=self.parameters[nme][a], name=nme+"__bioc")
                    const_list.append(const)
                    
                    x = _variable_name(nme,'__diff', col, index)
                    y = _variable_name(nme,'__bio', col, index)
                    to_sum = []
                    for strain in self.phenotypes_df:
                        to_sum.append(_variable_name("b_", strain, col, index))
                    const = Constraint(
                        (self.variables["rfpb"][y] - self.variables["rfpe"][x])/sum(self.variables["b"][item] for item in to_sum),
                        ub=3,  # does this look right?
                        lb=3, name=nme+"__diffc")
                    const_list.append(const)
                    
                    if "OD" in nme:
                        for strain in self.phenotypes_df:
                            a = _constraint_name("g_", strain, col, index)
                            b = _constraint_name("b_", strain, col, index) 
                            c = _constraint_name("v_", strain, col, index) 
                            const = Constraint(
                                self.variables["g"][a]/ self.variables["b"][b],
                                ub=self.parameters["v"][c], lb=self.parameters["v"][c],
                                name="gc_"+strain+ "_"+col+"_"+index)
                            const_list.append(const)

                            b = _constraint_name("b_", strain, col, index)
                            c = _constraint_name("cvt_", strain, col, index)
                            const = Constraint(
                                self.variables["cvt"][c] - (self.parameters["bcv"]*self.variables["b"][b]),
                                ub=self.parameters["cvmin"], lb=0,  # does this make sense?
                                name=_constraint_name("cvc_", strain, col, index)
                            const_list.append(const)
                            if "stationary" in strain:  # not done coding this constraint
                                b1 = _constraint_name("b1_", strain, col, index)
                                col2 = int(col)+1
                                b2 = _constraint_name("b_", strain, col, index)  # need to do something about the variables that are out of time range
                                c = _constraint_name("cvt_", strain, col, index)
                                f = _constraint_name("cvf_", strain, col, index)
                                const = Constraint(
                                    self.variables["b"][b2]-self.variables["b"][b1],
                                    ub=0, lb=0, name=_constraint_name("dbc_", strain, col, index))
                                const_list.append(const)
        self.model.add(const_list)               
