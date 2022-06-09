# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 18:45:54 2022

@author: Andrew Freiburger
"""

from pandas import read_table, read_csv, DataFrame
from optlang import Model, Variable, Constraint
from optlang.cplex_interface import Model
from configparser import ConfigParser
from cplex import Cplex
import sys, os

class MSCommFitting():
    def __init__(self):
        cplex_problem = Cplex()
        cplex_problem.read("baseproblem.lp")
        self.model = Model(problem=cplex_problem)
                
        config = ConfigParser()
        os.chdir('/Users/ffoflonker/Documents/Command_line/Notebooks/Synbio/ecoli_staph_coculture')
        config.read("config.cfg")
        paths = config.get("script", "syspaths").split(";")
        for path in paths:
            sys.path.append(path)
            
        # define data sources
        self.parameters, self.variables, self.constraints, self.dataframes = {}, {}, {}, {}
        
    def load_data(self,phenotypes_csv_path: dict = {},  # the dictionary of index names for each paths to signal CSV data that will be fitted
                  signal_tsv_paths: dict = {},       # the dictionary of index names for each paths to signal TSV data that will be fitted
                  ):
        self.num_species = len(signal_tsv_paths)-1 #!!! is this a good estimation for the number of species?
        for path, content in phenotypes_csv_path.items():
            with open(path, 'w') as data:
                self.phenotypes_df = read_csv(data)
                self.phenotypes_df.index = content['index']
                for col in content['drops']:
                    self.phenotypes_df.drop(col, axis=1, inplace=True)
        for path, content in signal_tsv_paths.items():
            with open(path, 'w') as data:
                self.dataframes[os.path.splitext(path)[0]] = read_table(data, delimiter ='\t')
                self.dataframes[os.path.splitext(path)[0]].index = content['index']
                for col in content['drops']:
                    self.dataframes[os.path.splitext(path)[0]].drop(col, axis=1, inplace=True)
                
    def define_conditions(self, 
                        conversion_rates = {'cvt':{}, 'cvf':{}},
                        constraints = {'cvc': {}},
                        ):
        species_phenotypes_bool_df = DataFrame(index = [signal for signal in self.dataframes if "OD" not in signal], columns = self.phenotypes_df.columns)
        species_phenotypes_bool_df.fillna(0)
        variables, constraints, parameters = {}, {}, {}
        variables['b'], variables['g'], variables['c'] = {}, {}, {}
        constraints['dbc'], constraints['gc'], constraints['dcc'], constraints['dbc'] = {}, {}, {}, {}
        parameters.update({
            "dt":{"dt":600},       # Size of time step used in simulation in seconds
            "cvct":{"cvct":1},     # Coefficient for the minimization of phenotype conversion to the stationary phase. 
            "cvcf":{"cvcf":1},     # Coefficient for the minimization of phenotype conversion from the stationary phase. 
            "bcv":{"bcv":1},       # This is the highest fraction of biomass for a given strain that can change phenotypes in a single time step
            "cvmin":{"cvmin":0.1}, # This is the lowest value the limit on phenotype conversion goes, 
            "y":{},                # Stoichiometry for interaction of strain k with metabolite i
            "v":{}
        })
        for signal in self.dataframes:
            parameters[signal] = {}
            variables[signal+'__c'], variables[signal+'__b'], variables[signal+'__e'] = {}, {}, {}
            constraints[signal+'__bc'], constraints[signal+'__ec'] = {}, {}
            if not "OD" in signal:
                for pheno in self.phenotypes_df.columns:
                    species_phenotypes_bool_df.loc[signal,pheno] = 1   #!!! This must still be applied in a later calculation
        self.variables.update(variables)
        self.constraints.update(constraints)
        self.parameters.update(parameters)

    def _constraints(self,):
        for var in self.varables:
            if '__c' in var:
                self.varables[var]['C_'+var] = Variable('C_'+var, lb=0,ub=1000)
        
        for name, df in self.dataframes.items():
            for index in df.index:
                for col in df:
                    x='-'.join([name+'__b', col, index])
                    y='-'.join([name+'__e', col, index])
                    a='-'.join([name+'__bc', col, index])
                    b='-'.join([name+'__ec', col, index])
                    
                    self.variables[name+'__b'][x]=Variable(x, lb=0,ub=1000)
                    self.variables[name+'__e'][y]=Variable(y, lb=-100,ub=100)
                    self.constraints[name+'__bc'][a]=a
                    self.constraints[name+'__ec'][b]=b
            
                    if 'OD' in name:              
                        for strain in self.phenotypes:
                            z='-'.join(["b_"+strain, col, index])
                            a='-'.join(["g_"+strain, col, index])
                            b='-'.join(["cvt_"+strain, col, index])
                            c='-'.join(["cvf_"+strain, col, index])
                            d='-'.join(["dbc_"+strain, col, index])
                            e='-'.join(["gc_"+strain, col, index])
                            f='-'.join(["cvc_"+strain, col, index])
                            
                            self.variables["b"][z]=Variable(z, lb=0,ub=1000)
                            self.variables["g"][a]=Variable(a, lb=0,ub=1000)
                            self.variables["cvt"][b]=Variable(b, lb=0,ub=100)
                            self.variables["cvf"][c]=Variable(c, lb=0,ub=100)
                            self.constraints["dbc"][d]=d
                            self.constraints["gc"][e]=e
                            self.constraints["cvc"][f]=f
            
            for met in self.phenotypes_df.index:
                for col in self.phenotypes_df.columns:
                    x='-'.join(["c_"+met, col, index])
                    y='-'.join(["dcc_"+met, col, index])
                    
                    self.variables["c"][x]=Variable(x, lb=0, ub=1000)
                    self.constraints["dcc"][y]=y
        
    def _parameterize(self,):
        #input parameters
        for name, df in self.dataframes.items():
            for index in df.index:
                for col in df:
                    a='-'.join([name, col, index])                    
                    self.parameters[name][a]= df.loc[index, col]
                
        for met in self.phenotypes_df.index:
            for col in self.phenotypes_df:
                d='-'.join(["PS_", col, met])
                z='-'.join(["V_", met, col, index])
                self.parameters["y"][d]= self.phenotypes_df.loc[met,col]
                self.parameters["v"][z]=z
                
    def calculate(self,):
        const_list=[]
        for nme, df in self.dataframes.items():
            for index in df.index:
                for col in df:
                    x='-'.join([nme+'__b', col, index])
                    a='-'.join([nme, col, index])
                    const = Constraint(
                            self.variables[nme+'__b'][x]/ self.variables[nme+'__c']["C_"+nme+'__c'],
                            ub=self.parameters[nme][a],
                            lb=self.parameters[nme][a],
                            name=nme+"__bc",
                        )
                    const_list.append(const)
                    
                    x='-'.join([nme+'__e', col, index])
                    y='-'.join([nme+'__b', col, index])
                    to_sum=[]
                    for strain in self.phenotypes_df:
                        to_sum.append('-'.join(["b_"+strain, col, index]))
                    const = Constraint(
                            (self.variables["rfpb"][y] - self.variables["rfpe"][x])/sum(self.variables["b"][item] for item in to_sum),
                            ub=3, #does this look right?
                            lb=3,
                            name=nme+"__ec",
                        )
                    const_list.append(const)
        self.model.add(const_list)               
