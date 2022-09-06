from collections import OrderedDict
from numpy import negative
from warnings import warn
import json, re, os

class MSCompatibility():
    def __init__(self,
                 modelseed_db_path: str, # the local path to the ModelSEEDDatabase repository
                 printing = True         # specifies whether results are printed
                 ):       
        self.printing = printing
        
        # import and parse ModelSEED Database reactions and compounds
        with open(os.path.join(modelseed_db_path, 'Biochemistry', 'reactions.json')) as rxns:
            self.reactions = OrderedDict(json.load(rxns))
            self.reaction_ids = OrderedDict()
            for rxn in self.reactions:
                self.reactions_ids[rxn['id']] = rxn['name']
        with open(os.path.join(modelseed_db_path, 'Biochemistry', 'compounds.json')) as rxns:
            self.compounds = json.load(rxns)
            self.compound_names = self.compound_ids = OrderedDict()
            for cpd in self.compounds:
                self.compound_ids[cpd['id']] = cpd['name']
                names = [name.strip() for name in cpd['aliases'][0].split(';')]
                for name in names:
                    self.compound_names[name] = cpd['id']
            
    def _parse_rxn_string(self,reaction_string):
        # parse the reaction string
        if '<=>' in reaction_string:
            compounds = reaction_string.split('<=>')
        elif '-->' in reaction_string:
            compounds = reaction_string.split('-->')
        elif '=>' in reaction_string:
            compounds = reaction_string.split('=>')
        elif '<=' in reaction_string:
            compounds = reaction_string.split('<=')
        else:
            warn(f'The reaction string {reaction_string} has an unexpected reagent delimiter.')
        reactant, product = compounds[0], compounds[1]
        reactants = [x.strip() for x in reactant.split('+')]
        products = [x.strip() for x in product.split('+')]
        reactant_met = [x.split(' ') for x in reactants]
        product_met = [x.split(' ') for x in products]
        
        # assemble a reaction dictionary that is amenable with the add_metabolites function of COBRA models
        reaction_dict = {}
        for met in reactant_met:
            if len(met) == 1:
                met.insert(0, '1') 
            stoich = float(re.search('(\d+)', met[0]).group())
            reaction_dict[met[1]] = negative(stoich)
        for met in product_met:
            if len(met) == 1:
                met.insert(0, '1') 
            stoich = float(re.search('(\d+)', met[0]).group())
            reaction_dict[met[1]] = stoich
            
        return reaction_dict
        
    def standardize_MSD(self,model):
        for met in model.metabolites:
            # standardize the metabolite names
            met.name = self.compound_ids[met.id]

        for rxn in model.reactions:
            # standardize the reaction names
            rxn.name = self.reactions_ids[rxn.id]
            
            # standardize the reactions to the ModelSEED Database
            for index, rxn_id in enumerate(self.reaction_ids):
                if rxn_id == rxn.id:
                    reaction_string = self.reaction[index]['equation']
                    reaction_dict = self._parse_rxn_string(reaction_string)
                    break
            
            for met in rxn.metabolites:
                rxn.add_metabolites({met:0}, combine = False)
            rxn.add_metabolites(reaction_dict, combine = False)
        
        return model
    
    def compare_models(self, model_1, model_2,       # arbitrary cobrakbase models
                       metabolites: bool = True,     # contrast metabolites (True) or reactions (False) between the models
                       standardize: bool = False     # standardize the model names and reactions to the ModelSEED Database
                       ): 
        if metabolites: # contrast metabolites 
            misaligned_metabolites = []
            compared_met_counter = 0
            for met in model_1.metabolites:
                if met.id in model_2.metabolites: 
                    if met.name != model_2.metabolites[met].name:
                        print(f'\nmisaligned met {met.id} names\n', f'model1: {met.name}', f'model2: {model_2.metabolites[met].name}')
                        misaligned_metabolites.append({
                                    'model_1': met.name,
                                    'model_2': model_2.metabolites[met].name,
                                })
                        if self.printing:
                            print(model_1.metabolites[met].name, model_2.metabolites[met].name)
                        
                    compared_met_counter += 1
                
            if self.printing:
                print(f'''\n\n{compared_met_counter} of the {len(model_1.metabolites)} model_1 metabolites and {len(model_2.metabolites)} model_2 metabolites are shared and were compared.''')        
        else: # contrast reactions 
            model2_rxns = {rxn.id:rxn for rxn in model_2.reactions}
            misaligned_reactions = []
            
            compared_rxn_counter = 0
            for rxn in model_1.reactions:
                if rxn['id'] in model2_rxns:
                    if rxn.name != model2_rxns[rxn.id].name:
                        print(f'\nmisaligned reaction {rxn.id} names\n', ' model1  ',rxn.name, ' model2  ',model2_rxns[rxn.id].name)
                        misaligned_reactions.append({
                                    'model_1': rxn.name,
                                    'model_2': model2_rxns[rxn.id].name,
                                })                            
                    elif rxn.reaction != model2_rxns[rxn.id].reaction:                            
                        print(f'\nmisaligned reaction {rxn.id} reagents\n', 'model1: ',rxn.reaction, 'model2: ',model2_rxns[rxn.id].reaction)
                        misaligned_reactions.append({
                                    'model_1': rxn.reaction,
                                    'model_2': model2_rxns[rxn.id].reaction,
                                })

                    compared_rxn_counter += 1
                    
            if self.printing:
                print(f'''\n\n{compared_rxn_counter}/{len(model_1.reactions)} model_1 reactions are shared with model_2 and were compared.''')
                
        if standardize:
            model_1 = self.standardize_MSD(model_1)
            model_2 = self.standardize_MSD(model_2)
            
        return misaligned_reactions, model_1, model_2
        
    def exchanges(self,
                  model # cobrakbase model
                  ):
        # homogenize isomeric metabolites in exchange reactions
        unique_base_met_id = OrderedDict()
        unique_base_met_name = []
        for rxn in model.reactions:
            if 'EX_' in rxn.id:
                for met in rxn.metabolites:
                    base_name = ''.join(met.name.split('-')[1:])
                    base_name_index = unique_base_met_name.index(base_name)
                    base_name_id = list(unique_base_met_id.keys())[base_name_index]
                    if base_name not in unique_base_met_name:
                        unique_base_met_name.append(base_name)
                        unique_base_met_id[met.id] = met
                    elif met.id != base_name_id:
                        # replace isomers of different IDs with a standard isomer
                        print('original reaction: ',rxn)
                        for rxn in met.reactions:
                            rxn.add_metabolites({
                                        met: 0, unique_base_met_id[base_name_id]: rxn.metabolites[met]
                                    }, 
                                    combine = False)
                        print('changed reaction: ',rxn)
                
        # standardize model metabolite IDs with the ModelSEED Database
        for met in model.metabolites:
            # correct non-standard IDs
            if 'cpd' not in met.id:
                original_id = met.id
                for name in self.compound_names:
                    if met.name == name:
                        met.id = self.compound_names[name]     
                if original_id == met.id:
                    warn(f'The metabolite {met.id} is not recognized by the ModelSEED Database')
        
        return model