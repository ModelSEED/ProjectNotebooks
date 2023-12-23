import modelseedpy
from modelseedpy.biochem import from_local
import cobra
import sys

# locally import the ModelSEED API 
from mseditorapi import MSEditorAPI
modelseed_path = '..\\..\\..\\Biofilm growth code\\GSWL code\\ModelSEEDDatabase'
modelseed = modelseedpy.biochem.from_local(modelseed_path)


def test_name(rxn_id = 'rxn00002'):
    #reinstantiate the COBRA model
    bigg_model_path = '.\\e_coli_core metabolism from BiGG.json'
    model = cobra.io.load_json_model(bigg_model_path)

    # store the reaction names
    original_name = modelseed.get_seed_reaction(rxn_id).data['name'] 
    MSEditorAPI.add_ms_reaction(model = model, modelseed = modelseed, rxn_id = rxn_id)
    new_cobra_name = model.reactions.get_by_id(rxn_id).name
    
    # evaluate the reaction names between ModelSEED and COBRA
    assert original_name == new_cobra_name


def test_backward(rxn_id = 'rxn00002'):
    #reinstantiate the COBRA model
    bigg_model_path = '.\\e_coli_core metabolism from BiGG.json'
    model = cobra.io.load_json_model(bigg_model_path)
    
    # implement a reaction 
    MSEditorAPI.add_ms_reaction(model = model, modelseed = modelseed, rxn_id = rxn_id, direction = 'backward')

    # store the reaction
    cobra_reaction = model.reactions.get_by_id(rxn_id)
    
    # assert that the reversibility equates the expected direction
    assert cobra_reaction.upper_bound == 0
    assert cobra_reaction.lower_bound == -1000


def test_reversible(rxn_id = 'rxn00002'):
    #reinstantiate the COBRA model
    bigg_model_path = '.\\e_coli_core metabolism from BiGG.json'
    model = cobra.io.load_json_model(bigg_model_path)
    
    # implement a reaction 
    MSEditorAPI.add_ms_reaction(model = model, modelseed = modelseed, rxn_id = rxn_id, direction = 'reversible')

    # store the reaction
    cobra_reaction = model.reactions.get_by_id(rxn_id)
    
    # assert that the reversibility equates the expected direction
    assert cobra_reaction.upper_bound == 1000
    assert cobra_reaction.lower_bound == -1000
    

def test_reaction_presence(rxn_id = 'rxn00004'):
    #reinstantiate the COBRA model
    bigg_model_path = '.\\e_coli_core metabolism from BiGG.json'
    model = cobra.io.load_json_model(bigg_model_path)
    
    # add a reaction to the model
    MSEditorAPI.add_ms_reaction(model = model, modelseed = modelseed, rxn_id = rxn_id)

    # determine the existance of the reaction in the new model instance
    for reaction in model.reactions:
        if reaction.id == rxn_id:
            assert True