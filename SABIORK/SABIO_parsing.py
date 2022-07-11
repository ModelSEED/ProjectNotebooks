# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:32:00 2022

@author: Andrew Freiburger
"""

import requests
from pandas import read_table

class parse_SABIO:
    
    def get_sabio_compounds(self,):
        QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/searchCompoundDetails'
        query = {"SabioCompoundID":"*", "fields[]":["Name","ChebiID","PubChemID","InChI","SabioCompoundID","KeggCompoundID","Smiles"]}
        request = requests.get(QUERY_URL, params = query)
        request.raise_for_status()
        with open('sabio_reagents.txt', 'w', encoding='utf-8') as out:
            out.write(request.text)
        self.reagents_df = read_table('sabio_reagents.txt')
        self.reagents_df.to_csv('sabio_reagents.csv')
        
    def get_sabio_reactions(self,):
        # reaction reagents
        # reagents_URL = 'http://sabiork.h-its.org/sabioRestWebServices/searchReactionParticipants'
        # query = {"SabioCompoundID":"*", "fields[]":["Name","Role","SabioCompoundID","ChebiID","PubChemID","KeggCompoundID", "InChI","Smiles","StochiometricValue"]}
        # request = requests.get(reagents_URL, params = query)
        # request.raise_for_status()
        # with open('sabio_reaction_reagents.txt', 'w') as out:
        #     out.write(request.text)
        # df = read_table('sabio_reaction_reagents.txt')
        # df.to_csv('sabio_reaction_reagents.csv')
        
        reagents2_URL = 'http://sabiork.h-its.org/sabioRestWebServices/searchReactionModifiers'
        query = {"SabioReactionID":"*", "fields[]":["EntryID","Name","Role","SabioCompoundID","ChebiID","PubChemID","KeggCompoundID","InChI","Smiles"]}
        request = requests.get(reagents2_URL, params = query)
        request.raise_for_status()
        with open('sabio_reaction_modifiers.txt', 'w', encoding='utf-8') as out:
            out.write(request.text)
        self.rxn_modifiers_df = read_table('sabio_reaction_modifiers.txt')
        self.rxn_modifiers_df.to_csv('sabio_reaction_modifiers.csv')
        
        # reaction metadata
        QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/searchReactionDetails'
        query = {"SabioReactionID":"*", "fields[]":["KeggReactionID","SabioReactionID","Enzymename","ECNumber", "UniProtKB_AC","ReactionEquation","TransportReaction"]}
        request = requests.get(QUERY_URL, params = query)
        request.raise_for_status()
        with open('sabio_reaction_details.txt', 'w', encoding='utf-8') as out:
            out.write(request.text)
        self.rxn_reagents_df = read_table('sabio_reaction_details.txt')
        self.rxn_reagents_df.to_csv('sabio_reaction_details.csv')
        
    def combine_reagents(self,):
        pass