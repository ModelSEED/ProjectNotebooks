from collections import Counter
from itertools import combinations, chain
from typing import Iterable
from math import prod

from cobra.medium import minimal_medium
from modelseedpy.community.mscommunity import MSCommunity
from modelseedpy.community.mscompatibility import MSCompatibility
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Variable, Constraint, Objective
from numpy import mean
from time import process_time
from cobra import Reaction


class Smetana:
    def __init__(self, cobra_models: Iterable, community_model, min_growth):
        self.community, self.models = Smetana._load_models(cobra_models, community_model)
        self.media = MSCommunity.estimate_minimal_community_media(self.models, self.community, True, min_growth)

        

    @staticmethod
    def _load_models(cobra_models: Iterable, community_model=None):
        if not community_model:
            return MSCommunity.build_from_species_models(cobra_models, name="SMETANA_example", cobra_model=True), cobra_models
        # models = PARSING_FUNCTION(community_model) # TODO the individual models of a community model must be parsed
        return Smetana._compatibilize_models([community_model])[0], Smetana._compatibilize_models(cobra_models)

    @staticmethod
    def _compatibilize_models(cobra_models:Iterable):
        return MSCompatibility.align_exchanges(cobra_models, standardize=True, conflicts_file_name='exchanges_conflicts.json')

    @staticmethod
    def sc_score(cobra_models:Iterable=None, msdb_path:str=None, community_model=None, min_growth=0.1, n_solutions=100, abstol=1e-6):
        """Calculate the frequency of interspecies dependency in a community"""
        if not any([cobra_models, community_model]):
            community_model, cobra_models = Smetana._load_models(cobra_models, msdb_path, community_model)
        # disable all biomass reactions in the community
        for rxn in community_model.reactions:
            rxn.lower_bound = 0 if 'bio' in rxn.id else rxn.lower_bound

        # constrain all fluxes of with the species binary variable
        # c_{rxn.id}_lb: rxn < 1000*y_{species_id}
        # c_{rxn.id}_ub: rxn > -1000*y_{species_id}
        variables = {}
        constraints = []
        for model in cobra_models:  # TODO this can be converted to an MSCommunity object by looping through each index
            variables[model.id] = Variable(name=f'y_{model.id}', lb=0, ub=1, type='binary')
            for rxn in model.reactions:
                if "bio" not in rxn.id:
                    lb = Constraint(rxn.flux_expression - 1000*variables[model.id], name=f'c_{rxn.id}_lb', ub=0)
                    ub = Constraint(rxn.flux_expression + 1000*variables[model.id], name=f"c_{rxn.id}_ub", lb=0)
                    constraints.extend([lb, ub])
        community = FBAHelper.add_vars_cons(community_model, list(variables.values()) + constraints)

        # calculate the SCS
        scores = {}
        for model in cobra_models:
            com_model = community
            other_members = [other for other in cobra_models if other.id != model.id]
            # SMETANA_Biomass: bio1 > {min_growth}
            smetana_biomass = Constraint(sum(rxn for rxn in model.reactions if "bio" in rxn.id), name='SMETANA_Biomass', lb=min_growth)
            com_model = FBAHelper.add_vars_cons(com_model, [smetana_biomass])
            com_model = FBAHelper.add_objective(com_model, {f"y_{other.id}": 1.0 for other in other_members}, "min")
            previous_constraints, donors_list = [], []
            for i in range(n_solutions):
                sol = com_model.optimize()
                if sol.status != 'optimal':
                    scores[model.id] = None
                    break
                donors = [o for o in other_members if sol.values[f"y_{o.id}"] > abstol]
                donors_list.append(donors)

                # the community is iteratively reduced
                # c_{rxn.id}_lb: sum(y_{species_id}) < # iterations - 1
                previous_con = f'iteration_{i}'
                previous_constraints.append(previous_con)
                com_model = FBAHelper.add_vars_cons(com_model, list(Constraint(
                    sum(variables[o.id] for o in donors), name=previous_con, ub=len(previous_constraints) - 1)))

            # calculate the score if the loop completed without an error exit
            if i == n_solutions-1:
                donors_counter = Counter(chain(*donors_list))
                scores[model.id] = {o: donors_counter[o] / len(donors_list) for o in other_members}
        return scores

    @staticmethod
    def mu_score(cobra_models:Iterable, msdb_path:str, min_growth=0.1):
        """the fractional frequency of each received metabolite amongst all possible alternative syntrophic solutions"""
        scores = {}
        cobra_models = Smetana._compatibilize_models(cobra_models, msdb_path)
        comm_min_media = MSCommunity.estimate_minimal_community_media(cobra_models, min_growth=min_growth)
        for model in cobra_models:
            ex_rxns = {ex_rxn.id: met for ex_rxn in FBAHelper.exchange_reactions(model) for met in ex_rxn.metabolites}
            min_media = minimal_medium(model, min_growth, minimize_components=True)
            counter = Counter(min_media)
            scores[model.id] = {met: counter[ex] / len(min_media) for ex, met in ex_rxns.items()}
        return scores

    @staticmethod
    def mp_score(cobra_models:Iterable=None, msdb_path:str=None, community_model=None, min_growth=0.1, abstol=1e-3):  # TODO this must be validated with the Machado formulation
        """Discover the metabolites that each species contributes to a community"""
        if not any([community_model, cobra_models]):
            community_model, cobra_models = Smetana._load_models(cobra_models, msdb_path, community_model)

        community_medium = MSCommunity.estimate_minimal_community_media(cobra_models, community_model, False, min_growth)
        scores = {}
        for model in cobra_models:
            scores[model.id] = []
            # !!! This excludes cross-feeding that does not completely satisfy the community needs for the respective metabolite
            possible_contributions = [ex_rxn for ex_rxn in FBAHelper.exchange_reactions(model) if ex_rxn.id not in community_medium]
            while len(possible_contributions) > 0:
                community_model.objective = Objective({ex_rxn.flux_expression: 1 for ex_rxn in possible_contributions})
                sol = community_model.optimize()
                blocked = [ex_rxn.id for ex_rxn in possible_contributions if sol.values[ex_rxn.id] < abstol]
                if sol.status != 'optimal' or len(blocked) == len(possible_contributions):
                    break
                for ex_rxn in possible_contributions:
                    if sol.values[ex_rxn.id] >= abstol:
                        for met in ex_rxn:
                            scores[model.id].append(met.id)
                possible_contributions = blocked

            for ex_rxn in possible_contributions:
                community_model.objective = Objective({ex_rxn.flux_expression: 1}) # !!! What is the practical difference of this objective versus the prior objective?
                sol = community_model.optimize()
                score = 1 if sol.status == 'optimal' and sol.fobj > abstol else 0  # TODO the simple binary description of whether a metabolite is contributed by a species may be more concisely determined
                for met in ex_rxn:
                    scores[model.id].append(met.id)
        return scores

    @staticmethod
    def mip_score(cobra_models:Iterable, msdb_path:str, com_model=None, min_growth=0.1):
        """Determine the maximum quantity of nutrients that can be sourced through syntrophy"""
        cobra_models = Smetana._compatibilize_models(cobra_models, msdb_path)
        noninteracting_medium = MSCommunity.estimate_minimal_community_media(cobra_models, com_model, syntrophy=False, min_growth=min_growth)
        # TODO verify that the com_model block does not eradicate the effects of the syntrophy block
        interacting_medium = MSCommunity.estimate_minimal_community_media(cobra_models, com_model, min_growth=min_growth)
        return len(noninteracting_medium) - len(interacting_medium)

    @staticmethod
    def mro_score(cobra_models:Iterable, min_growth=0.1):
        """Determine the maximal overlap of minimal media between member organisms."""
        cobra_models = Smetana._compatibilize_models(cobra_models)
        ind_media = {model.id: set(minimal_medium(model, min_growth, minimize_components=True).index) for model in cobra_models}
        pairs = {(model1, model2): ind_media[model1.id] & ind_media[model2.id] for model1, model2 in combinations(cobra_models, 2)}
        # ratio of the average size of intersecting minimal media between any two members and the minimal media of all members
        return mean(list(map(len, pairs.values()))) / mean(list(map(len, ind_media.values())))

    @staticmethod
    def smetana_score(cobra_models: Iterable, msdb_path:str, community_model=None, min_growth=0.1, n_solutions=100, abstol=1e-6):
        """Quantifies the extent of syntrophy as the sum of all exchanges in a given nutritional environment"""
        community, models = Smetana._load_models(cobra_models, msdb_path, community_model)
        scs = Smetana.sc_score(models, community_model=community, min_growth=min_growth, n_solutions=n_solutions, abstol=abstol)
        mus = Smetana.mu_score(models, community_model=community, min_growth=min_growth)
        mps = Smetana.mp_score(models, community_model=community, abstol=abstol)

        smtna_score = 0
        for model in models:
            for model2 in models:
                if model != model2:
                    if all([mus[model.id], scs[model.id], mps[model.id]]) and all(
                            [model2.id in x for x in [mus[model.id], scs[model.id], mps[model.id]]]):
                        smtna_score += prod([mus[model.id][model2.id], scs[model.id][model2.id], len(mps[model.id])])  # !!! the contribution of MPS must be discerned
        return smtna_score