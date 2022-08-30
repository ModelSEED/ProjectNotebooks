from collections import Counter
from itertools import combinations, chain
from typing import Iterable
from math import prod
from json import load

from cobra.medium import minimal_medium
from modelseedpy.community.mscommunity import MSCommunity
from modelseedpy.community.mscompatibility import MSCompatibility
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Variable, Constraint, Objective
from pprint import pprint
from numpy import mean


class Smetana:
    def __init__(self, cobra_models: Iterable, community_model, min_growth=0.1, n_solutions=100, abstol=1e-3, media_dict=None):
        self.min_growth = min_growth; self.abstol = abstol; self.n_solutions = n_solutions
        self.community, self.models = Smetana._load_models(cobra_models, community_model)
        self.media = media_dict or MSCommunity.minimal_community_media(self.models, self.community, True, min_growth)

    def mro_score(self):
        self.mro = Smetana.mro(self.models, self.min_growth, self.media)
        return self.mro

    def mip_score(self, interacting_media:dict=None, noninteracting_media:dict=None):
        self.mip = Smetana.mip(self.models, self.community, self.min_growth, interacting_media, noninteracting_media)
        return self.mip

    def mu_score(self):
        self.mu = Smetana.mu(self.models, self.min_growth, self.media)
        return self.mu

    def mp_score(self):
        self.mp = Smetana.mp(self.models, self.community, self.min_growth, self.abstol, self.media)
        return self.mp

    def sc_score(self):
        self.sc = Smetana.sc(self.models, self.community, self.min_growth, self.n_solutions, self.abstol)
        return self.sc

    def smetana_score(self):
        if not hasattr(self, "sc"):
            self.sc = Smetana.sc(self.models, self.community, self.min_growth)
        if not hasattr(self, "mu"):
            self.mu = Smetana.mu(self.models, self.media)
        if not hasattr(self, "mp"):
            self.mp = Smetana.mp(self.models, self.community, media_dict=self.media)

        self.smetana = Smetana.smetana(
            self.models, self.community, self.min_growth, self.n_solutions, self.abstol, (self.sc, self.mu, self.mp), self.media)
        return self.smetana

    ###### STATIC METHODS OF THE SMETANA SCORES, WHICH ARE APPLIED IN THE ABOVE CLASS OBJECT ######

    @staticmethod
    def _load_models(cobra_models: Iterable, community_model=None):
        if not community_model:
            return MSCommunity.build_from_species_models(cobra_models, name="SMETANA_example", cobra_model=True), cobra_models
        # models = PARSING_FUNCTION(community_model) # TODO the individual models of a community model must be parsed
        return Smetana._compatibilize_models([community_model])[0], Smetana._compatibilize_models(cobra_models)

    @staticmethod
    def _compatibilize_models(cobra_models:Iterable):
        return cobra_models
        # return MSCompatibility.align_exchanges(cobra_models, True, 'exchanges_conflicts.json')

    @staticmethod
    def _get_media(media, model_s_=None, min_growth=None, com_model=None, syntrophy=True):
        if not media:  # May be either a singular model or a list of models
            return MSCommunity.minimal_community_media(model_s_, com_model, syntrophy, min_growth)
        if not com_model:  # Must be a singular model
            return media["members"][model_s_.id]["media"]
        return media["community_media"]

    @staticmethod
    def sc(cobra_models:Iterable=None, community_model=None, min_growth=0.1, n_solutions=100, abstol=1e-6):
        """Calculate the frequency of interspecies dependency in a community"""
        if not all([cobra_models, community_model]):
            community_model, cobra_models = Smetana._load_models(cobra_models, community_model)
        for rxn in community_model.reactions:
            rxn.lower_bound = 0 if 'bio' in rxn.id else rxn.lower_bound

        # c_{rxn.id}_lb: rxn < 1000*y_{species_id}
        # c_{rxn.id}_ub: rxn > -1000*y_{species_id}
        variables = {}
        constraints = []
        for model in cobra_models:  # TODO this can be converted to an MSCommunity object by looping through each index
            variables[model.id] = Variable(name=f'y_{model.id}', lb=0, ub=1, type='binary')
            FBAHelper.add_cons_vars(community_model, [variables[model.id]])
            for rxn in model.reactions:
                if "bio" not in rxn.id:
                    # print(rxn.flux_expression)
                    lb = Constraint(rxn.forward_variable + 1000*variables[model.id], name=f'c_{rxn.id}_lb', lb=0)
                    ub = Constraint(rxn.forward_variable - 1000*variables[model.id], name=f"c_{rxn.id}_ub", ub=0)
                    constraints.extend([lb, ub])
        # for cons in constraints:
        #     print(type(cons.expression), str(cons.expression))
        #     break
        for cons in community_model.constraints:
            if "EX_Ser_Thr_e0" in str(cons.expression):
                print(cons)
        FBAHelper.add_cons_vars(community_model, list(constraints))  #

        # calculate the SCS
        scores = {}
        for model in cobra_models:
            com_model = community_model.copy()
            other_members = [other for other in cobra_models if other.id != model.id]
            # SMETANA_Biomass: bio1 > {min_growth}
            smetana_biomass = Constraint(sum(rxn for rxn in model.reactions if "bio" in rxn.id), name='SMETANA_Biomass', lb=min_growth)
            FBAHelper.add_cons_vars(com_model, [smetana_biomass])
            FBAHelper.add_objective(com_model, {f"y_{other.id}": 1.0 for other in other_members}, "min")
            previous_constraints, donors_list = [], []
            for i in range(n_solutions):
                sol = com_model.optimize()
                if sol.status != 'optimal':
                    scores[model.id] = None
                    break
                donors = [o for o in other_members if sol.values[f"y_{o.id}"] > abstol]
                donors_list.append(donors)

                # this solution is removed from the search space
                # c_{rxn.id}_lb: sum(y_{species_id}) < # iterations - 1
                previous_con = f'iteration_{i}'
                previous_constraints.append(previous_con)
                FBAHelper.add_cons_vars(com_model, list(Constraint(
                    sum(variables[o.id] for o in donors), name=previous_con, ub=len(previous_constraints) - 1)))

            # calculate the score if the loop completed without error
            if i == n_solutions-1:
                donors_counter = Counter(chain(*donors_list))
                scores[model.id] = {o: donors_counter[o] / len(donors_list) for o in other_members}
        return scores

    @staticmethod
    def mu(cobra_models:Iterable, min_growth=0.1, media_dict=None):
        """the fractional frequency of each received metabolite amongst all possible alternative syntrophic solutions"""
        # TODO: support for parsing a community model should be added to mirror Machado's functionality
        scores = {}
        cobra_models = Smetana._compatibilize_models(cobra_models)
        for model in cobra_models:
            ex_rxns = {ex_rxn.id: met for ex_rxn in FBAHelper.exchange_reactions(model) for met in ex_rxn.metabolites}
            min_media = Smetana._get_media(media_dict, model, min_growth, syntrophy=True)
            counter = Counter(min_media)
            scores[model.id] = {met: counter[ex] / len(min_media) for ex, met in ex_rxns.items() if counter[ex] > 0}
        return scores

    @staticmethod
    def mp(cobra_models:Iterable=None, community_model=None, min_growth=0.1, abstol=1e-3, media_dict=None):
        """Discover the metabolites that each species contributes to a community"""
        if not all([community_model, cobra_models]):
            community_model, cobra_models = Smetana._load_models(cobra_models, community_model)

        noninteracting_community_medium = Smetana._get_media(media_dict, cobra_models, min_growth, community_model, False)
        scores = {}
        for model in cobra_models:
            scores[model.id] = []
            # !!! This approach excludes cross-feeding that incompletely satisfies community needs
            possible_contributions = [ex_rxn for ex_rxn in FBAHelper.exchange_reactions(model) if ex_rxn.id not in noninteracting_community_medium]
            while len(possible_contributions) > 0:
                print("remaining possible_contributions", len(possible_contributions), end="\r")
                community_model.objective = Objective(sum(ex_rxn.flux_expression for ex_rxn in possible_contributions))
                sol = community_model.optimize()
                # pprint([ex for ex in sol.fluxes.keys() if "EX_" in ex and sol.fluxes[ex] > 0])
                # TODO - affirm that missing exchange reactions of individual in the community fluxes is acceptable
                fluxes_contributions = [ex for ex in possible_contributions if ex.id in sol.fluxes.keys()]
                blocked = [ex for ex in fluxes_contributions if sol.fluxes[ex.id] < abstol]
                if sol.status != 'optimal' or len(blocked) == len(possible_contributions):
                    break
                # log confirmed contributions, where the fluxes exceed an absolute tolerance
                for ex_rxn in fluxes_contributions:
                    if sol.fluxes[ex_rxn.id] >= abstol:
                        for met in ex_rxn.metabolites:
                            scores[model.id].append(met.id)
                possible_contributions = blocked

            # TODO - evaluate the necessity of this block considering that the ultimate removal is never triggered
            for ex_rxn in possible_contributions:  # This assumes that possible_contributions never reaches zero
                community_model.objective = Objective(ex_rxn.flux_expression)
                sol = community_model.optimize()
                for met in ex_rxn.metabolites:
                    if sol.status != 'optimal' or sol.objective_value < abstol and met.id in scores[model.id]:
                        print("removing", met.id)
                        scores[model.id].remove(met.id)
        return scores

    @staticmethod
    def mip(cobra_models:Iterable, com_model=None, min_growth=0.1, interacting_media_dict=None, noninteracting_media_dict=None):
        """Determine the maximum quantity of nutrients that can be sourced through syntrophy"""
        cobra_models = Smetana._compatibilize_models(cobra_models)
        if noninteracting_media_dict:
            noninteracting_medium = noninteracting_media_dict["community_media"]
        else:
            noninteracting_medium = Smetana._get_media(noninteracting_media_dict, cobra_models, min_growth, com_model, False)["community_media"]
        if interacting_media_dict:
            interacting_medium = interacting_media_dict["community_media"]
        else:
            interacting_medium = Smetana._get_media(interacting_media_dict, cobra_models, min_growth, com_model, False)["community_media"]
        print(f"non-interacting keys", list(noninteracting_medium.keys()))
        pprint(noninteracting_medium)
        pprint(interacting_medium)
        return len(noninteracting_medium) - len(interacting_medium)

    @staticmethod
    def mro(cobra_models:Iterable, min_growth=0.1, media_dict=None):
        """Determine the maximal overlap of minimal media between member organisms."""
        cobra_models = Smetana._compatibilize_models(cobra_models)
        ind_media = {model.id: set(Smetana._get_media(media_dict, model, min_growth)) for model in cobra_models}
        pairs = {(model1, model2): ind_media[model1.id] & ind_media[model2.id] for model1, model2 in combinations(cobra_models, 2)}
        # ratio of the average size of intersecting minimal media between any two members and the minimal media of all members
        return mean(list(map(len, pairs.values()))) / mean(list(map(len, ind_media.values())))

    @staticmethod
    def smetana(cobra_models: Iterable, community=None, min_growth=0.1, n_solutions=100, abstol=1e-6, prior_values=None, media_dict=None):
        """Quantifies the extent of syntrophy as the sum of all exchanges in a given nutritional environment"""
        if not all([community, cobra_models]):
            community, cobra_models = Smetana._load_models(cobra_models, community)
        if not prior_values:
            sc = Smetana.sc(cobra_models, community, min_growth, n_solutions, abstol)
            mu = Smetana.mu(cobra_models, min_growth, media_dict)
            mp = Smetana.mp(cobra_models, community, min_growth, abstol, media_dict)
        else:
            sc, mu, mp = prior_values

        smtna_score = 0
        for model in cobra_models:
            for model2 in cobra_models:
                if model != model2:
                    if all([mu[model.id], sc[model.id], mp[model.id]]) and all(
                            [model2.id in x for x in [mu[model.id], sc[model.id], mp[model.id]]]):
                        smtna_score += prod([mu[model.id][model2.id], sc[model.id][model2.id], len(mp[model.id])])
        return smtna_score