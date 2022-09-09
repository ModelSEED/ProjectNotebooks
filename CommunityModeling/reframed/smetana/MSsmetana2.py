from collections import Counter
from itertools import combinations, permutations, chain
from typing import Iterable
from math import prod
from json import load

from cobra.medium import minimal_medium
from modelseedpy.community.mscommunity import MSCommunity
from modelseedpy.core.minimalmediapkg import MinimalMediaPkg
from modelseedpy.community.mscompatibility import MSCompatibility
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Variable, Constraint, Objective
from optlang.symbolics import Zero
from deepdiff import DeepDiff  # (old, new)
from pprint import pprint
from numpy import mean


class Smetana:
    def __init__(self, cobra_models: Iterable, community_model, min_growth=0.1,
                 n_solutions=100, abstol=1e-3, media_dict=None, printing=True, compatibilize=True):

        self.min_growth = min_growth ; self.abstol = abstol ; self.n_solutions = n_solutions
        self.printing = printing ; self.compatibilize = compatibilize

        self.community, self.models = Smetana._load_models(cobra_models, community_model, compatibilize)
        self.media = media_dict or MSCommunity.minimal_community_media(self.models, self.community, True, min_growth)

    def mro_score(self):
        self.mro = Smetana.mro(self.models, self.min_growth, self.media, self.compatibilize)
        if self.printing:
            print(f"MRO score:\t\t\t{self.mro*100:.2f}% of member minimal media, on average, overlap with other member minimal media.")
        return self.mro

    def mip_score(self, interacting_media:dict=None, noninteracting_media:dict=None):
        self.mip = Smetana.mip(self.models, self.community, self.min_growth,
                               interacting_media, noninteracting_media, compatibilize=self.compatibilize)
        if self.printing:
            print(f"MIP score:\t\t\t{self.mip} required community compounds are sourced via syntrophy in the community.")
        return self.mip

    def mu_score(self):
        self.mu = Smetana.mu(self.models, self.n_solutions, self.abstol, compatibilize=self.compatibilize)
        if self.printing:
            print("MU score:\t\t\tThe fraction of solutions in which each member is the syntrophic receiver that contain a respective metabolite.\n")
            pprint(self.mu)
        return self.mu

    def mp_score(self):
        self.mp = Smetana.mp(self.models, self.community, self.abstol, compatibilize=self.compatibilize)
        if self.printing:
            print("MP score:\t\t\tThe possible contributions of each member in the member media include:\n")
            pprint(self.mp)
        return self.mp

    def sc_score(self):
        self.sc = Smetana.sc(self.models, self.community, self.min_growth, self.n_solutions, self.abstol, compatibilize=self.compatibilize)
        if self.printing:
            print("SC score:\t\t\tThe fraction of community members who syntrophically contribute to each species:\n")
            pprint(self.sc)
        return self.sc

    def smetana_score(self):
        if not hasattr(self, "sc"):
            self.sc = self.sc_score()
            sc_coupling = self.sc is not None
        if not hasattr(self, "mu"):
            self.mu = self.mu_score()
        if not hasattr(self, "mp"):
            self.mp = self.mp_score()

        self.smetana = Smetana.smetana(
            self.models, self.community, self.min_growth, self.n_solutions, self.abstol, (self.sc, self.mu, self.mp), self.media)
        return self.smetana

    ###### STATIC METHODS OF THE SMETANA SCORES, WHICH ARE APPLIED IN THE ABOVE CLASS OBJECT ######

    @staticmethod
    def _load_models(cobra_models: Iterable, community_model=None, compatibilize=True):
        if not community_model and cobra_models:
            return MSCommunity.build_from_species_models(cobra_models, name="SMETANA_example", cobra_model=True), cobra_models
        # models = PARSING_FUNCTION(community_model) # TODO the individual models of a community model must be parsed
        if compatibilize:
            return Smetana._compatibilize_models([community_model])[0], Smetana._compatibilize_models(cobra_models)
        return community_model, cobra_models

    @staticmethod
    def _compatibilize_models(cobra_models:Iterable, printing=False):
        return cobra_models
        # return MSCompatibility.standardize(cobra_models, conflicts_file_name='exchanges_conflicts.json', printing=printing)

    @staticmethod
    def _get_media(media, model_s_=None, min_growth=None, com_model=None, syntrophy=True):
        if not media:  # May be either a singular model or a list of models
            return MSCommunity.minimal_community_media(model_s_, com_model, syntrophy, min_growth)
        if not com_model:  # Must be a singular model
            return media["members"][model_s_.id]["media"]
        return media["community_media"]


    @staticmethod
    def mro(cobra_models:Iterable, min_growth=0.1, media_dict=None, compatibilize=True):
        """Determine the maximal overlap of minimal media between member organisms."""
        if compatibilize:
            cobra_models = Smetana._compatibilize_models(cobra_models)
        ind_media = {model.id: set(Smetana._get_media(media_dict, model, min_growth)) for model in cobra_models}
        pairs = {(model1, model2): ind_media[model1.id] & ind_media[model2.id] for model1, model2 in combinations(cobra_models, 2)}
        # ratio of the average size of intersecting minimal media between any two members and the minimal media of all members
        return mean(list(map(len, pairs.values()))) / mean(list(map(len, ind_media.values())))

    @staticmethod
    def mip(cobra_models:Iterable, com_model=None, min_growth=0.1, interacting_media_dict=None,
            noninteracting_media_dict=None, compatibilize=True):
        """Determine the maximum quantity of nutrients that can be sourced through syntrophy"""
        if noninteracting_media_dict and interacting_media_dict:
            noninteracting_medium = Smetana._get_media(noninteracting_media_dict, cobra_models, min_growth, com_model, False)
            interacting_medium = Smetana._get_media(interacting_media_dict, cobra_models, min_growth, com_model, True)
            return len(noninteracting_medium) - len(interacting_medium)

        if compatibilize:
            cobra_models = Smetana._compatibilize_models(cobra_models)

        if noninteracting_media_dict:
            noninteracting_medium = noninteracting_media_dict["community_media"]
        else:
            noninteracting_medium = Smetana._get_media(None, cobra_models, min_growth, com_model, False)["community_media"]

        if interacting_media_dict:
            interacting_medium = interacting_media_dict["community_media"]
        else:
            interacting_medium = Smetana._get_media(None, cobra_models, min_growth, com_model, False)["community_media"]
        print("Difference between the non-interacting and interacting media.")
        pprint(DeepDiff(noninteracting_medium, interacting_medium))
        return len(noninteracting_medium) - len(interacting_medium)

    @staticmethod
    def mu(cobra_models:Iterable, n_solutions=100, abstol=1e-3, compatibilize=True):
        """the fractional frequency of each received metabolite amongst all possible alternative syntrophic solutions"""
        # determine the solutions for each member
        # member_solutions = member_solutions if member_solutions else {model.id: model.optimize() for model in cobra_models}
        scores = {}
        if compatibilize:
            cobra_models = Smetana._compatibilize_models(cobra_models)
        for model in cobra_models:
            ex_rxns = {ex_rxn: met for ex_rxn in FBAHelper.exchange_reactions(model) for met in ex_rxn.metabolites}
            variables = {ex_rxn.id: Variable('___'.join([model.id, ex_rxn.id]), lb=0, ub=1, type="binary") for ex_rxn in ex_rxns}
            FBAHelper.add_cons_vars(model, [list(variables.values())])
            media, solutions = [], []
            for i in range(0, n_solutions):
                if i > 0:
                    constraint = Constraint(sum([variables[ex.id] for ex in medium]), ub=len(medium)-1, name=f"iteration_{i}")
                    FBAHelper.add_cons_vars(model, [constraint])
                sol = model.optimize()
                if sol.status != 'optimal':
                    break
                # determine the de facto medium for this simulated growth
                solutions.append(sol)
                medium = set([ex_rxn for ex_rxn in ex_rxns if sol.fluxes[ex_rxn.id] < -abstol])
                media.append(medium)
            counter = Counter(chain(*media))
            scores[model.id] = {met.id: counter[ex] / len(media) for ex, met in ex_rxns.items() if counter[ex] > 0}
        return scores

    @staticmethod
    def mp(cobra_models:Iterable=None, community_model=None, abstol=1e-3, compatibilize=True):
        """Discover the metabolites that each species can contribute to a community"""
        community_model, cobra_models = Smetana._load_models(cobra_models, community_model, compatibilize)
        scores = {}
        for model in cobra_models:
            scores[model.id] = []
            # determines possible member contributions in the community environment, where the excretion of media compounds is irrelevant
            approximate_minimal_media = MinimalMediaPkg.minimize_flux(community_model)
            possible_contributions = [ex_rxn for ex_rxn in FBAHelper.exchange_reactions(model) if ex_rxn.id not in approximate_minimal_media]
            while len(possible_contributions) > 0:
                print("remaining possible_contributions", len(possible_contributions), end="\r")
                FBAHelper.add_objective(community_model, sum(ex_rxn.flux_expression for ex_rxn in possible_contributions))
                sol = community_model.optimize()
                fluxes_contributions, uncertain_contributions = [], []
                for ex in possible_contributions:
                    if ex.id in sol.fluxes.keys():
                        if sol.fluxes[ex.id] >= abstol:
                            fluxes_contributions.append(ex)
                            possible_contributions.remove(ex)

                if sol.status != 'optimal' or not fluxes_contributions:
                    break
                # log confirmed contributions
                for ex_rxn in fluxes_contributions:
                    for met in ex_rxn.metabolites:
                        scores[model.id].append(met.id)

            # double-check the remaining possible contributions for excretion
            for ex_rxn in possible_contributions:
                community_model.objective = Objective(ex_rxn.flux_expression)
                sol = community_model.optimize()
                if sol.status != 'optimal' or sol.objective_value < abstol:
                    for met in ex_rxn.metabolites:
                        if met.id in scores[model.id]:
                            print("removing", met.id)
                            scores[model.id].remove(met.id)
        return scores

    @staticmethod
    def sc(cobra_models:Iterable=None, community_model=None, min_growth=0.1, n_solutions=100, abstol=1e-6, compatibilize=True):
        """Calculate the frequency of interspecies dependency in a community"""
        community_model, cobra_models = Smetana._load_models(cobra_models, community_model, compatibilize)
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
                    lb = Constraint(rxn.flux_expression + 1000*variables[model.id], name="_".join(["c", model.id, rxn.id, "lb"]), lb=0)
                    ub = Constraint(rxn.flux_expression - 1000*variables[model.id], name="_".join(["c", model.id, rxn.id, "ub"]), ub=0)
                    constraints.extend([lb, ub])
        FBAHelper.add_cons_vars(community_model, constraints, sloppy=True)

        # calculate the SCS
        scores = {}
        for model in cobra_models:
            com_model = community_model.copy()
            other_members = [other for other in cobra_models if other.id != model.id]
            # model growth is guaranteed while minimizing the growing members of the community
            ## SMETANA_Biomass: {biomass_reactions} > {min_growth}
            smetana_biomass = Constraint(sum(rxn.flux_expression for rxn in model.reactions if "bio" in rxn.id),
                                         name='SMETANA_Biomass', lb=min_growth)
            FBAHelper.add_cons_vars(com_model, [smetana_biomass], sloppy=True)
            FBAHelper.add_objective(com_model, sum([variables[other.id] for other in other_members]), "min")
            previous_constraints, donors_list = [], []
            for i in range(n_solutions):
                sol = com_model.optimize()  # FIXME The solution is not optimal
                if sol.status != 'optimal':
                    scores[model.id] = None
                    break
                donors = [o for o in other_members if sol.values[f"y_{o.id}"] > abstol]
                donors_list.append(donors)
                previous_con = f'iteration_{i}'
                previous_constraints.append(previous_con)
                FBAHelper.add_cons_vars(com_model, list(Constraint(
                    sum(variables[o.id] for o in donors), name=previous_con, ub=len(previous_constraints)-1)), sloppy=True)
            if i != 0:
                donors_counter = Counter(chain(*donors_list))
                scores[model.id] = {o.id: donors_counter[o] / len(donors_list) for o in other_members}
        return scores

    @staticmethod
    def smetana(cobra_models: Iterable, community=None, min_growth=0.1, n_solutions=100, abstol=1e-6,
                prior_values=None, compatibilize=False, sc_coupling=False):
        """Quantifies the extent of syntrophy as the sum of all exchanges in a given nutritional environment"""
        community, cobra_models = Smetana._load_models(cobra_models, community, compatibilize)
        sc = None
        if not prior_values:
            mu = Smetana.mu(cobra_models, n_solutions, abstol, compatibilize)
            mp = Smetana.mp(cobra_models, community, abstol, compatibilize)
            if sc_coupling:
                sc = Smetana.sc(cobra_models, community, min_growth, n_solutions, abstol, compatibilize)
        else:
            if sc_coupling:
                sc, mu, mp = prior_values
            else:
                mu, mp = prior_values

        smetana_scores = {}
        for pairs in combinations(cobra_models, 2):
            for model1, model2 in permutations(pairs):
                if model1.id not in smetana_scores:
                    smetana_scores[model1.id] = {}
                if not any([not sc_coupling and sc, not mu[model1.id], not mp[model1.id]]):
                    sc_score = 1 if not sc_coupling else sc[model1.id][model2.id]
                    models_mets = list(model1.metabolites)+list(model2.metabolites)
                    unique_mets = set([met.id for met in models_mets])
                    smetana_scores[model1.id][model2.id] = 0
                    for met in models_mets:
                        if met.id in unique_mets:
                            mp_score = 0 if met.id not in mp[model1.id] else 1
                            smetana_scores[model1.id][model2.id] += prod([mu[model1.id].get(met.id,0), sc_score, mp_score])
        return smetana_scores