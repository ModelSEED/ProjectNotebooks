from collections import Counter
from itertools import combinations, chain
from typing import Union
from math import inf

from cobra.medium import minimal_medium
from modelseedpy.community.mscommunity import MSCommunity
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Variable, Constraint, Objective
from time import process_time
from cobra import Reaction


class Smetana:

    def __init__(self, cobra_models: Union[list, tuple, set], msdb:str, community_model=None):
        # convert COBRA model into ReFramed model
        self.models = cobra_models  # TODO the individual models of a community model must be parsed
        if not community_model:
            self.community, self.biomass_indicies = MSCommunity.build_from_species_models(
                cobra_models, msdb, name="SMETANA_example", cobra_model=True)  # abundances argument may be valuable
        else:
            self.community = community_model
            self.biomass_indicies = {model.id: [rxn.id for rxn in model.reactions if "bio" in rxn.id] for model in self.models}
        # self.community = FBAHelper.update_model_media(self.community, media)  !!! adding a media causes the objective value to be zero

    def sc_score(self, min_growth=0.1, n_solutions=100, abstol=1e-6):
        """Calculate the frequency of interspecies dependency in a community"""
        # community = community.copy(copy_models=False, interacting=True, create_biomass=False, merge_extracellular_compartments=False)

        # identify the biomass compounds of each respective community model
        for rxn in self.community.reactions:
            rxn.lower_bound = 0 if 'bio' in rxn.id else rxn.lower_bound

        # reaction flux bounds, with consideration of each species
        # c_{rxn.id}_lb: rxn < 1000*y_{species_id}
        # c_{rxn.id}_ub: rxn > -1000*y_{species_id}
        variables = {}
        constraints = []
        for model in self.models:
            variables[model.id] = Variable(name=f'y_{model.id}', lb=0, ub=1, type='binary')
            for rxn in model.reactions:
                if rxn.id != self.biomass_indicies[model.id]:
                    lb = Constraint(rxn.flux_expression - 1000 * variables[model.id], name=f'c_{rxn.id}_lb', ub=0)
                    ub = Constraint(rxn.flux_expression + 1000 * variables[model.id], name=f"c_{rxn.id}_ub", lb=0)
                    constraints.extend([lb, ub])
        self.community = FBAHelper.add_vars_cons(self.community, list(variables.values()) + constraints)

        # calculate the SCS
        scores = {}
        com_model = self.community  # .copy()
        for model in self.models:
            other_members = [other for other in self.models if other.id != model.id]
            # SMETANA_Biomass: bio1 > {min_growth}
            smetana_biomass = Constraint(model.reactions.bio1, name='SMETANA_Biomass', lb=min_growth)
            com_model = FBAHelper.add_vars_cons(com_model, [smetana_biomass])
            com_model.problem.objective = Objective({f"y_{other.id}": 1.0 for other in other_members}, direction="min")
            com_model.solver.update()

            previous_constraints, donors_list = [], []
            failed = False
            for i in range(n_solutions):
                sol = com_model.optimize()
                if sol.status != 'optimal':
                    failed = i == 0
                    break

                donors = [o for o in other_members if sol.values[f"y_{o.id}"] > abstol]
                donors_list.append(donors)

                # the community is iteratively reduced
                # c_{rxn.id}_lb: sum(y_{species_id}) < # iterations - 1
                previous_con = f'iteration_{i}'
                previous_constraints.append(previous_con)
                com_model = FBAHelper.add_vars_cons(com_model, list(Constraint(
                    sum(variables[o.id] for o in donors), name=previous_con, ub=len(previous_constraints) - 1)))

            com_model.remove_cons_vars([smetana_biomass] + previous_constraints)
            scores[model.id] = None
            if not failed:
                donors_list_n = float(len(donors_list))
                donors_counter = Counter(chain(*donors_list))
                scores[model.id] = {o: donors_counter[o] / donors_list_n for o in other_members}
        return scores

    def mu_score(self, min_growth=None):
        """Calculate the quantity of metabolic requirements for species growth"""
        scores = {}
        for model in self.models:
            time1 = process_time()
            # change the community biomass reaction of that of the individual species
            for rxn in model.reactions:
                if "bio" in rxn.id:
                    self.community.reactions.bio1 = rxn
                    break
            time2 = process_time()
            print(f"done defining the biomass reaction: {(time2-time1)/60}")
            ex_rxns = {ex_rxn.id.removeprefix("EX_"): met for ex_rxn in FBAHelper.exchange_reactions(model) for met in ex_rxn.metabolites}
            min_growth = min_growth #or model.optimize().objective_value
            minimal_media = minimal_medium(self.community, min_growth, minimize_components=True)
            time3 = process_time()
            print(f"done processing the minimal media for {model.id}: {(time3 - time2) / 60}")
            counter = Counter(minimal_media)
            scores[model.id] = {met: counter[ex] / len(minimal_media) for ex, met in ex_rxns.items()}
        return scores

    def mp_score(self, abstol=1e-3):
        """Discover the metabolites that each species contributes to a community"""
        scores = {}
        for model in self.models:
            scores[model.id] = {}
            ex_rxns = {ex_rxn.id.removeprefix("EX_"): met for ex_rxn in FBAHelper.exchange_reactions(model) for met in ex_rxn.metabolites}
            remaining = [ex_rxn for ex_rxn, met in ex_rxns.items() if met.id not in self.media_cpds]  # TODO the media needs to be defined and applied to the community
            while len(remaining) > 0:
                self.community.objective = Objective({ex_rxn.flux_expression: 1 for ex_rxn in remaining})
                sol = self.community.optimize()
                blocked = [ex_rxn.id for ex_rxn in remaining if sol.values[ex_rxn.id] < abstol]
                if sol.status != 'optimal' or len(blocked) == len(remaining):
                    break
                for ex_rxn in remaining:
                    if sol.values[ex_rxn.id] >= abstol:
                        for met in ex_rxn:
                            scores[model.id][met.id] = 1
                remaining = blocked

            for ex_rxn in remaining:
                self.community.objective = Objective({ex_rxn.flux_expression: 1 for ex_rxn in remaining})
                sol = self.community.optimize()
                score = 1 if sol.status == 'optimal' and sol.fobj > abstol else 0
                for met in ex_rxn:
                    scores[model.id][met.id] = score
        return scores

    def mip_score(self, min_growth=None):
        """Determine the maximum quantity of nutrients that are sourced through syntrophy"""
        with self.community as com_model:
            interacting = com_model
            noninteracting = FBAHelper.non_interacting_community(com_model)

        # an abiotic environment is limited to non-interacting community minimal media
        min_growth = min_growth #or self.community.optimize().objective_value
        noninteracting_medium = minimal_medium(noninteracting, min_growth, minimize_components=True)
        interacting.medium = noninteracting_medium  # TODO apply the noninteracting_medium as the exchanges of the interacting model
        interacting_medium = minimal_medium(interacting, min_growth, minimize_components=True)
        return len(noninteracting_medium) - len(interacting_medium)

    def mro_score(self, new_media=None, min_growth:float=None, exclude:set=None):
        """Determine the maximal overlap of minimal media between member organisms."""
        min_growth = min_growth #or self.community.optimize().objective_value
        minimal_media = minimal_medium(self.community, min_growth, minimize_components=True)
        self.community.medium = minimal_media  # TODO apply the minimal_media as the exchanges of the interacting model

        exclude = exclude or set()
        individual_media = {}
        for model in self.models:
            for rxn in model.reactions:
                if 'bio' in rxn.id:
                    self.community.reactions.bio1 = rxn  # TODO assign the community biomass reaction
                    break
            species_exchange = {met for met in rxn.metabolites.keys() for rxn in FBAHelper.exchange_reactions(model)}
            # TODO apply species_exchange to the exchanges of the community model
            model_minimal_medium = minimal_medium(self.community, min_growth, minimize_components=True)
            individual_media[model.id] = {substrate for substrate in model_minimal_medium if substrate in species_exchange} - exclude

        pairwise = {(model1, model2): individual_media[model1.id] & individual_media[model2.id] for model1, model2 in combinations(self.models, 2)}
        return (sum(map(len, pairwise.values())) / len(pairwise)) / (sum(map(len, individual_media.values())) / len(individual_media))

    def smetana_score(self, min_growth=0.1, n_solutions=100, abstol=1e-6):
        """Quantifies the extend of syntrophy as the sum of all exchanges in a given nutritional environment"""
        scs = self.sc_score(min_growth, n_solutions, abstol)
        mus = self.mu_score(min_growth, abstol, n_solutions)
        mps = self.mp_score(abstol)

        smtna_score = 0
        for index, model in enumerate(self.models):
            other_models = self.models.pop(index)
            for model2 in other_models:
                if all([mus[model.id], scs[model.id], mps[model.id]]) and all(
                        [model2.id in x for x in [mus[model.id], scs[model.id], mps[model.id]]]):
                    smtna_score += mus[model.id][model2.id] * scs[model.id][model2.id] * mps[model.id][model2.id]
        return smtna_score