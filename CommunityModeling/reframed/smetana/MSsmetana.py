from collections import Counter
from itertools import combinations, chain
from typing import Union
from math import inf
from warnings import warn

from modelseedpy.community.mscommunity import MSCommunity
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Variable, Constraint, Objective
from cobra import Reaction


class Smetana:

    def __init__(self, cobra_models: Union[list, tuple, set], media):
        # convert COBRA model into ReFramed model
        self.models = cobra_models
        self.community, self.biomass_indicies = MSCommunity.build_from_species_models(cobra_models, name="SMETANA_example", cobra_model=True)  # abundances argument may be valuable
        self.community = FBAHelper.update_model_media(self.community, media)

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
                    lb = Constraint(rxn.flux_expression - 1000*variables[model.id], name=f'c_{rxn.id}_lb', ub=0)
                    ub = Constraint(rxn.flux_expression + 1000*variables[model.id], name=f"c_{rxn.id}_ub", lb=0)
                    constraints.extend([lb, ub])
        self.community = FBAHelper.add_vars_cons(self.community, list(variables.values())+constraints)

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
                if sol.status.value != 'optimal':
                    failed = i == 0
                    break

                donors = [o for o in other_members if sol.values[f"y_{o.id}"] > abstol]
                donors_list.append(donors)

                # the community is iteratively reduced
                # c_{rxn.id}_lb: sum(y_{species_id}) < # iterations - 1
                previous_con = f'iteration_{i}'
                previous_constraints.append(previous_con)
                com_model = FBAHelper.add_vars_cons(com_model, list(Constraint(
                    sum(variables[o.id] for o in donors), name=previous_con, ub=len(previous_constraints)-1)))

            com_model.remove_cons_vars([smetana_biomass]+previous_constraints)
            if not failed:
                donors_list_n = float(len(donors_list))
                donors_counter = Counter(chain(*donors_list))
                scores[model.id] = {o: donors_counter[o] / donors_list_n for o in other_members}
            else:
                warn('SCS: Failed to find a solution for growth of ' + model.id)
                scores[model.id] = None
        return scores

    def mu_score(self, min_growth=0.1, max_uptake=10.0, abstol=1e-6, n_solutions=100):
        """Calculate the quantity of metabolic requirements for species growth"""
        max_uptake *= len(self.models)
        scores = {}
        for model in self.models:
            # change the community biomass reaction of that of the individual species
            self.community.reactions.bio1 = model.reactions.bio1

            ex_rxns = {ex_rxn.id:met for ex_rxn in model.exchanges for met in ex_rxn.metabolites}
            medium_list, sols = Smetana.minimal_medium(self.community, exchange_reactions=ex_rxns, min_growth=min_growth,
                n_solutions=n_solutions, max_uptake=max_uptake, validate=validate, abstol=abstol, warnings=False)

            if medium_list:
                counter = Counter(chain(*medium_list))
                scores[model.id] = {met: counter[ex] / len(medium_list) for ex, met in ex_rxns.items()}
            else:
                warn('MUS: Failed to find a minimal growth medium for ' + model.id)
                scores[model.id] = None

        return scores

    def mp_score(self, abstol=1e-3):
        scores = {}
        for model in self.models:
            scores[model.id] = {}
            remaining = [ex_rxn for ex_rxn in model.exchanges if not any([met.id in self.media_cpds for met in ex_rxn])]
            while len(remaining) > 0:
                self.community.objective = Objective({ex_rxn.flux_expression: 1 for ex_rxn in remaining})
                sol = self.community.optimize()
                if sol.status != 'optimal':
                    break
                blocked = [ex_rxn.id for ex_rxn in remaining if sol.values[ex_rxn.id] < abstol]
                if len(blocked) == len(remaining):
                    break

                for ex_rxn in remaining:
                    if sol.values[ex_rxn.id] >= abstol:
                        for met in ex_rxn:  # !!! only one metabolite in the exchange reaction; hence, a simplification should be explored
                            scores[model.id][met.id] = 1

                remaining = blocked

            for ex_rxn in remaining:
                self.community.objective = Objective({ex_rxn.flux_expression: 1 for ex_rxn in remaining})
                sol = self.community.optimize()

                score = 0
                if sol.status == 'optimal' and sol.fobj > abstol:
                    scores = 1
                for met in ex_rxn:
                    scores[model.id][met.id] = 1

        return scores

    def mip_score(self, community=None, environment=None, min_growth=0.1, direction=-1,
                  max_uptake=10, validate=False, use_lp=False, exclude=None):
        noninteracting = self.non_interacting(self.community.copy())
        max_uptake *= len(community.organisms)

        noninteracting_medium, sol1 = Smetana.minimal_medium(
            noninteracting, exchange_reactions=noninteracting.exchanges, direction=direction,
            min_growth=min_growth, max_uptake=max_uptake, validate=validate, warnings=False, milp=(not use_lp))
        if noninteracting_medium is None:
            warn('MIP: Failed to find a valid solution for non-interacting community')
            return None

        # anabiotic environment is limited to non-interacting community minimal media
        interacting_medium, sol2 = Smetana.minimal_medium(
            self.community, direction=direction, exchange_reactions=noninteracting_medium,
            min_growth=min_growth, milp=(not use_lp), max_uptake=max_uptake, validate=validate, warnings=False)
        if interacting_medium is None:
            warn('MIP: Failed to find a valid solution for interacting community')
            return None

        if exclude is not None:
            exclude_rxns = {'R_EX_M_{}_e_pool'.format(x) for x in exclude}
            interacting_medium = set(interacting_medium) - exclude_rxns
            noninteracting_medium = set(noninteracting_medium) - exclude_rxns

        return len(noninteracting_medium) - len(interacting_medium)

    def non_interacting(self, community):
        # !!! divert all exchange reactions to a sink
        for rxn in community.reactions:
            if "EX_" in rxn.id:
                community.add_boundary(list(rxn.metabolites.keys())[0], lb=0, type="sink")
        return community


    def mro_score(self, community=None, new_media=None, direction=-1, min_growth=0.1,
                  max_uptake=10, validate=False, use_lp=False, exclude=None):
        exch_reactions = self.community.exchanges
        max_uptake *= len(community.organisms)
        medium, sol = Smetana.minimal_medium(
            self.community, exchange_reactions=exch_reactions, direction=direction,
            min_growth=min_growth, max_uptake=max_uptake, validate=validate, warnings=False, milp=(not use_lp))

        if sol.status != 'optimal':
            warn('MRO: Failed to find a valid solution for community')
            return None

        new_media   #!!! apply a new media to the model

        exclude = exclude or set()
        medium -= exclude
        individual_media = {}
        for model in self.models:
            for rxn in model.reactions:
                if 'bio' in rxn.id:
                    community.merged.biomass_reaction = rxn  # !!! assign the community biomass reaction
                    break
            species_exchanges = model.exchanges
            species_exchanged_substrates = {met for met in rxn.metabolites.keys() for rxn in model.exchanges}

            medium_i, sol = Smetana.minimal_medium(self.community, direction=direction,
                min_growth=min_growth, max_uptake=max_uptake, validate=validate, milp=(not use_lp))

            if sol.status != 'optimal':
                warn('MRO: Failed to find a valid solution for: ' + model.id)
                return None

            individual_media[model.id] = {substrate for substrate in medium_i if substrate in species_exchanged_substrates} - exclude

        pairwise = {(model1, model2): individual_media[model1.id] & individual_media[model2.id] for model1, model2 in combinations(self.model, 2)}
        return (sum(map(len, pairwise.values())) / len(pairwise)) / (sum(map(len, individual_media.values())) / len(individual_media))

    @staticmethod
    def minimal_medium(model, direction=-1, min_growth=1, max_uptake=100, max_compounds=None, n_solutions=1, validate=True, milp=True):
        def get_medium(solution, exchange_reactions, direction):
            return set(ex_rxn.id for ex_rxn in exchange_reactions
                       if (direction < 0 and solution.fluxes[ex_rxn.id] < 0 or direction > 0 and solution.fluxes[ex_rxn.id] > 0))

        def validate_solution(model, medium, direction, min_growth, max_uptake):
            for ex_rxn in model.exchanges:
                if direction == -1:
                    ex_rxn.bounds = (min_growth, inf) if ex_rxn.id in medium else (0, inf)
                else:
                    ex_rxn.bounds = (-inf, max_uptake) if ex_rxn.id in medium else (-inf, 0)

            sol = model.optimize()
            if sol.objective_value < min_growth:
                warn(f'The objective value of {sol.objective_value} is less than the minimal growth {min_growth}.')

        exchange_reactions = [rxn for rxn in model.reactions if "EX_" in rxn.id]

        if not milp and max_compounds is not None:
            raise RuntimeError("max_compounds can only be used with MILP formulation")
        if not milp and n_solutions > 1:
            raise RuntimeError("n_solutions can only be used with MILP formulation")

        variables = {}
        constraints = []
        if milp:
            for ex_rxn in exchange_reactions:
                variables['y_' + ex_rxn.id] = Variable(name=f'y_{ex_rxn.id}', lb=0, ub=1, type='binary')
                if direction < 0:
                    # c_{ex_rxn.id}: ex_rxn > -y_{ex_rxn.id}
                    constraints.append(Constraint(ex_rxn.flux_expression + max_uptake*variables['y_' + ex_rxn.id], lb=0, name='c_'+ex_rxn.id))
                else:
                    # c_{ex_rxn.id}: ex_rxn < y_{ex_rxn.id}
                    constraints.append(Constraint(ex_rxn.flux_expression - max_uptake*variables['y_' + ex_rxn.id], ub=0, name='c_'+ex_rxn.id))
            if max_compounds:
                # c_{ex_rxn.id}: sum(y_{ex_rxn.id}) < max_compounds
                constraints.append(Constraint(
                    sum(variables['y_' + ex_rxn.id] for ex_rxn in exchange_reactions), ub=max_compounds, name="max_compounds"))
            objective = Objective(sum(variables['y_' + ex_rxn.id] for ex_rxn in exchange_reactions), direction="min")
        else:
            for ex_rxn in exchange_reactions:
                variables['f_' + ex_rxn.id] = Variable('f_' + ex_rxn.id, 0, max_uptake)
                if direction < 0:
                    # c_{ex_rxn.id}: ex_rxn < -f_{ex_rxn.id}
                    constraints.append(Constraint(ex_rxn.flux_expression + max_uptake*variables['f_' + ex_rxn.id], lb=0, name='c_'+ex_rxn.id))
                else:
                    # c_{ex_rxn.id}: ex_rxn > f_{ex_rxn.id}
                    constraints.append(Constraint(ex_rxn.flux_expression - max_uptake*variables['f_' + ex_rxn.id], ub=0, name='c_'+ex_rxn.id))
            objective = Objective(sum(variables['f_' + ex_rxn.id] for ex_rxn in exchange_reactions), direction="min")

        model = FBAHelper.add_vars_cons(model, list(variables.values())+constraints)
        model.objective = objective
        model.solver.update()

        medium, solution = None, None
        for ex_rxn in exchange_reactions:  # !!! are the default bounds of (-max_uptake,-max_uptake) and (0,0) sensible?
            if direction < 0:
                ex_rxn.bounds = (0,ex_rxn.upper_bound) if ex_rxn not in exchange_reactions else (-max_uptake, -max_uptake)
            else:
                ex_rxn.bounds = (ex_rxn.lower_bound, max_uptake) if ex_rxn not in exchange_reactions else (0, 0)
        model.reactions.bio1.bounds = (min_growth, inf)

        if n_solutions == 1:
            solution = model.optimize()
            medium = get_medium(solution, exchange_reactions, direction)
            if validate:
                validate_solution(model, medium, direction, min_growth, max_uptake)
            return medium, solution

        media, solutions = [], []
        for i in range(n_solutions):
            if i > 0:
                previous_sol = [variables['y_' + ex_rxn.id] for r_id in medium]
                model = FBAHelper.add_vars_cons(model, [Constraint(sum(previous_sol), ub=len(previous_sol)-1, name=f"iteration_{i}")])
            solution = model.optimize('min')
            if solution.status != 'optimal':
                break

            medium = get_medium(solution, exchange_reactions, direction)
            media.append(medium)
            solutions.append(solution)

        return medium, solution

    def smetana_score(self, min_growth=0.1, n_solutions=100, abstol=1e-6, min_mol_weight=False, max_uptake=10.0):  # the sum of all interspecies dependencies under a given nutritional environment
        scs = self.sc_score(min_growth, n_solutions, abstol)
        mus = self.mu_score(min_mol_weight, min_growth, max_uptake, abstol, n_solutions)
        mps = self.mp_score(abstol)

        smtna_score = 0
        for index, model in enumerate(self.models):
            other_models = self.models.pop(index)
            for model2 in other_models:
                if all([mus[model.id], scs[model.id], mps[model.id]]) and all([model2.id in x for x in [mus[model.id], scs[model.id], mps[model.id]]]):
                    smtna_score += mus[model.id][model2.id] * scs[model.id][model2.id] * mps[model.id][model2.id]
        return smtna_score