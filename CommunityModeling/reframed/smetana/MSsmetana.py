from reframed import minimal_medium, Environment, solver_instance
from smetana.legacy import Community
from modelseedpy.core.fbahelper import FBAHelper
from collections import Counter
from itertools import combinations, chain
from typing import Union
from warnings import warn
from math import isinf, inf


from modelseedpy.community.mscommunity import MSCommunity
from modelseedpy.core.fbahelper import FBAHelper
from optlang import Variable, Constraint, Objective
from copy import deepcopy


class Smetana:

    def __init__(self, cobra_models: Union[list, tuple, set], media):
        # convert COBRA model into ReFramed model
        self.models = cobra_models
        self.community = MSCommunity.build_from_species_models(cobra_models, name="SMETANA_example")  # abundances argument may be valuable

        if environment:  # !!! set fluxes to respective reactions in a model based upon the environmental media
            environment.apply(community.merged, inplace=True, warning=False)
            self.media_cpds =

    def sc_score(self, min_growth=0.1, n_solutions=100, abstol=1e-6):
        # community = community.copy(copy_models=False, interacting=True, create_biomass=False, merge_extracellular_compartments=False)

        # identify the biomass compounds of each respective community model
        for rxn in self.community.reactions:
            rxn.lb = 0 if 'bio' in rxn.id else rxn.lb
        model_biomasses = {}
        for model in self.models:
            msid_cobraid_hash = FBAHelper.msid_hash(model)
            if "cpd11416" not in msid_cobraid_hash:
                warn(f"Could not find biomass compound for the {model.id} model.")
            for biomass_cpd in msid_cobraid_hash["cpd11416"]:
                if biomass_cpd.compartment == "c0":
                    model_biomasses[model.id] = biomass_cpd
                    break

        # reaction flux bounds, with consideration of each species
        # c_{rxn.id}_lb: rxn > 1000*y_{species_id}
        # c_{rxn.id}_ub: rxn > 1000*y_{species_id}
        variables = {}
        constraints = []
        for model in self.models:
            variables[model.id] = Variable(f'y_{model.id}', 0, 1, 'binary')
            for rxn in model.reactions:
                if rxn.id != model_biomasses[model.id]:
                    lb = Constraint(rxn - 1000*variables[model.id], name=f'c_{rxn.id}_lb', ub=0)
                    ub = Constraint(rxn + 1000*variables[model.id], name=f"c_{rxn.id}_ub", lb=0)
                    constraints.extend([lb, ub])

        self.community.problem.add_cons_vars(list(variables.values())+constraints)
        self.community.problem.update()

        # calculate the SCS
        scores = {}
        com_model = self.community.copy()
        for model in self.models:
            other_members = {other for other in self.models if other.id != model.id}
            com_model.problem.add_cons_vars(Constraint(model.reactions.bio1, name='SMETANA_Biomass', lb=min_growth))
            objective = {f"y_{other}": 1.0 for other in other_members}

            previous_constraints, donors_list = [], []
            failed = False
            for i in range(n_solutions):
                com_model.problem.objective = Objective(objective, direction="min")
                sol = com_model.optimize()

                if sol.status.value != 'optimal':
                    failed = i == 0
                    break

                donors = [o for o in other_members if sol.values[f"y_{o.id}"] > abstol]
                donors_list.append(donors)

                previous_con = f'iteration_{i}'
                previous_constraints.append(previous_con)
                com_model.problem.add_cons_vars(Constraint(
                    sum(variables[o.id] for o in donors), name=previous_con, ub=len(previous_constraints)-1))

            com_model.problem.remove("SMETANA_biomass")
            if not failed:
                donors_list_n = float(len(donors_list))
                donors_counter = Counter(chain(*donors_list))
                scores[model.id] = {o: donors_counter[o] / donors_list_n for o in other_members}
            else:
                warn('SCS: Failed to find a solution for growth of ' + model.id)
                scores[model.id] = None
        return scores

    def mu_score(self, min_mol_weight=False, min_growth=0.1, max_uptake=10.0, abstol=1e-6, validate=False, n_solutions=100):
        max_uptake = max_uptake * len(self.models)
        scores = {}
        for model in self.models:
            biomass_reaction = community.organisms_biomass_reactions[org_id]

            community.merged.biomass_reaction = biomass_reaction  # !!! change the biomass reaction of the community to the species model

            ex_rxns = {ex_rxn.id:met for ex_rxn in model.exchanges for met in ex_rxn.metabolites}
            medium_list, sols = minimal_medium(
                self.community, exchange_reactions=ex_rxns, min_mass_weight=min_mol_weight,
                min_growth=min_growth, n_solutions=n_solutions, max_uptake=max_uptake, validate=validate,
                abstol=abstol, warnings=False)

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
                self.community.objective = Objective({ex_rxn: 1 for ex_rxn in remaining})
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
                self.community.objective = Objective({ex_rxn: 1 for ex_rxn in remaining})
                sol = self.community.optimize()

                score = 0
                if sol.status == 'optimal' and sol.fobj > abstol:
                    scores = 1
                for met in ex_rxn:
                    scores[model.id][met.id] = 1

        return scores

    def mip_score(self, community=None, environment=None, min_mol_weight=False, min_growth=0.1, direction=-1,
                  max_uptake=10, validate=False, verbose=True, use_lp=False, exclude=None):
        """
        Implements the metabolic interaction potential (MIP) score as defined in (Zelezniak et al, 2015).

        Args:
            community (Community): microbial community model
            environment (Environment): Metabolic environment in which the SMETANA score is calculated
            direction (int): direction of uptake reactions (negative or positive, default: -1)
            extracellular_id (str): extracellular compartment id
            min_mol_weight (bool): minimize by molecular weight of nutrients (default: False)
            min_growth (float): minimum growth rate (default: 0.1)
            max_uptake (float): maximum uptake rate (default: 10)
            validate (bool): validate solution using FBA (for debugging purposes, default: False)

        Returns:
            float: MIP score
        """
        community = community or self.model

        noninteracting = community.copy(copy_models=False, interacting=False)
        exch_reactions = set(community.merged.get_exchange_reactions())
        max_uptake = max_uptake * len(community.organisms)

        if environment:
            environment.apply(noninteracting.merged, inplace=True, warning=False)
            exch_reactions &= set(environment)

        noninteracting_medium, sol1 = minimal_medium(noninteracting.merged, exchange_reactions=exch_reactions,
                                                     direction=direction, min_mass_weight=min_mol_weight,
                                                     min_growth=min_growth, max_uptake=max_uptake, validate=validate,
                                                     warnings=False, milp=(not use_lp))
        if noninteracting_medium is None:
            if verbose:
                warn('MIP: Failed to find a valid solution for non-interacting community')
            return None, None

        # anabiotic environment is limited to non-interacting community minimal media
        noninteracting_env = Environment.from_reactions(noninteracting_medium, max_uptake=max_uptake)
        noninteracting_env.apply(community.merged, inplace=True)

        interacting_medium, sol2 = minimal_medium(community.merged, direction=direction,
                                                  exchange_reactions=noninteracting_medium,
                                                  min_mass_weight=min_mol_weight, min_growth=min_growth,
                                                  milp=(not use_lp),
                                                  max_uptake=max_uptake, validate=validate, warnings=False)

        if interacting_medium is None:
            if verbose:
                warn('MIP: Failed to find a valid solution for interacting community')
            return None, None

        if exclude is not None:
            exclude_rxns = {'R_EX_M_{}_e_pool'.format(x) for x in exclude}
            interacting_medium = set(interacting_medium) - exclude_rxns
            noninteracting_medium = set(noninteracting_medium) - exclude_rxns

        score = len(noninteracting_medium) - len(interacting_medium)

        noninteracting_medium = [r_id[7:-7] for r_id in noninteracting_medium]
        interacting_medium = [r_id[7:-7] for r_id in interacting_medium]

        extras = {
            'noninteracting_medium': noninteracting_medium,
            'interacting_medium': interacting_medium
        }

        return score, extras

    def mro_score(self, community=None, environment=None, direction=-1, min_mol_weight=False, min_growth=0.1,
                  max_uptake=10, validate=False, verbose=True, use_lp=False, exclude=None):
        """
        Implements the metabolic resource overlap (MRO) score as defined in (Zelezniak et al, 2015).

        Args:
            community (Community): microbial community model
            environment (Environment): Metabolic environment in which the SMETANA score is colulated
            direction (int): direction of uptake reactions (negative or positive, default: -1)
            extracellular_id (str): extracellular compartment id
            min_mol_weight (bool): minimize by molecular weight of nutrients (default: False)
            min_growth (float): minimum growth rate (default: 0.1)
            max_uptake (float): maximum uptake rate (default: 10)

        Returns:
            float: MRO score
        """
        community = community or self.model

        exch_reactions = set(community.merged.get_exchange_reactions())
        max_uptake = max_uptake * len(community.organisms)

        if environment:
            environment.apply(community.merged, inplace=True, warning=False)
            exch_reactions &= set(environment)

        medium, sol = minimal_medium(community.merged, exchange_reactions=exch_reactions, direction=direction,
                                     min_mass_weight=min_mol_weight, min_growth=min_growth, max_uptake=max_uptake,
                                     validate=validate, warnings=False, milp=(not use_lp))

        if sol.status.value != 'Optimal':
            if verbose:
                warn('MRO: Failed to find a valid solution for community')
            return None, None

        interacting_env = Environment.from_reactions(medium, max_uptake=max_uptake)
        interacting_env.apply(community.merged, inplace=True)

        if exclude is None:
            exclude = set()

        medium = {x[7:-7] for x in medium} - exclude
        individual_media = {}
        solver = solver_instance(community.merged)

        for org_id in community.organisms:
            biomass_reaction = community.organisms_biomass_reactions[org_id]
            community.merged.biomass_reaction = biomass_reaction
            org_interacting_exch = community.organisms_exchange_reactions[org_id]

            medium_i, sol = minimal_medium(community.merged, exchange_reactions=org_interacting_exch,
                                           direction=direction,
                                           min_mass_weight=min_mol_weight, min_growth=min_growth, max_uptake=max_uptake,
                                           validate=validate, solver=solver, warnings=False, milp=(not use_lp))

            if sol.status.value != 'Optimal':
                warn('MRO: Failed to find a valid solution for: ' + org_id)
                return None, None

            individual_media[org_id] = {org_interacting_exch[r].original_metabolite[2:-2] for r in medium_i} - exclude

        pairwise = {(o1, o2): individual_media[o1] & individual_media[o2] for o1, o2 in
                    combinations(community.organisms, 2)}

        numerator = sum(map(len, pairwise.values())) / len(pairwise) if len(pairwise) != 0 else 0
        denominator = sum(map(len, individual_media.values())) / len(individual_media) if len(
            individual_media) != 0 else 0
        score = numerator / denominator if denominator != 0 else None

        extras = {
            'community_medium': medium,
            'individual_media': individual_media
        }

        return score, extras

    def minimal_environment(self, community=None, aerobic=None, min_mol_weight=False, min_growth=0.1,
                            max_uptake=10, validate=False, verbose=True, use_lp=False):

        exch_reactions = set(community.merged.get_exchange_reactions())

        exch_reactions -= {"R_EX_M_h2o_e_pool"}
        community.merged.set_flux_bounds("R_EX_M_h2o_e_pool", -inf, inf)

        if aerobic is not None:
            exch_reactions -= {"R_EX_M_o2_e_pool"}
            if aerobic:
                community.merged.set_flux_bounds("R_EX_M_o2_e_pool", -max_uptake, inf)
            else:
                community.merged.set_flux_bounds("R_EX_M_o2_e_pool", 0, inf)

        ex_rxns, sol = minimal_medium(community.merged, exchange_reactions=exch_reactions,
                                      min_mass_weight=min_mol_weight, min_growth=min_growth, milp=(not use_lp),
                                      max_uptake=max_uptake, validate=validate, warnings=False)

        if ex_rxns is None:
            if verbose:
                warn('Failed to find a medium for interacting community.')
            return None
        else:
            if aerobic is not None and aerobic:
                ex_rxns |= {"R_EX_M_o2_e_pool"}
            env = Environment.from_reactions(ex_rxns, max_uptake=max_uptake)
            env["R_EX_M_h2o_e_pool"] = (-inf, inf)
            return env

    def smetana_score(self, community=None, environment=None, min_mol_weight=False, min_growth=0.1, max_uptake=10.0,
                      validate=False, n_solutions=100, verbose=True,
                      abstol=1e-6):  # the sum of all interspecies dependencies under a given nutritional environment
        # define the individual scores
        community = community or self.model
        scs = self.sc_score(community, environment=None, min_growth=0.1, n_solutions=100, abstol=1e-6)
        mus = self.mu_score(community, environment=None, min_mol_weight=False, min_growth=0.1, max_uptake=10.0,
                            abstol=1e-6, validate=False, n_solutions=100, verbose=True)
        mps = self.mp_score(community, environment=None, abstol=1e-6)

        # calculate the total score, which is normalized 0<x<1
        smtna_score = 0
        for species in community.organisms:
            for species2 in community.organisms:
                if species != species2 and all([mus[species], scs[species], mps[species]]):
                    if all([species2 in x for x in [mus[species], scs[species], mps[species]]]):
                        smtna_score += mus[species][species2] * scs[species][species2] * mps[species][species2]
        return smtna_score