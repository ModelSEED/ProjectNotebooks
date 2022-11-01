from types import FunctionType
from collections import Counter
from collections.abc import Iterable
from itertools import combinations, chain
from chemw import ChemMW
from warnings import warn
from math import isinf, inf
from enum import Enum

class ReactionType(Enum):
    """ Enumeration of possible reaction types. """
    ENZYMATIC = 'enzymatic'
    TRANSPORT = 'transport'
    EXCHANGE = 'exchange'
    SINK = 'sink'
    OTHER = 'other'

class Status(Enum):
    """ Enumeration of possible solution status. """
    OPTIMAL = 'Optimal'
    UNKNOWN = 'Unknown'
    SUBOPTIMAL = 'Suboptimal'
    UNBOUNDED = 'Unbounded'
    INFEASIBLE = 'Infeasible'
    INF_OR_UNB = 'Infeasible or Unbounded'
    
class VarType(Enum):
    """ Enumeration of possible variable types. """
    BINARY = 'binary'
    INTEGER = 'integer'
    CONTINUOUS = 'continuous'
    
class AttrOrderedDict(OrderedDict):
    """Helper class to extend ordered dictionaries with indexing"""

    def __init__(self, *args, **nargs):
        super(AttrOrderedDict, self).__init__(*args, **nargs)

    def __getattr__(self, name):
        if not name.startswith('_'):
            return self[name]
        super(AttrOrderedDict, self).__getattr__(name)

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            self[name] = value
        else:
            super(AttrOrderedDict, self).__setattr__(name, value)

    def __dir__(self):
        return dir(OrderedDict) + list(self.keys())

    def __copy__(self):
        my_copy = AttrOrderedDict()
        for key, val in self.items():
            my_copy[key] = copy(val)
        return my_copy

    def __deepcopy__(self, memo):
        my_copy = AttrOrderedDict()
        for key, val in self.items():
            my_copy[key] = deepcopy(val)
        return my_copy
    

def FBA(model, objective=None, minimize=False, constraints=None, solver=None, get_values=True,
        shadow_prices=False, reduced_costs=False):
    if not objective:
        objective = model.get_objective()

        if len(objective) == 0:
            warn('Model objective undefined.')

    if not solver:
        solver = model.solver

    solution = solver.solve(objective, minimize=minimize, constraints=constraints, get_values=get_values,
                            shadow_prices=shadow_prices, reduced_costs=reduced_costs)
    return solution


class Environment(AttrOrderedDict):
    """ This class represents the exchange of compounds between an organism and the environment. """

    def __init__(self):
        AttrOrderedDict.__init__(self)

    def __str__(self):
        entries = (f"{r_id}\t{lb}\t{ub}" for r_id, (lb, ub) in self.items())
        return '\n'.join(entries)

    def __repr__(self):
        return str(self)

    def get_compounds(self, fmt_func=None):
        if fmt_func is None:
            fmt_func = lambda x: x[5:-2]
        elif not isinstance(fmt_func, FunctionType):
            raise RuntimeError("fmt_func argument must be a string or function.")

        compounds = []

        for r_id, (lb, _) in self.items():
            if lb < 0:
                compounds.append(fmt_func(r_id))

        return compounds

    def apply(self, model, exclusive=True, inplace=True, warning=True):
        if exclusive:
            env = Environment.empty(model)
            env.update(self)
        else:
            env = self

        if not inplace:
            constraints = {}

        for r_id, (lb, ub) in env.items():
            if r_id in model.reactions:
                if inplace:
                    model.set_flux_bounds(r_id, lb, ub)
                else:
                    constraints[r_id] = (lb, ub)
            elif warning:
                warn(f'Exchange reaction not in model: {r_id}')

        if not inplace:
            return constraints

    @staticmethod
    def from_reactions(reactions, max_uptake=10.0):
        env = Environment()
        for r_id in reactions:
            env[r_id] = (-max_uptake, inf)

        return env

def minimal_medium(model, exchange_reactions=None, direction=-1, min_mass_weight=False, min_growth=1, max_uptake=100, max_compounds=None,
                   n_solutions=1, validate=True, abstol=1e-6, warnings=True, milp=True, use_pool=False, pool_gap=None, solver=None):
    def warn_wrapper(message):
        if warnings:
            warn(message)

    if exchange_reactions is None:
        exchange_reactions = set([rxn.id for rxn in model.reactions.values() 
                          if rxn.reaction_type == ReactionType.EXCHANGE])

    if not solver:
        solver = model.solver

    if not milp and max_compounds is not None:
        raise RuntimeError("max_compounds can only be used with MILP formulation")

    if not milp and n_solutions > 1:
        raise RuntimeError("n_solutions can only be used with MILP formulation")

    if milp:
        for r_id in exchange_reactions:
            solver.add_variable('y_' + r_id, 0, 1, vartype=VarType.BINARY, update=False)
    else:
        for r_id in exchange_reactions:
            solver.add_variable('f_' + r_id, 0, max_uptake, update=False)

    solver.update()

    if milp:
        for r_id in exchange_reactions:
            if direction < 0:
                solver.add_constraint('c_' + r_id, {r_id: 1, 'y_' + r_id: max_uptake}, '>', 0, update=False)
            else:
                solver.add_constraint('c_' + r_id, {r_id: 1, 'y_' + r_id: -max_uptake}, '<', 0, update=False)

        if max_compounds:
            lhs = {'y_' + r_id: 1 for r_id in exchange_reactions}
            solver.add_constraint('max_cmpds', lhs, '<', max_compounds, update=False)

    else:
        for r_id in exchange_reactions:
            if direction < 0:
                solver.add_constraint('c_' + r_id, {r_id: 1, 'f_' + r_id: 1}, '>', 0, update=False)
            else:
                solver.add_constraint('c_' + r_id, {r_id: 1, 'f_' + r_id: -1}, '<', 0, update=False)

    solver.update()

    valid_reactions = []

    if min_mass_weight:
        objective = {}

        multiple_compounds =[]
        no_compounds = []
        no_formula = []
        invalid_formulas = []

        for r_id in exchange_reactions:

            if direction < 0:
                compounds = model.reactions[r_id].get_substrates()
            else:
                compounds = model.reactions[r_id].get_products()

            if len(compounds) > 1:
                multiple_compounds.append(r_id)
                continue

            if len(compounds) == 0:
                no_compounds.append(r_id)
                continue

            metabolite = model.metabolites[compounds[0]]

            if 'FORMULA' not in metabolite.metadata:
                no_formula.append(metabolite.id)
                continue

            formula = metabolite.metadata['FORMULA']
            weight = molecular_weight(formula)

            if weight is None:
                invalid_formulas.append(metabolite.id)
                continue

            if milp:
                objective['y_' + r_id] = weight
            else:
                objective['f_' + r_id] = weight

            valid_reactions.append(r_id)

        if multiple_compounds:
            warn_wrapper(f"Reactions ignored (multiple compounds): {multiple_compounds}")
        if no_compounds:
            warn_wrapper(f"Reactions ignored (no compounds): {no_compounds}")
        if multiple_compounds:
            warn_wrapper(f"Compounds ignored (no formula): {no_formula}")
        if invalid_formulas:
            warn_wrapper(f"Compounds ignored (invalid formula): {invalid_formulas}")

    else:
        if milp:
            objective = {'y_' + r_id: 1 for r_id in exchange_reactions}
        else:
            objective = {'f_' + r_id: 1 for r_id in exchange_reactions}

        valid_reactions = exchange_reactions

    result, ret_sols = None, None

    if direction < 0:
        constraints = {r_id: (-max_uptake if r_id in valid_reactions else 0, model.reactions[r_id].ub)
                       for r_id in exchange_reactions}
    else:
        constraints = {r_id: (model.reactions[r_id].lb, max_uptake if r_id in valid_reactions else 0)
                       for r_id in exchange_reactions}

    constraints[model.biomass_reaction] = (min_growth, inf)

    if n_solutions == 1:

        solution = solver.solve(objective, minimize=True, constraints=constraints, get_values=exchange_reactions)

        if solution.status != Status.OPTIMAL:
            warn_wrapper('No solution found')
            result, ret_sols = None, solution
        else:
            medium = get_medium(solution, exchange_reactions, direction, abstol)

            if validate:
                validate_solution(model, medium, exchange_reactions, direction, min_growth, max_uptake)

            result, ret_sols = medium, solution

    elif use_pool:
        solutions = solver.solve(objective, minimize=True, constraints=constraints, get_values=exchange_reactions,
                                 pool_size=n_solutions, pool_gap=pool_gap)

        if solutions is None:
            result, ret_sols = [], []
        else:
            media = [get_medium(solution, exchange_reactions, direction, abstol) for solution in solutions]
            result, ret_sols = media, solutions
    else:
        media = []
        solutions = []

        for i in range(0, n_solutions):
            if i > 0:
                constr_id = f"iteration_{i}"
                previous_sol = {'y_' + r_id: 1 for r_id in medium}
                solver.add_constraint(constr_id, previous_sol, '<', len(previous_sol) - 1)

            solution = solver.solve(objective, minimize=True, constraints=constraints, get_values=exchange_reactions)

            if solution.status != Status.OPTIMAL:
                break

            medium = get_medium(solution, exchange_reactions, direction, abstol)
            media.append(medium)
            solutions.append(solution)

            result, ret_sols = media, solutions

    return result, ret_sols


def sc_score(community, environment=None, min_growth=0.1, n_solutions=100, verbose=True, abstol=1e-6,
             use_pool=True):
    community = community.copy(copy_models=False, interacting=True, create_biomass=False,
                               merge_extracellular_compartments=False)

    if environment:
        environment.apply(community.merged, inplace=True, warning=False)

    for b in community.organisms_biomass_reactions.values():
        community.merged.reactions[b].lb = 0

    solver = community.merged.solver

    for org_id in community.organisms:
        org_var = 'y_{}'.format(org_id)
        solver.add_variable(org_var, 0, 1, vartype=VarType.BINARY, update=False)

    solver.update()

    bigM = 1000
    for org_id, rxns in community.organisms_reactions.items():
        org_var = 'y_{}'.format(org_id)
        for r_id in rxns:
            if r_id == community.organisms_biomass_reactions[org_id]:
                continue
            solver.add_constraint('c_{}_lb'.format(r_id), {r_id: 1, org_var: bigM}, '>', 0, update=False)
            solver.add_constraint('c_{}_ub'.format(r_id), {r_id: 1, org_var: -bigM}, '<', 0, update=False)

    solver.update()

    scores = {}

    for org_id, biomass_id in community.organisms_biomass_reactions.items():
        other = {o for o in community.organisms if o != org_id}
        solver.add_constraint('SMETANA_Biomass', {community.organisms_biomass_reactions[org_id]: 1}, '>', min_growth)
        objective = {"y_{}".format(o): 1.0 for o in other}

        if not use_pool:
            previous_constraints = []
            donors_list = []
            failed = False

            for i in range(n_solutions):
                sol = solver.solve(objective, minimize=True, get_values=list(objective.keys()))

                if sol.status != Status.OPTIMAL:
                    failed = i == 0
                    break

                donors = [o for o in other if sol.values["y_{}".format(o)] > abstol]
                donors_list.append(donors)

                previous_con = 'iteration_{}'.format(i)
                previous_constraints.append(previous_con)
                previous_sol = {"y_{}".format(o): 1 for o in donors}
                solver.add_constraint(previous_con, previous_sol, '<', len(previous_sol) - 1)

            solver.remove_constraints(['SMETANA_Biomass'] + previous_constraints)

            if not failed:
                donors_list_n = float(len(donors_list))
                donors_counter = Counter(chain(*donors_list))
                scores[org_id] = {o: donors_counter[o] / donors_list_n for o in other}
            else:
                if verbose:
                    warn('SCS: Failed to find a solution for growth of ' + org_id)
                scores[org_id] = None

        else:
            sols = solver.solve(objective, minimize=True, get_values=list(objective.keys()),
                                pool_size=n_solutions, pool_gap=0.5)
            solver.remove_constraint('SMETANA_Biomass')

            if len(sols) == 0:
                scores[org_id] = None
                if verbose:
                    warn('SCS: Failed to find a solution for growth of ' + org_id)
            else:
                donor_count = [o for sol in sols for o in other if sol.values["y_{}".format(o)] > abstol]
                donor_count = Counter(donor_count)
                scores[org_id] = {o: donor_count[o] / len(sols) for o in other}

    return scores


def mu_score(community, environment=None, min_mol_weight=False, min_growth=0.1, max_uptake=10.0,
             abstol=1e-6, validate=False, n_solutions=100, pool_gap=0.5, verbose=True):
    if environment:
        environment.apply(community.merged, inplace=True, warning=False)

    max_uptake = max_uptake * len(community.organisms)
    scores = {}
    solver = community.merged.solver

    for org_id in community.organisms:
        exchange_rxns = community.organisms_exchange_reactions[org_id]
        biomass_reaction = community.organisms_biomass_reactions[org_id]
        community.merged.biomass_reaction = biomass_reaction

        medium_list, sols = minimal_medium(community.merged, exchange_reactions=list(exchange_rxns.keys()),
                                           min_mass_weight=min_mol_weight, min_growth=min_growth,
                                           n_solutions=n_solutions, max_uptake=max_uptake, validate=validate,
                                           abstol=abstol, use_pool=True, pool_gap=pool_gap, solver=solver,
                                           warnings=False)

        if medium_list:
            counter = Counter(chain(*medium_list))

            scores[org_id] = {cnm.original_metabolite: counter[ex] / len(medium_list)
                              for ex, cnm in exchange_rxns.items()}
        else:
            if verbose:
                warn('MUS: Failed to find a minimal growth medium for ' + org_id)
            scores[org_id] = None

    return scores


def mp_score(community, environment=None, abstol=1e-3):
    if environment:
        environment.apply(community.merged, inplace=True, warning=False)
        env_compounds = environment.get_compounds(fmt_func=lambda x: x[5:-5])
    else:
        env_compounds = set()

    for exchange_rxns in community.organisms_exchange_reactions.values():  # !!! where is organisms_exchange_reactions defined?
        for r_id in exchange_rxns.keys():
            rxn = community.merged.reactions[r_id]
            if isinf(rxn.ub):
                rxn.ub = 1000

    solver = community.merged.solver

    scores = {}

    for org_id, exchange_rxns in community.organisms_exchange_reactions.items():
        scores[org_id] = {}

        remaining = [r_id for r_id, cnm in exchange_rxns.items() if cnm.original_metabolite not in env_compounds]

        while len(remaining) > 0:
            sol = solver.solve(linear={r_id: 1 for r_id in remaining}, minimize=False, get_values=remaining)

            if sol.status != Status.OPTIMAL:
                break

            blocked = [r_id for r_id in remaining if sol.values[r_id] < abstol]

            if len(blocked) == len(remaining):
                break

            for r_id in remaining:
                if sol.values[r_id] >= abstol:
                    cnm = exchange_rxns[r_id]
                    scores[org_id][cnm.original_metabolite] = 1

            remaining = blocked

        for r_id in remaining:
            sol = solver.solve(linear={r_id: 1}, minimize=False, get_values=False)
            cnm = exchange_rxns[r_id]

            if sol.status == Status.OPTIMAL and sol.fobj > abstol:
                scores[org_id][cnm.original_metabolite] = 1
            else:
                scores[org_id][cnm.original_metabolite] = 0

    return scores


def mip_score(community, environment=None, min_mol_weight=False, min_growth=0.1, direction=-1, max_uptake=10,
              validate=False, verbose=True, use_lp=False, exclude=None):
    noninteracting = community.copy(copy_models=False, interacting=False)
    exch_reactions = set([rxn.id for rxn in community.merged.reactions.values() 
                          if rxn.reaction_type == ReactionType.EXCHANGE])
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

    interacting_medium, sol2 = minimal_medium(community.merged, direction=direction, exchange_reactions=noninteracting_medium,
                                              min_mass_weight=min_mol_weight, min_growth=min_growth, milp=(not use_lp),
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


def mro_score(community, environment=None, direction=-1, min_mol_weight=False, min_growth=0.1, max_uptake=10,
              validate=False, verbose=True, use_lp=False, exclude=None):
    
    exch_reactions = set([rxn.id for rxn in community.merged.reactions.values() 
                          if rxn.reaction_type == ReactionType.EXCHANGE])
    max_uptake = max_uptake * len(community.organisms)

    if environment:
        environment.apply(community.merged, inplace=True, warning=False)
        exch_reactions &= set(environment)

    medium, sol = minimal_medium(community.merged, exchange_reactions=exch_reactions, direction=direction,
                                 min_mass_weight=min_mol_weight, min_growth=min_growth, max_uptake=max_uptake,
                                 validate=validate,  warnings=False, milp=(not use_lp))

    if sol.status != Status.OPTIMAL:
        if verbose:
            warn('MRO: Failed to find a valid solution for community')
        return None, None

    interacting_env = Environment.from_reactions(medium, max_uptake=max_uptake)
    interacting_env.apply(community.merged, inplace=True)

    if exclude is None:
        exclude = set()

    medium = {x[7:-7] for x in medium} - exclude
    individual_media = {}
    solver = community.merged.solver

    for org_id in community.organisms:
        biomass_reaction = community.organisms_biomass_reactions[org_id]
        community.merged.biomass_reaction = biomass_reaction
        org_interacting_exch = community.organisms_exchange_reactions[org_id]

        medium_i, sol = minimal_medium(community.merged, exchange_reactions=org_interacting_exch, direction=direction,
                                     min_mass_weight=min_mol_weight, min_growth=min_growth, max_uptake=max_uptake,
                                     validate=validate, solver=solver, warnings=False, milp=(not use_lp))

        if sol.status != Status.OPTIMAL:
            warn('MRO: Failed to find a valid solution for: ' + org_id)
            return None, None

        individual_media[org_id] = {org_interacting_exch[r].original_metabolite[2:-2] for r in medium_i} - exclude

    pairwise = {(o1, o2): individual_media[o1] & individual_media[o2] for o1, o2 in combinations(community.organisms, 2)}

    numerator = sum(map(len, pairwise.values())) / len(pairwise) if len(pairwise) != 0 else 0
    denominator = sum(map(len, individual_media.values())) / len(individual_media) if len(individual_media) != 0 else 0
    score = numerator / denominator if denominator != 0 else None

    extras = {
        'community_medium': medium,
        'individual_media': individual_media
    }

    return score, extras


def minimal_environment(community, aerobic=None, min_mol_weight=False, min_growth=0.1, max_uptake=10,
                        validate=False, verbose=True, use_lp=False):

    exch_reactions = set([rxn.id for rxn in community.merged.reactions.values() 
                          if rxn.reaction_type == ReactionType.EXCHANGE])

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