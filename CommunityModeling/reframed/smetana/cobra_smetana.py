from collections import Counter
from itertools import chain
from warnings import warn
from chemw import ChemMW
from math import inf


def minimal_medium(model, exchange_reactions=None, direction=-1, min_mass_weight=False, min_growth=1, max_uptake=100, max_compounds=None, n_solutions=1, validate=True, abstol=1e-6, warnings=True, milp=True, use_pool=False, pool_gap=None, solver=None):  
    """ Establishes the minimal media for a model to grow """
    def warn_wrapper(message):
        if warnings:
            warn(message)

    if exchange_reactions is None:
        exchange_reactions = set([rxn.id for rxn in model.reactions.values() if 'EX_' in rxn.id])

    if not solver:
        solver = model.solver

    if not milp and max_compounds is not None:
        raise RuntimeError("max_compounds can only be used with MILP formulation")

    if not milp and n_solutions > 1:
        raise RuntimeError("n_solutions can only be used with MILP formulation")

    if milp:
        for r_id in exchange_reactions:
            solver.add_variable('y_' + r_id, 0, 1, vartype='binary', update=False)
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
            chem_mw = ChemMW()
            weight = chem_mw.mass(formula)

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

        if solution.status != 'Optimal':
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

            if solution.status != 'Optimal':
                break

            medium = get_medium(solution, exchange_reactions, direction, abstol)
            media.append(medium)
            solutions.append(solution)

            result, ret_sols = media, solutions

    return result, ret_sols

class smetana():
    
    def __init__(self,community):
        pass
    
    def mip_score(self,):  # minimal nutritional requirements were determined through the method that is articulated by Segre in 2010, where each substrate is systematically removed and those iterations that reduce growth are deemed essential
        
        return  # the difference between the minimal quantities of compounds to grow a community with and without syntropy
    
    def mro_score(self,):  # maximum possible overlap of the minial nutritional requirements of all community species
        
        species_nutrients = {}
        for species_model in community:
            species_nutrients[species] = minimal_medium(species_model)  # returns a set
        
        min_nutrients = len(set(x for x in species_nutrients[species] for species in species_nutrients))  # The minimal quantity of required nutrients for community growth with syntropy
        mros_species = {}
        for species, num_nutrients in species_nutrients.items():
            mros_species[species] = mros_species[species] if species in mros_species else {}
            mros_species[species] += len(species_nutrients[spcs] & species_nutrients[species] for spcs in species_nutrients if spcs != species)  # The minimal quantity of required nutrients for community growth with syntropy
            
        
        return  # the difference between the minimal quantities of compounds to grow a community with and without syntropy
    
    def sc_score(self):
        return 
    


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

    def sc_score(community, environment=None, min_growth=0.1, n_solutions=100, verbose=True, abstol=1e-6,
                 use_pool=True):
        community = community.copy(copy_models=False, interacting=True, create_biomass=False,
                                   merge_extracellular_compartments=False)  # define a copy of the model with explicit attributes

        if environment:
            environment.apply(community.merged, inplace=True, warning=False)  # apply reaction constraints upon the Community model

        for b in community.organisms_biomass_reactions.values():  # no negative biomass
            community.merged.reactions[b].lb = 0

        solver = community.merged.solver  # define LP model

        for org_id in community.organisms:  # create a binary variable for each community member, which later comprises the objective
            org_var = 'y_{}'.format(org_id)
            solver.add_variable(org_var, 0, 1, vartype='binary', update=False)

        solver.update()

        for org_id, rxns in community.organisms_reactions.items():  # list of reactions (non-exchange or separate extracellular compartments) for each community species
            org_var = 'y_{}'.format(org_id)
            for r_id in rxns:
                if r_id == community.organisms_biomass_reactions[org_id]:  # biomass reaction
                    continue
                solver.add_constraint('c_{}_lb'.format(r_id), {r_id: 1, org_var: 1000}, '>', 0, update=False)
                solver.add_constraint('c_{}_ub'.format(r_id), {r_id: 1, org_var: -1000}, '<', 0, update=False)

        solver.update()

        scores = {}
        for org_id, biomass_id in community.organisms_biomass_reactions.items():
            other = {o for o in community.organisms if o != org_id}
            solver.add_constraint('SMETANA_Biomass', {community.organisms_biomass_reactions[org_id]: 1}, '>', min_growth)  # constrain organismal growth to the minimal growth rate
            objective = {"y_{}".format(o): 1.0 for o in other}  # an objective of binary variables for the community members

            previous_constraints = []
            donors_list = []
            failed = False

            for i in range(n_solutions):  # solves the system for a number of alternative solutions
                sol = solver.solve(objective, minimize=True, get_values=list(objective.keys()))

                if sol.status != 'Optimal':
                    failed = i == 0
                    break

                donors = [donor for donor in other if sol.values["y_{}".format(donor)] > abstol]  # acquire the list of community members whose primal values surpass a threshold  !!! how can the binary variables have float primal values?
                donors_list.append(donors)
                previous_sol = {"y_{}".format(o): 1 for o in donors}
                previous_con = 'iteration_{}'.format(i)
                
                previous_constraints.append(previous_con)
                solver.add_constraint(previous_con, previous_sol, '<', len(previous_sol) - 1)  # one of activated community member variables must be deactivated

            solver.remove_constraints(['SMETANA_Biomass'] + previous_constraints)

            scores[org_id] = None
            if not failed:  # count the prevalence of community members in all objective solutions
                donors_counter = Counter(chain(*donors_list))
                scores[org_id] = {o: donors_counter[o] / float(len(donors_list)) for o in other}
            else:
                if verbose:
                    warn('SCS: Failed to find a solution for growth of ' + org_id)

        return scores
    
    
    
    def mu_score(community, environment=None, min_mol_weight=False, min_growth=0.1, max_uptake=10.0,
                 abstol=1e-6, validate=False, n_solutions=100, pool_gap=0.5, verbose=True):
        if environment:  # apply reaction constraints upon the Community model
            environment.apply(community.merged, inplace=True, warning=False)

        max_uptake = max_uptake * len(community.organisms)  # a maximum intake per species is amplified to the whole community
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

            if medium_list:  # count the prevalence of exchange reactions in the media
                counter = Counter(chain(*medium_list))

                scores[org_id] = {cnm.original_metabolite: counter[ex] / len(medium_list)
                                  for ex, cnm in exchange_rxns.items()}
            else:
                if verbose:
                    warn('MUS: Failed to find a minimal growth medium for ' + org_id)
                scores[org_id] = None

        return scores