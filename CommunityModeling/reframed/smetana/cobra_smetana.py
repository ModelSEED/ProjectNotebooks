
class smetana():
    
    def __init__(self,community):
        

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
            solver.add_variable(org_var, 0, 1, vartype=VarType.BINARY, update=False)

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

                if sol.status != Status.OPTIMAL:
                    failed = i == 0
                    break

                donors = [donor for donor in other if sol.values["y_{}".format(o)] > abstol]  # acquire the list of community members whose primal values surpass a threshold  !!! how can the binary variables have float primal values?
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
        solver = solver_instance(community.merged)

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