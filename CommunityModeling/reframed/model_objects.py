import re
from warnings import warn
from collections import OrderedDict
from math import inf
from optlang import Model, Variable, Constraint, Objective
from optlang.symbolics import Zero, add
from .solver import Solver, VarType, Parameter, default_parameters
from collections.abc import Iterable
from copy import deepcopy


class VarType(Enum):
    """ Enumeration of possible variable types. """
    BINARY = 'binary'
    INTEGER = 'integer'
    CONTINUOUS = 'continuous'
    
class Parameter(Enum):
    """ Enumeration of parameters common to all solvers. """
    TIME_LIMIT = 0
    FEASIBILITY_TOL = 1
    INT_FEASIBILITY_TOL = 2
    OPTIMALITY_TOL = 3
    MIP_REL_GAP = 4
    MIP_ABS_GAP = 5
    POOL_SIZE = 6
    POOL_GAP = 7


default_parameters = {
    Parameter.FEASIBILITY_TOL: 1e-9,
    Parameter.OPTIMALITY_TOL: 1e-9,
}

class CommunityNameMapping(object):
    def __init__(self, original_reaction=None, organism_reaction=None, original_metabolite=None,
                 organism_metabolite=None, extracellular_metabolite=None, community_exchange_reaction=None):
        """
        This class is used to represent mapping between original and merged community model metabolites and reactions

        Args:
            original_reaction (str): Name of reaction in original model
            organism_reaction (str): Name of reaction in merged community model
            original_metabolite (str): Name of metabolite in original model
            organism_metabolite (str): Name of metabolite in merged community model
            extracellular_metabolite (str): Name of "common environment" metabolite in merged community model
        """
        self.original_reaction = original_reaction
        self.organism_reaction = organism_reaction
        self.original_metabolite = original_metabolite
        self.organism_metabolite = organism_metabolite
        self.extracellular_metabolite = extracellular_metabolite
        self.community_exchange_reaction = community_exchange_reaction

    def __repr__(self):
        repr_str =  "<orig_m: {}, org_m: {}, ex_m: {}, orig_r: {}, org_r: {}, exch_r: {}>"
        return repr_str.format(self.original_metabolite,
                               self.organism_metabolite,
                               self.extracellular_metabolite,
                               self.original_reaction,
                               self.organism_reaction,
                               self.community_exchange_reaction)


class Community(object):
    """
    This class implements a microbial community model.

    It serves as a container for multiple organisms, and can be used to merge multiple single-species models (CBModel)
    into a single multi-species model (CBModel) which is compatible with most types of constraint-based methods.
    """

    def __init__(self, community_id, models=None, copy_models=True,
                 merge_extracellular_compartments=False, create_biomass=True, interacting=True,
                 exchanged_metabolites_blacklist=set()):
        """

        Args:
            community_id (str): community identifier
            models (list): list of models to be merged into single community
            copy_models (bool): If true copies for merged models  are created
            extracellular_compartment_id (str): Extracellular compartment id is used when merging extracellular compartments
            merge_extracellular_compartments (bool): Do not create organism specific extracellular compartment
            create_biomass (bool): create biomass reaction with biomass metabolites as reactants
            interacting (bool): If true models will be able to exchange metabolites. Otherwise all produced metabolites will go to sink
            exchanged_metabolites_blacklist (set): List of metabolites that can not be exchanged between species. This is done
             by separating 'pool' (uptake) compartment and 'pool' ('export') compartments for certain metabolites.
        """

        if not interacting and merge_extracellular_compartments:
            raise RuntimeError("Non-interacting models are not supported when merging extracellular compartment")

        self.id = community_id
        self._organisms = AttrOrderedDict()
        self._merge_extracellular_compartments = merge_extracellular_compartments
        self._create_biomass = create_biomass
        self._merged_model = None
        self._copy_models = copy_models
        self._interacting = interacting
        self._organisms_exchange_reactions = {}
        self._organisms_biomass_reactions = {}
        self._exchanged_metabolites_blacklist = set(exchanged_metabolites_blacklist)

        if models is not None:
            for model in models:
                self.add_organism(model, copy_models)

    @property
    def copy_models(self):
        """
        If true copies for merged models  are created

        Returns: bool
        """
        return self._copy_models

    @property
    def create_biomass_reaction(self):
        """
        Create biomass reaction with biomass metabolites as reactants

        Returns: bool
        """
        return self._create_biomass

    @create_biomass_reaction.setter
    def create_biomass_reaction(self, value):
        """
        Create biomass reaction with biomass metabolites as reactants

        Args:
            value: bool
        """
        self._clear_merged_model()
        self._create_biomass = value

    @property
    def size(self):
        return float(len(self._organisms))

    @property
    def interacting(self):
        """
        If true models will be able to exchange metabolites. Otherwise all produced metabolites will go to sink

        Returns: bool
        """
        return self._interacting

    @interacting.setter
    def interacting(self, value):
        """
        If true models will be able to exchange metabolites. Otherwise all produced metabolites will go to sink

        Args:
            value: bool
        """
        self._clear_merged_model()
        self._interacting = value

    @property
    def organisms_exchange_reactions(self):
        """
        Returns dictionary containing list of reactions exchanging model metabolites with common environment.
        Dictionary keys are model ids. Values are dictionaries with keys containing exchange reaction ids and values
        containing various information about these reactions.

        Returns: dict
        """
        if not self._merged_model:
            self._merged_model = self.generate_merged_model()

        return self._organisms_exchange_reactions

    @property
    def organisms_reactions(self):
        """
        Returns dictionary containing list of community organisms specific reactions

        Returns: dict
        """
        if not self._merged_model:
            self._merged_model = self.generate_merged_model()

        return self._organisms_reactions

    @property
    def organisms_biomass_reactions(self):
        """
        Returns dictionary containing reaction exporting biomass to common environment. Keys are model ids, and values
        are reaction ids

        Returns: dict
        """
        if not self._merged_model:
            self._merged_model = self.generate_merged_model()

        return self._organisms_biomass_reactions

    @property
    def merge_extracellular_compartments(self):
        """
        Do not create organism specific extracellular compartment

        Returns: bool
        """
        return self._merge_extracellular_compartments

    @merge_extracellular_compartments.setter
    def merge_extracellular_compartments(self, value):
        """
        Do not create organism specific extracellular compartment

        Args:
            value: bool
        """
        self._clear_merged_model()
        self._merge_extracellular_compartments = value

    @property
    def merged(self):
        """
        Merged models containing every organism as separate compartment

        Returns: CBModel
        """
        if not self._merged_model:
            self._merged_model = self.generate_merged_model()

        return self._merged_model

    @property
    def organisms(self):
        """
        Dictionary of organism models which are part of the community. Keys are model ids and values are models
        Returns: dict
        """
        return self._organisms

    def __str__(self):
        return '\n'.join(self._organisms.keys())

    def _clear_merged_model(self):
        self._merged_model = None
        self._organisms_exchange_reactions = {}
        self._organisms_reactions = {}

    def add_organism(self, model, copy=True):
        """ Add an organism to this community.

        Args:
            model (CBModel): model of the organism
            copy (bool): create a copy of the given model (default: True)

        """
        self._clear_merged_model()

        if model.id in self._organisms:
            warn("Organism '{}' is already in this community".format(model.id))
        else:
            if copy:
                model = model.copy()

            self._organisms[model.id] = model

    def remove_organism(self, organism):
        """ Remove an organism from this community

        Args:
            organism (str): organism id

        """
        self._clear_merged_model()

        if organism not in self._organisms:
            warn('Organism {} is not in this community'.format(organism))
        else:
            del self._organisms[organism]

    def generate_merged_model(self):
        def _id_pattern(object_id, organism_id):
            return "{}_{}".format(object_id, organism_id)

        def _name_pattern(object_name, organism_name):
            return "{} ({})".format(object_name, organism_name)

        def _copy_object(obj, org_id, compartment=None):
            new_obj = deepcopy(obj)
            new_obj.id = _id_pattern(obj.id, org_id)
            new_obj.name = _name_pattern(obj.name, org_id)
            if compartment:
                new_obj.compartment = compartment

            return new_obj

        models_missing_biomass = [m.id for m in self._organisms.values() if not m.biomass_reaction]
        if models_missing_biomass:
            raise RuntimeError("Biomass reaction not found in models: {}".format("', '".join(models_missing_biomass)))

        merged_model = CBModel(self.id)

        organisms_biomass_metabolites = {}
        community_metabolite_exchange_lookup = {}

        for org_id, model in self._organisms.items():
            self._organisms_reactions[org_id] = []
            self._organisms_exchange_reactions[org_id] = {}
            self._organisms_biomass_reactions[org_id] = {}
            exchanged_metabolites = {m_id for r_id in model.get_exchange_reactions()
                                     for m_id in model.reactions[r_id].stoichiometry}
            #
            # Create additional extracellular compartment
            #
            if not self._merge_extracellular_compartments:
                pool_compartment = Compartment('pool', 'common pool')
                merged_model.add_compartment(pool_compartment)
                export_pool_compartment = Compartment('pool_blacklist', 'blacklisted metabolite pool')
                merged_model.add_compartment(export_pool_compartment)

            for c_id, comp in model.compartments.items():
                if not comp.external or not self._merge_extracellular_compartments:
                    new_comp = _copy_object(comp, org_id)
                    merged_model.add_compartment(new_comp)
                elif c_id not in merged_model.compartments:
                    merged_model.add_compartment(deepcopy(comp))

            for m_id, met in model.metabolites.items():
                if not model.compartments[met.compartment].external or not self._merge_extracellular_compartments:
                    new_met = _copy_object(met, org_id, _id_pattern(met.compartment, org_id))
                    merged_model.add_metabolite(new_met)
                elif m_id not in merged_model.metabolites:
                    merged_model.add_metabolite(deepcopy(met))

                m_blacklisted = met.id in self._exchanged_metabolites_blacklist

                if met.id in exchanged_metabolites and not self._merge_extracellular_compartments:
                    #
                    # For blacklisted metabolites create a separate pool from which metabolites can not be reuptaken
                    #
                    if m_blacklisted and self._interacting:
                        pool_id = _id_pattern(m_id, "pool_blacklist")
                        if pool_id not in merged_model.metabolites:
                            new_met = _copy_object(met, "pool_blacklist", "pool_blacklist")
                            merged_model.add_metabolite(new_met)

                            exch_id = _id_pattern("R_EX_" + m_id, "pool_blacklist")
                            exch_name = _name_pattern(met.name, "pool (blacklist) exchange")
                            blk_rxn = CBReaction(exch_id, name=exch_name, reversible=False,
                                                 reaction_type=ReactionType.SINK)
                            blk_rxn.stoichiometry[pool_id] = -1.0
                            community_metabolite_exchange_lookup[new_met.id] = exch_id
                            merged_model.add_reaction(blk_rxn)

                    pool_id = _id_pattern(m_id, "pool")
                    if pool_id not in merged_model.metabolites:
                        new_met = _copy_object(met, "pool", "pool")
                        merged_model.add_metabolite(new_met)

                        exch_id = _id_pattern("R_EX_" + m_id, "pool")
                        exch_name = _name_pattern(met.name, "pool exchange")
                        new_rxn = CBReaction(exch_id, name=exch_name, reversible=True,
                                             reaction_type=ReactionType.EXCHANGE)
                        new_rxn.stoichiometry[pool_id] = -1.0
                        community_metabolite_exchange_lookup[new_met.id] = exch_id
                        merged_model.add_reaction(new_rxn)

            for r_id, rxn in model.reactions.items():

                is_exchange = rxn.reaction_type == ReactionType.EXCHANGE

                if not is_exchange or not self._merge_extracellular_compartments:
                    new_rxn = _copy_object(rxn, org_id)

                    for m_id, coeff in rxn.stoichiometry.items():
                        m_blacklisted = m_id in self._exchanged_metabolites_blacklist
                        if (not model.compartments[model.metabolites[m_id].compartment].external
                                or not self._merge_extracellular_compartments):
                            del new_rxn.stoichiometry[m_id]
                            new_id = _id_pattern(m_id, org_id)
                            new_rxn.stoichiometry[new_id] = coeff

                        if is_exchange:
                            new_rxn.reaction_type = ReactionType.OTHER
                            if (model.compartments[model.metabolites[m_id].compartment].external
                                    and not self._merge_extracellular_compartments):
                                # TODO: if m_id in self._exchanged_metabolites_blacklist:
                                pool_id = _id_pattern(m_id, "pool")
                                new_rxn.stoichiometry[pool_id] = -coeff
                                cnm = CommunityNameMapping(
                                    organism_reaction=new_rxn.id,
                                    original_reaction=r_id,
                                    organism_metabolite=new_id,
                                    extracellular_metabolite=pool_id,
                                    original_metabolite=m_id,
                                    community_exchange_reaction=community_metabolite_exchange_lookup[pool_id])
                                self._organisms_exchange_reactions[org_id][new_rxn.id] = cnm

                                if not self.interacting:
                                    sink_rxn = CBReaction('Sink_{}'.format(new_id), reaction_type=ReactionType.SINK,
                                                          reversible=False)
                                    sink_rxn.stoichiometry = {new_id: -1}
                                    sink_rxn.lb = 0.0
                                    merged_model.add_reaction(sink_rxn)
                                elif m_blacklisted:
                                    pool_blacklist_id = _id_pattern(m_id, "pool_blacklist")
                                    blacklist_export_rxn = CBReaction('R_EX_BLACKLIST_{}'.format(new_id),
                                                                      reaction_type=ReactionType.OTHER,
                                                                      reversible=False)
                                    blacklist_export_rxn.stoichiometry = {new_id: -1, pool_blacklist_id: 1}
                                    blacklist_export_rxn.lb = 0.0
                                    merged_model.add_reaction(blacklist_export_rxn)

                    if is_exchange and not self._merge_extracellular_compartments:
                        new_rxn.reversible = True
                        new_rxn.lb = -inf
                        new_rxn.ub = inf if self.interacting and not m_blacklisted else 0.0

                    if rxn.id == model.biomass_reaction:
                        new_rxn.reversible = False

                    if self._create_biomass and rxn.id == model.biomass_reaction:
                        new_rxn.objective = False

                        # Add biomass metabolite to biomass equation
                        m_id = _id_pattern('Biomass', org_id)
                        name = _name_pattern('Community biomass', org_id)
                        comp = 'pool'
                        biomass_met = Metabolite(m_id, name, comp)
                        merged_model.add_metabolite(biomass_met)
                        new_rxn.stoichiometry[m_id] = 1
                        organisms_biomass_metabolites[org_id] = m_id

                        sink_rxn = CBReaction('Sink_biomass_{}'.format(org_id), reaction_type=ReactionType.SINK,
                                              reversible=False)
                        sink_rxn.stoichiometry = {m_id: -1}
                        sink_rxn.lb = 0.0
                        merged_model.add_reaction(sink_rxn)

                    self._organisms_reactions[org_id].append(new_rxn.id)
                    merged_model.add_reaction(new_rxn)

                else:
                    if is_exchange and self._merge_extracellular_compartments:
                        self._organisms_exchange_reactions[org_id][rxn.id] = CommunityNameMapping(
                            organism_reaction=r_id,
                            original_reaction=r_id,
                            extracellular_metabolite=list(rxn.stoichiometry.keys())[0],
                            original_metabolite=list(rxn.stoichiometry.keys())[0],
                            organism_metabolite=None)
                        self._organisms_reactions[org_id].append(rxn.id)

                    if r_id in merged_model.reactions:
                        continue

                    new_rxn = deepcopy(rxn)
                    new_rxn.reaction_type = ReactionType.EXCHANGE
                    if rxn.id == model.biomass_reaction and self._create_biomass:
                        new_rxn.reversible = False
                        new_rxn.objective = False

                        m_id = _id_pattern('Biomass', org_id)
                        name = _name_pattern('Biomass', org_id)
                        comp = 'pool'
                        biomass_met = Metabolite(m_id, name, comp)
                        merged_model.add_metabolite(biomass_met)
                        new_rxn.stoichiometry[m_id] = 1
                        organisms_biomass_metabolites[org_id] = m_id

                    merged_model.add_reaction(new_rxn)

                if r_id == model.biomass_reaction:
                    self._organisms_biomass_reactions[org_id] = new_rxn.id

        if self._create_biomass:
            biomass_rxn = CBReaction('R_Community_Growth', name="Community Growth",
                                     reversible=False, reaction_type=ReactionType.SINK, objective=1.0)
            for org_biomass in organisms_biomass_metabolites.values():
                biomass_rxn.stoichiometry[org_biomass] = -1

            merged_model.add_reaction(biomass_rxn)
            merged_model.biomass_reaction = biomass_rxn.id

        return merged_model

    def copy(self, merge_extracellular_compartments=None, copy_models=None, interacting=None, create_biomass=None,
             exchanged_metabolites_blacklist=None):
        """
        Copy model object
        Args:
            copy_models (bool): If true copies for merged models  are created
            interacting (bool): If true models will be able to exchange metabolites. Otherwise all produced metabolites will go to sink
            create_biomass (bool): create biomass reaction with biomass metabolites as reactants
        Returns:
            Community
        """
        if copy_models is None:
            copy_models = self._copy_models

        if interacting is None:
            interacting = self._interacting

        if create_biomass is None:
            create_biomass = self._create_biomass

        if merge_extracellular_compartments is None:
            merge_extracellular_compartments = self._merge_extracellular_compartments

        if exchanged_metabolites_blacklist is None:
            exchanged_metabolites_blacklist = self._exchanged_metabolites_blacklist

        copy_community = Community(self.id, models=list(self._organisms.values()),
                                   copy_models=copy_models, create_biomass=create_biomass,
                                   merge_extracellular_compartments=merge_extracellular_compartments,
                                   interacting=interacting,
                                   exchanged_metabolites_blacklist=exchanged_metabolites_blacklist)

        return copy_community

    def split_fluxes(self, fluxes):
        """ Decompose a flux balance solution of the merged community into organism-specific flux vectors.

        Args:
            fluxes (dict): flux distribution as a single dict

        Returns:
            dict: community flux distribution as a nested dict
        """

        comm_fluxes = OrderedDict()

        for org_id, model in self._organisms.items():
            org_fluxes = [(r_id[:-(1 + len(org_id))], val) for r_id, val in fluxes.items() if r_id.endswith(org_id)]
            comm_fluxes[org_id] = OrderedDict(org_fluxes)

        return comm_fluxes

status_mapping = {
    'optimal': Status.OPTIMAL,
    'unbounded': Status.UNBOUNDED,
    'infeasible': Status.INFEASIBLE,
    'infeasible_or_unbounded': Status.INF_OR_UNB,
    'suboptimal': Status.SUBOPTIMAL,
}

class Status(Enum):
    """ Enumeration of possible solution status. """
    OPTIMAL = 'Optimal'
    UNKNOWN = 'Unknown'
    SUBOPTIMAL = 'Suboptimal'
    UNBOUNDED = 'Unbounded'
    INFEASIBLE = 'Infeasible'
    INF_OR_UNB = 'Infeasible or Unbounded'
    
    
class Solution(object):
    """ Stores the results of an optimization.

    Instantiate without arguments to create an empty Solution representing a failed optimization.
    """

    def __init__(self, status=Status.UNKNOWN, message=None, fobj=None, values=None, shadow_prices=None, reduced_costs=None):
        self.status = status
        self.message = message
        self.fobj = fobj
        self.values = values
        self.shadow_prices = shadow_prices
        self.reduced_costs = reduced_costs

    def __str__(self):
        return f"Objective: {self.fobj}\nStatus: {self.status.value}\n"

    def __repr__(self):
        return str(self)

    def show_values(self, pattern=None, sort=False, abstol=1e-9):
        """ Show solution results.

        Arguments:
            pattern (str): show only reactions that contain pattern (optional)
            sort (bool): sort values by magnitude (default: False)
            abstol (float): abstolute tolerance to hide null values (default: 1e-9)

        Returns:
            str: printed table with variable values
        """

        if self.values:
            print_values(self.values, pattern=pattern, sort=sort, abstol=abstol)

    def show_shadow_prices(self, pattern=None, sort=False, abstol=1e-9):
        """ Show shadow prices.

        Arguments:
            pattern (str): show only metabolites that contain pattern (optional)
            sort (bool): sort values by magnitude (default: False)
            abstol (float): abstolute tolerance to hide null values (default: 1e-9)

        Returns:
            str: printed table with shadow prices
        """

        if self.shadow_prices:
            print_values(self.shadow_prices, pattern=pattern, sort=sort, abstol=abstol)

    def show_reduced_costs(self, pattern=None, sort=False, abstol=1e-9):
        """ Show reduced costs.

        Arguments:
            pattern (str): show only reactions that contain pattern (optional)
            sort (bool): sort values by magnitude (default: False)
            abstol (float): abstolute tolerance to hide null values (default: 1e-9)

        Returns:
            str: printed table with shadow prices
        """

        if self.reduced_costs:
            print_values(self.reduced_costs, pattern=pattern, sort=sort, abstol=abstol)

    def show_metabolite_balance(self, m_id, model, sort=False, percentage=False, equations=False, abstol=1e-9):
        """ Show metabolite balance details.

        Arguments:
            m_id (str): metabolite id
            model (CBModel): model that generated the solution
            sort (bool): sort values by magnitude (default: False)
            percentage (bool): show percentage of total turnover instead of flux (default: False)
            equations (bool): show reaction equations (default: False)
            abstol (float): abstolute tolerance to hide null values (default: 1e-9)

        Returns:
            str: formatted output
        """

        if self.values:
            print_balance(self.values, m_id, model, sort=sort, percentage=percentage, equations=equations, abstol=abstol)

    def get_metabolites_turnover(self, model):
        """ Calculate metabolite turnover.

        Arguments:
            model (CBModel): model that generated the solution

        Returns:
            dict: metabolite turnover rates
        """

        if not self.values:
            return None

        m_r_table = model.metabolite_reaction_lookup()
        t = {m_id: 0.5*sum([abs(coeff * self.values[r_id]) for r_id, coeff in neighbours.items()])
             for m_id, neighbours in m_r_table.items()}
        return t

    def show_metabolite_turnover(self, model, pattern=None, sort=False, abstol=1e-9):
        """ Show solution results.

        Arguments:
            model (CBModel): model that generated the solution
            pattern (str): show only reactions that contain pattern (optional)
            sort (bool): sort values by magnitude (default: False)
            abstol (float): abstolute tolerance to hide null values (default: 1e-9)

        Returns:
            str: printed table
        """

        if self.values:
            turnover = self.get_metabolites_turnover(model)
            print_values(turnover, pattern=pattern, sort=sort, abstol=abstol)

    def to_dataframe(self):
        """ Convert reaction fluxes to *pandas.DataFrame*

        Returns:
            pandas.DataFrame: flux values
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("Pandas is not installed.")

        return pd.DataFrame(self.values.values(), columns=["value"], index=self.values.keys())


class OptLangSolver(Solver):
    """ Implements the gurobi solver interface. """

    def __init__(self, model=None):
        Solver.__init__(self)
        self.problem = Model()

        self.parameter_mapping = {
            Parameter.TIME_LIMIT: self.problem.configuration.timeout,
            Parameter.FEASIBILITY_TOL: self.problem.configuration.tolerances.feasibility,
            Parameter.OPTIMALITY_TOL: self.problem.configuration.tolerances.optimality,
            Parameter.INT_FEASIBILITY_TOL: self.problem.configuration.tolerances.integrality,
        }

        self.set_parameters(default_parameters)
        self.set_logging(False)

        if model:
            self.build_problem(model)

    def add_variable(self, var_id, lb=-inf, ub=inf, vartype=VarType.CONTINUOUS, update=True):
        """ Add a variable to the current problem.

        Arguments:
            var_id (str): variable identifier
            lb (float): lower bound
            ub (float): upper bound
            vartype (VarType): variable type (default: CONTINUOUS)
            update (bool): update problem immediately (default: True)
        """

        if var_id in self.var_ids:
            var = self.problem.variables[var_id]
            var.lb = lb
            var.ub = ub
            var.type = vartype.value
        else:
            var = Variable(var_id, lb=lb, ub=ub, type=vartype.value)
            self.problem.add(var)
            self.var_ids.append(var_id)

        if update:
            self.problem.update()

    def add_constraint(self, constr_id, lhs, sense='=', rhs=0, update=True):
        """ Add a constraint to the current problem.

        Arguments:
            constr_id (str): constraint identifier
            lhs (dict): variables and respective coefficients
            sense (str): constraint sense (any of: '<', '=', '>'; default '=')
            rhs (float): right-hand side of equation (default: 0)
            update (bool): update problem immediately (default: True)
        """

        if constr_id in self.constr_ids:
            self.problem.remove(constr_id)

        if sense == '=':
            constr = Constraint(Zero, lb=rhs, ub=rhs, name=constr_id)
        elif sense == '>':
            constr = Constraint(Zero, lb=rhs, name=constr_id)
        elif sense == '<':
            constr = Constraint(Zero, ub=rhs, name=constr_id)
        else:
            raise RuntimeError(f"Invalid constraint direction: {sense}")

        self.problem.add(constr)
        self.constr_ids.append(constr_id)

        expr = {self.problem.variables[r_id]: coeff for r_id, coeff in lhs.items() if coeff}
        self.problem.constraints[constr_id].set_linear_coefficients(expr)

        if update:
            self.problem.update()

    def remove_variable(self, var_id):
        """ Remove a variable from the current problem.

        Arguments:
            var_id (str): variable identifier
        """
        self.remove_variables([var_id])

    def remove_variables(self, var_ids):
        """ Remove variables from the current problem.

        Arguments:
            var_ids (list): variable identifiers
        """

        for var_id in var_ids:
            if var_id in self.var_ids:
                self.problem.remove(var_id)
                self.var_ids.remove(var_id)

    def remove_constraint(self, constr_id):
        """ Remove a constraint from the current problem.

        Arguments:
            constr_id (str): constraint identifier
        """
        self.remove_constraints([constr_id])

    def remove_constraints(self, constr_ids):
        """ Remove constraints from the current problem.

        Arguments:
            constr_ids (list): constraint identifiers
        """

        for constr_id in constr_ids:
            if constr_id in self.constr_ids:
                self.problem.remove(constr_id)
                self.constr_ids.remove(constr_id)

    def set_objective(self, linear=None, quadratic=None, minimize=True):
        """ Set a predefined objective for this problem.

        Args:
            linear (dict): linear coefficients (optional)
            quadratic (dict): quadratic coefficients (optional)
            minimize (bool): solve a minimization problem (default: True)

        Notes:
            Setting the objective is optional. It can also be passed directly when calling **solve**.

        """

        if linear is None:
            linear = {}

        if quadratic is None:
            quadratic = {}

        if linear and not quadratic:
            objective = {}

            if isinstance(linear, str):
                objective = {self.problem.variables[linear]: 1}
                if linear not in self.var_ids:
                    warn(f"Objective variable not previously declared: {linear}")
            else:
                for r_id, val in linear.items():
                    if r_id not in self.var_ids:
                        warn(f"Objective variable not previously declared: {r_id}")
                    elif val != 0:
                        objective[self.problem.variables[r_id]] = val

            self.problem.objective = Objective(Zero, direction=('min' if minimize else 'max'), sloppy=True)
            self.problem.objective.set_linear_coefficients(objective)
        else:
            objective = []

            for r_id, val in linear.items():
                if r_id not in self.var_ids:
                    warn(f"Objective variable not previously declared: {r_id}")
                elif val != 0:
                    objective.append(val * self.problem.variables[r_id])

            for (r_id1, r_id2), val in quadratic.items():
                if r_id1 not in self.var_ids:
                    warn(f"Objective variable not previously declared: {r_id1}")
                elif r_id2 not in self.var_ids:
                    warn(f"Objective variable not previously declared: {r_id2}")
                elif val != 0:
                    objective.append(val * self.problem.variables[r_id1] * self.problem.variables[r_id2])

            objective_expr = add(objective)
            self.problem.objective = Objective(objective_expr, direction=('min' if minimize else 'max'), sloppy=True)

    def solve(self, linear=None, quadratic=None, minimize=None, model=None, constraints=None, get_values=True,
              shadow_prices=False, reduced_costs=False, pool_size=0, pool_gap=None):
        """ Solve the optimization problem.

        Arguments:
            linear (str or dict): linear coefficients (or a single variable to optimize)
            quadratic (dict): quadratic objective (optional)
            minimize (bool): solve a minimization problem (default: True)
            model (CBModel): model (optional, leave blank to reuse previous model structure)
            constraints (dict): additional constraints (optional)
            get_values (bool or list): set to false for speedup if you only care about the objective (default: True)
            shadow_prices (bool): return shadow prices if available (default: False)
            reduced_costs (bool): return reduced costs if available (default: False)
            pool_size (int): calculate solution pool of given size (only for MILP problems)
            pool_gap (float): maximum relative gap for solutions in pool (optional)

        Returns:
            Solution: solution
        """

        if model:
            self.build_problem(model)

        problem = self.problem

        if constraints:
            old_constraints = {}
            for r_id, x in constraints.items():
                lb, ub = x if isinstance(x, tuple) else (x, x)
                if r_id in self.var_ids:
                    lpvar = problem.variables[r_id]
                    old_constraints[r_id] = (lpvar.lb, lpvar.ub)
                    lpvar.lb, lpvar.ub = lb, ub
                else:
                    warn(f"Constrained variable '{r_id}' not previously declared")
            problem.update()

        self.set_objective(linear, quadratic, minimize)

        # run the optimization
        if pool_size > 1:
            raise RuntimeError("OptLang interface does not support solution pools.")

        problem.optimize()

        status = status_mapping.get(problem.status, Status.UNKNOWN)
        message = str(problem.status)

        if status == Status.OPTIMAL:
            fobj = problem.objective.value
            values, s_prices, r_costs = None, None, None

            if get_values:
                values = dict(problem.primal_values)

                if isinstance(get_values, Iterable):
                    values = {x: values[x] for x in get_values}

            if shadow_prices:
                s_prices = dict(problem.shadow_prices)

            if reduced_costs:
                r_costs = dict(problem.reduced_costs)

            solution = Solution(status, message, fobj, values, s_prices, r_costs)
        else:
            solution = Solution(status, message)

        # restore values of temporary constraints
        if constraints:
            for r_id, (lb, ub) in old_constraints.items():
                lpvar = problem.variables[r_id]
                lpvar.lb, lpvar.ub = lb, ub
            problem.update()

        return solution

    def set_parameter(self, parameter, value):
        """ Set a parameter value for this optimization problem

        Arguments:
            parameter (Parameter): parameter type
            value (float): parameter value
        """

        if parameter in self.parameter_mapping:
            self.parameter_mapping[parameter] = value
        else:
            raise RuntimeError('Parameter unknown (or not yet supported).')

    def set_logging(self, enabled=False):
        """ Enable or disable log output:

        Arguments:
            enabled (bool): turn logging on (default: False)
        """

        self.problem.configuration.verbosity = 3 if enabled else 0

    def write_to_file(self, filename):
        """ Write problem to file:

        Arguments:
            filename (str): file path
        """

        with open(filename, "w") as f:
            f.write(self.problem.to_lp())


class ReactionParser(object):

    def __init__(self):
        id_re = '[a-zA-Z]\w*'
        pos_float_re = '\d+(?:\.\d+)?(?:e[+-]?\d+)?'
        float_re = '-?\d+(?:\.\d+)?(?:e[+-]?\d+)?'

        compound = '(?:' + pos_float_re + '\s+)?' + id_re
        expression = compound + '(?:\s*\+\s*' + compound + ')*'
        bounds = '\[\s*(?P<lb>' + float_re + ')?\s*,\s*(?P<ub>' + float_re + ')?\s*\]'
        objective = '@' + float_re
        reaction = '^(?P<reaction_id>' + id_re + ')\s*:' + \
                   '\s*(?P<substrates>' + expression + ')?' + \
                   '\s*(?P<direction>-->|<->)' + \
                   '\s*(?P<products>' + expression + ')?' + \
                   '\s*(?P<bounds>' + bounds + ')?' + \
                   '\s*(?P<objective>' + objective + ')?$'

        self.regex_compound = re.compile('(?P<coeff>' + pos_float_re + '\s+)?(?P<met_id>' + id_re + ')')
        self.regex_bounds = re.compile(bounds)
        self.regex_reaction = re.compile(reaction)

    def parse_reaction(self, reaction_str, kind=None):
        match = self.regex_reaction.match(reaction_str)

        if not match:
            raise SyntaxError('Unable to parse: ' + reaction_str)

        r_id = match.group('reaction_id')
        reversible = match.group('direction') == '<->'
        substrates = match.group('substrates')
        products = match.group('products')

        stoichiometry = OrderedDict()

        if substrates:
            left_coeffs = self.parse_coefficients(substrates, sense=-1)
            stoichiometry.update(left_coeffs)

        if products:
            right_coeffs = self.parse_coefficients(products, sense=1)
            for m_id, val in right_coeffs:
                if m_id in stoichiometry:
                    new_val = val + stoichiometry[m_id]
                    stoichiometry[m_id] = new_val
                else:
                    stoichiometry[m_id] = val

        if kind is None:
            return r_id, reversible, stoichiometry

        if kind == 'cb':
            bounds = match.group('bounds')
            lb, ub = self.parse_bounds(bounds, reversible)
            objective = match.group('objective')
            obj_coeff = float(objective[1:]) if objective else 0
            return r_id, reversible, stoichiometry, lb, ub, obj_coeff

    def parse_coefficients(self, expression, sense):
        coefficients = []
        terms = expression.split('+')

        for term in terms:
            match = self.regex_compound.match(term.strip())
            coeff = sense * float(match.group('coeff')) if match.group('coeff') else sense
            m_id = match.group('met_id')
            coefficients.append((m_id, coeff))

        return coefficients

    def parse_bounds(self, expression, reversible):
        lb = -inf if reversible else 0.0
        ub = inf

        if expression:
            match = self.regex_bounds.match(expression)
            if match.group('lb'):
                lb = float(match.group('lb'))
            if match.group('ub'):
                ub = float(match.group('ub'))

        return lb, ub
    

class Compartment(object):
    """ Base class for modeling compartments. """

    def __init__(self, comp_id, name=None, external=False, size=1.0):
        """
        Arguments:
            comp_id (str): a valid unique identifier
            name (str): compartment name (optional)
            external (bool): is external (default: false)
            size (float): compartment size (default: 1.0)
        """
        self.id = comp_id
        self.name = name if name is not None else comp_id
        self.size = size
        self.external = external
        self.metadata = OrderedDict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
    
    
    

class Metabolite(object):
    """ Base class for modeling metabolites. """

    def __init__(self, met_id, name=None, compartment=None):
        """
        Arguments:
            met_id (str): a valid unique identifier
            name (str): common metabolite name
            compartment (str): compartment containing the metabolite
        """
        self.id = met_id
        self.name = name if name is not None else met_id
        self.compartment = compartment
        self.metadata = OrderedDict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class ReactionType(Enum):
    """ Enumeration of possible reaction types. """
    ENZYMATIC = 'enzymatic'
    TRANSPORT = 'transport'
    EXCHANGE = 'exchange'
    SINK = 'sink'
    OTHER = 'other'
    
class Reaction(object):
    """ Base class for modeling reactions. """

    def __init__(self, reaction_id, name=None, reversible=True, stoichiometry=None, regulators=None,
                 reaction_type=None):
        """
        Arguments:
            reaction_id (str): a valid unique identifier
            name (str): common reaction name
            reversible (bool): reaction reversibility (default: True)
            stoichiometry (dict): stoichiometry
            regulators (dict): reaction regulators
            reaction_type (ReactionType): reaction type
        """
        self.id = reaction_id
        self.name = name if name is not None else reaction_id
        self.reversible = reversible
        self.reaction_type = reaction_type if reaction_type is not None else ReactionType.OTHER
        self.stoichiometry = OrderedDict()
        self.regulators = OrderedDict()
        self.metadata = OrderedDict()

        if stoichiometry:
            self.stoichiometry.update(stoichiometry)
        if regulators:
            self.regulators.update(regulators)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def get_substrates(self):
        """ Get list of reaction substrates

        Returns:
            list: reaction substrates
        """

        return [m_id for m_id, coeff in self.stoichiometry.items() if coeff < 0]

    def get_products(self):
        """ Get list of reaction products

        Returns:
            list: reaction products
        """

        return [m_id for m_id, coeff in self.stoichiometry.items() if coeff > 0]

    def get_activators(self):
        """ Get list of reaction activators

        Returns:
            list: reaction activators
        """

        return [m_id for m_id, kind in self.regulators.items() if kind == RegulatorType.ACTIVATOR]

    def get_inhibitors(self):
        """ Get list of reaction inhibitors

        Returns:
            list: reaction inhibitors
        """

        return [m_id for m_id, kind in self.regulators.items() if kind == RegulatorType.INHIBITOR]

    def to_equation(self, metabolite_names=None):
        """ Returns reaction equation string

        Arguments:
            metabolite_names (dict): replace metabolite id's with names (optional)

        Returns:
            str: reaction string
        """

        if metabolite_names:
            def met_repr(m_id):
                return metabolite_names[m_id]
        else:
            def met_repr(m_id):
                return m_id

        left = ' + '.join(met_repr(m_id) if coeff == -1.0 else str(-coeff) + ' ' + met_repr(m_id)
                          for m_id, coeff in self.stoichiometry.items() if coeff < 0)
        arrow = '<->' if self.reversible else '-->'
        right = ' + '.join(met_repr(m_id) if coeff == 1.0 else str(coeff) + ' ' + met_repr(m_id)
                           for m_id, coeff in self.stoichiometry.items() if coeff > 0)
        return f"{left} {arrow} {right}"

    def to_string(self, metabolite_names=None):
        """ Returns reaction as a string

        Arguments:
            metabolite_names (dict): replace metabolite id's with names (optional)

        Returns:
            str: reaction string
        """
        return self.id + ': ' + self.to_equation(metabolite_names=metabolite_names)


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


class Model(object):
    """ Base class for all metabolic models."""

    def __init__(self, model_id):
        """
        Arguments:
            model_id (str): a valid unique identifier
        """
        self.id = model_id
        self.metabolites = AttrOrderedDict()
        self.reactions = AttrOrderedDict()
        self.compartments = AttrOrderedDict()
        self.metadata = OrderedDict()
        self._m_r_lookup = None
        self._reg_lookup = None
        self._s_matrix = None
        self._parser = None
        self._needs_update = False

    def copy(self):
        return deepcopy(self)

    def update(self):
        self._m_r_lookup = None
        self._reg_lookup = None
        self._s_matrix = None
        self._needs_update = False

    def add_compartment(self, compartment, replace=True):
        """ Add a compartment to the model.

        Arguments:
            compartment (Compartment): compartment to add
            replace (bool): replace previous compartment with same id (default: True)
        """
        if compartment.id in self.compartments and not replace:
            raise RuntimeError(f"Compartment {compartment.id} already exists.")
        self.compartments[compartment.id] = compartment

    def add_metabolite(self, metabolite, replace=True):
        """ Add a metabolite to the model.

        Arguments:
            metabolite (Metabolite): metabolite to add
            replace (bool): replace previous metabolite with same id (default: True)
        """

        if metabolite.id in self.metabolites and not replace:
            raise RuntimeError(f"Metabolite {metabolite.id} already exists.")

        if metabolite.compartment not in self.compartments:
            raise RuntimeError(f"Metabolite {metabolite.id} has invalid compartment {metabolite.compartment}.")

        self.metabolites[metabolite.id] = metabolite
        self._needs_update = True

    def add_reaction(self, reaction, replace=True):
        """ Add a reaction to the model.

        Arguments:
            reaction (Reaction): reaction to add
            replace (bool): replace previous reaction with same id (default: True)
        """
        if reaction.id in self.reactions and not replace:
            raise RuntimeError(f"Reaction {reaction.id} already exists.")
        self.reactions[reaction.id] = reaction
        self._needs_update = True

    def add_reaction_from_str(self, reaction_str, compartment=None):
        """ Parse a reaction from a string and add it to the model.

        Arguments:
            reaction_str (str): string representation a the reaction
            compartment (str): reaction compartment id (optional)

        Notes:
            If the metabolites specified in the reaction are not yet in the model, they will be automatically added.
            If the compartment id is not given, it will use the first available compartment.
        """

        if not self._parser:
            self._parser = ReactionParser()

        if compartment is None:
            compartment = list(self.compartments.keys())[0]

        r_id, reversible, stoichiometry = self._parser.parse_reaction(reaction_str)

        for m_id in stoichiometry:
            if m_id not in self.metabolites:
                self.add_metabolite(Metabolite(m_id, m_id, compartment=compartment))

        reaction = Reaction(r_id, r_id, reversible, stoichiometry)
        self.add_reaction(reaction)
        self._needs_update = True

        return r_id

    def get_reactions_by_type(self, reaction_type):
        return [rxn.id for rxn in self.reactions.values() if rxn.reaction_type == reaction_type]

    def get_exchange_reactions(self):
        return self.get_reactions_by_type(ReactionType.EXCHANGE)

    def get_compartment_metabolites(self, c_id):
        if c_id not in self.compartments.keys():
            raise RuntimeError(f"No such compartment: {c_id}")

        return [m_id for m_id, met in self.metabolites.items() if met.compartment == c_id]

    def get_external_metabolites(self, from_reactions=False):
        # TODO: a unit test should assert that result is the same from reactions and from compartments

        if from_reactions:
            external = [m_id for r_id in self.get_exchange_reactions()
                        for m_id in self.reactions[r_id].stoichiometry]
        else:
            external = [m_id for m_id, met in self.metabolites.items()
                        if self.compartments[met.compartment].external]
        return external

    def get_reaction_compartments(self, r_id):
        return {self.metabolites[m_id].compartment for m_id in self.reactions[r_id].stoichiometry}

    def get_metabolite_producers(self, m_id, reversible=False):
        """ Return the list of reactions producing a given metabolite

        Arguments:
            m_id (str): metabolite id
            reversible (bool): also include reversible consumers

        Returns:
            list: producing reactions
        """
        table = self.metabolite_reaction_lookup()

        producers = []
        for r_id, coeff in table[m_id].items():
            if coeff > 0 or reversible and self.reactions[r_id].reversible:
                producers.append(r_id)

        return producers

    def get_metabolite_consumers(self, m_id, reversible=False):
        """ Return the list of reactions consuming a given metabolite

        Arguments:
            m_id (str): metabolite id
            reversible (bool): also include reversible producers

        Returns:
            list: consuming reactions
        """
        table = self.metabolite_reaction_lookup()

        consumers = []
        for r_id, coeff in table[m_id].items():
            if coeff < 0 or reversible and self.reactions[r_id].reversible:
                consumers.append(r_id)

        return consumers

    def get_metabolite_reactions(self, m_id):
        """ Return the list of reactions associated with a given metabolite

        Arguments:
            m_id (str): metabolite id

        Returns:
            list: associated reactions
        """
        table = self.metabolite_reaction_lookup()

        return list(table[m_id].keys())

    def get_activation_targets(self, m_id):
        table = self.regulatory_lookup()
        return [r_id for r_id, kind in table[m_id].items() if kind == RegulatorType.ACTIVATOR]

    def get_inhibition_targets(self, m_id):
        table = self.regulatory_lookup()
        return [r_id for r_id, kind in table[m_id].items() if kind == RegulatorType.INHIBITOR]

    def remove_compartment(self, c_id):
        """ Remove a compartment from the model.

        Arguments:
            c_id (str): compartment id
        """
        self.remove_compartments([c_id])

    def remove_compartments(self, c_ids):
        """ Remove a compartment from the model.

        Arguments:
            c_ids (list): compartment ids
        """

        for c_id in c_ids:
            if c_id in self.compartments:
                del self.compartments[c_id]
            else:
                warn(f"No such compartment {c_id}")

        metabolites = [m_id for m_id, met in self.metabolites.items() if met.compartment in c_ids]
        self.remove_metabolites(metabolites)

    def remove_metabolite(self, m_id):
        """ Remove a metabolite from the model.

        Arguments:
            m_id (str): metabolite id
        """
        self.remove_metabolites([m_id])

    def remove_metabolites(self, id_list, safe_delete=True):
        """ Remove a list of metabolites from the model.

        Arguments:
            id_list (list): metabolite ids
            safe_delete (bool): also remove metabolites from reactions (default: True)
        """

        if safe_delete:
            m_r_lookup = self.metabolite_reaction_lookup()
            reactions = set()

        for m_id in list(id_list):
            if m_id in self.metabolites:
                del self.metabolites[m_id]
            else:
                warn(f"No such metabolite {m_id}")

            if safe_delete:
                for r_id in m_r_lookup[m_id]:
                    del self.reactions[r_id].stoichiometry[m_id]
                    reactions.add(r_id)

        if safe_delete:
            to_delete = [r_id for r_id in reactions if len(self.reactions[r_id].stoichiometry) == 0]
            self.remove_reactions(to_delete)

        self._needs_update = True

    def remove_reaction(self, r_id):
        """ Remove a reaction from the model.

        Arguments:
            r_id (str): reaction id
        """
        self.remove_reactions([r_id])

    def remove_reactions(self, id_list):
        """ Remove a list of reactions from the model.

        Arguments:
            id_list (list of str): reaction ids
        """
        for r_id in id_list:
            if r_id in self.reactions:
                del self.reactions[r_id]
            else:
                warn(f"No such reaction {r_id}")
        self._needs_update = True

    def search_metabolites(self, pattern, by_name=False, ignore_case=False):
        """ Search metabolites in model.

        Arguments:
            pattern (str): regular expression pattern
            by_name (bool): search by metabolite name instead of id (default: False)
            ignore_case (bool): case-insensitive search (default: False)
        """

        re_expr = re.compile(pattern, flags=re.IGNORECASE) if ignore_case else re.compile(pattern)

        if by_name:
            return [m_id for m_id, met in self.metabolites.items() if re_expr.search(met.name) is not None]
        else:
            return [m_id for m_id in self.metabolites if re_expr.search(m_id) is not None]

    def search_reactions(self, pattern, by_name=False, ignore_case=False):
        """ Search reactions in model.

        Arguments:
            pattern (str): regular expression pattern
            by_name (bool): search by reaction name (case insensitive) instead of id (default: False)
            ignore_case (bool): case-insensitive search (default: False)
        """

        re_expr = re.compile(pattern, flags=re.IGNORECASE) if ignore_case else re.compile(pattern)

        if by_name:
            return [r_id for r_id, rxn in self.reactions.items() if re_expr.search(rxn.name) is not None]
        else:
            return [r_id for r_id in self.reactions if re_expr.search(r_id) is not None]

    def metabolite_reaction_lookup(self):
        if not self._m_r_lookup or self._needs_update:
            self._m_r_lookup = {m_id: {} for m_id in self.metabolites}

            for r_id, reaction in self.reactions.items():
                for m_id, coeff in reaction.stoichiometry.items():
                    self._m_r_lookup[m_id][r_id] = coeff

        return self._m_r_lookup

    def regulatory_lookup(self):
        if not self._reg_lookup or self._needs_update:
            self._reg_lookup = {m_id: {} for m_id in self.metabolites}

            for r_id, reaction in self.reactions.items():
                for m_id, kind in reaction.regulators.items():
                    self._reg_lookup[m_id][r_id] = kind

        return self._reg_lookup

    def stoichiometric_matrix(self):
        """ Return a stoichiometric matrix (as a nested list)

        Returns:
            list: stoichiometric matrix
        """

        if not self._s_matrix or self._needs_update:
            self._s_matrix = [[reaction.stoichiometry[m_id] if m_id in reaction.stoichiometry else 0
                               for reaction in self.reactions.values()]
                              for m_id in self.metabolites]

        return self._s_matrix

    def print_reaction(self, r_id, use_names=False):
        """ Print a reaction to a text based representation.

        Arguments:
            r_id (str): reaction id
            use_names (bool): print metabolite names instead of ids (default: False)

        Returns:
            str: reaction string
        """

        if use_names:
            metabolite_names = {m_id: met.name for m_id, met in self.metabolites.items()}
        else:
            metabolite_names = None

        print(self.reactions[r_id].to_string(metabolite_names))

    def to_string(self, use_names=False):
        """ Print the model to a text based representation.

        Arguments:
            use_names (bool): print metabolite names instead of ids (default: False)

        Returns:
            str: model as a string
        """

        if use_names:
            metabolite_names = {m_id: met.name for m_id, met in self.metabolites.items()}
        else:
            metabolite_names = None

        return '\n'.join(rxn.to_string(metabolite_names) for rxn in self.reactions.values())

    def summary(self):
        print("Metabolites:")
        for c_id in self.compartments:
            print(c_id, len(self.get_compartment_metabolites(c_id)))

        print("\nReactions:")
        for rxn_type in ReactionType:
            print(rxn_type.value, len(self.get_reactions_by_type(rxn_type)))

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    
    

class Community(object):
    def __init__(self, community_id, models, copy_models=False):
        self.id = community_id
        self.organisms = AttrOrderedDict()
        self._merged_model = None
        self.reaction_map = None
        self.metabolite_map = None

        model_ids = {model.id for model in models}

        if len(model_ids) < len(models):
            warn("Model ids are not unique, repeated models will be discarded.")

        for model in models:
            self.organisms[model.id] = model.copy() if copy_models else model

    def size(self):
        return len(self.organisms)

    @property
    def merged_model(self):
        if self._merged_model is None:
            self._merged_model = self.merge_models()

        return self._merged_model

    def merge_models(self):
        comm_model = CBModel(self.id)
        old_ext_comps = []
        ext_mets = []
        self.reaction_map = {}
        self.metabolite_map = {}

        # default IDs
        ext_comp_id = "ext"
        biomass_id = "community_biomass"
        comm_growth = "community_growth"

        # create external compartment

        comp = Compartment(ext_comp_id, "extracellular environment", external=True)
        comm_model.add_compartment(comp)

        # community biomass

        met = Metabolite(biomass_id, "Total community biomass", ext_comp_id)
        comm_model.add_metabolite(met)

        rxn = CBReaction(comm_growth, name="Community growth rate",
                         reversible=False, stoichiometry={biomass_id: -1},
                         lb=0, ub=inf, objective=1)

        comm_model.add_reaction(rxn)

        # add each organism

        for org_id, model in self.organisms.items():

            def rename(old_id):
                return f"{old_id}_{org_id}"

            # add internal compartments

            for c_id, comp in model.compartments.items():
                if comp.external:
                    old_ext_comps.append(c_id)
                else:
                    new_comp = Compartment(rename(c_id), comp.name)
                    comm_model.add_compartment(new_comp)

            # add metabolites

            for m_id, met in model.metabolites.items():
                if met.compartment not in old_ext_comps:  # if is internal
                    new_id = rename(m_id)
                    new_met = Metabolite(new_id, met.name, rename(met.compartment))
                    new_met.metadata = met.metadata.copy()
                    comm_model.add_metabolite(new_met)
                    self.metabolite_map[(org_id, m_id)] = new_id

                elif m_id not in comm_model.metabolites:  # if is external but was not added yet
                    new_met = Metabolite(m_id, met.name, ext_comp_id)
                    new_met.metadata = met.metadata.copy()
                    comm_model.add_metabolite(new_met)
                    ext_mets.append(new_met.id)

            # add genes

            for g_id, gene in model.genes.items():
                new_id = rename(g_id)
                new_gene = Gene(new_id, gene.name)
                new_gene.metadata = gene.metadata.copy()
                comm_model.add_gene(new_gene)

            # add internal reactions

            for r_id, rxn in model.reactions.items():

                if rxn.reaction_type == ReactionType.EXCHANGE:
                    continue

                new_id = rename(r_id)
                new_stoichiometry = {
                    m_id if m_id in ext_mets else rename(m_id): coeff
                    for m_id, coeff in rxn.stoichiometry.items()
                }

                if r_id == model.biomass_reaction:
                    new_stoichiometry[biomass_id] = 1

                if rxn.gpr is None:
                    new_gpr = None
                else:
                    new_gpr = GPRAssociation()
                    new_gpr.metadata = rxn.gpr.metadata.copy()

                    for protein in rxn.gpr.proteins:
                        new_protein = Protein()
                        new_protein.genes = [rename(g_id) for g_id in protein.genes]
                        new_protein.metadata = protein.metadata.copy()
                        new_gpr.proteins.append(new_protein)

                new_rxn = CBReaction(
                    new_id,
                    name=rxn.name,
                    reversible=rxn.reversible,
                    stoichiometry=new_stoichiometry,
                    reaction_type=rxn.reaction_type,
                    lb=rxn.lb,
                    ub=rxn.ub,
                    gpr_association=new_gpr
                )

                comm_model.add_reaction(new_rxn)
                new_rxn.metadata = rxn.metadata.copy()
                self.reaction_map[(org_id, r_id)] = new_id

        # Add exchange reactions

        for m_id in ext_mets:
            r_id = f"R_EX_{m_id[2:]}" if m_id.startswith("M_") else f"R_EX_{m_id}"
            rxn = CBReaction(r_id, reversible=True, stoichiometry={m_id: -1},
                             reaction_type=ReactionType.EXCHANGE)
            comm_model.add_reaction(rxn)

        return comm_model



class Gene(object):
    """ Base class for modeling genes. """

    def __init__(self, gene_id, name=None):
        """
        Arguments:
            gene_id (str): a valid unique identifier
            name (str): common gene name
        """
        self.id = gene_id
        self.name = name if name is not None else gene_id
        self.metadata = OrderedDict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class Protein(object):
    """ Base class for modeling proteins.

        One protein is composed of a list of genes encoding one or more subunits.
    """

    def __init__(self):
        self.genes = []
        self.metadata = OrderedDict()

    def __str__(self):
        protein_str = ' and '.join(self.genes)

        if len(self.genes) > 1:
            protein_str = '(' + protein_str + ')'

        return protein_str

    def __repr__(self):
        return str(self)


class GPRAssociation(object):
    """ Base class for modeling Gene-Protein-Reaction associations.

        Each GPR association is composed by a list of proteins that can catalyze a reaction.
        Each protein is encoded by one or several genes.
    """

    def __init__(self):
        self.proteins = []
        self.metadata = OrderedDict()

    def __str__(self):

        gpr_str = ' or '.join(map(str, self.proteins))

        if len(self.proteins) > 1:
            gpr_str = '(' + gpr_str + ')'

        return gpr_str

    def __repr__(self):
        return str(self)

    def get_genes(self):
        """ Return the set of all associated genes. """

        return {gene for protein in self.proteins for gene in protein.genes}

    def remove_gene(self, gene_id):
        for protein in self.proteins:
            if gene_id in protein.genes:
                del protein.genes[gene_id]

        self.proteins = [protein for protein in self.proteins if len(protein.genes) > 0]


class CBReaction(Reaction):

    def __init__(self, reaction_id, name=None, reversible=True, stoichiometry=None, regulators=None,
                 lb=-inf, ub=inf, objective=0, gpr_association=None, reaction_type=None):

        Reaction.__init__(self, reaction_id, name=name, reversible=reversible, stoichiometry=stoichiometry,
                          regulators=regulators, reaction_type=reaction_type)

        self.lb = 0 if reversible == False and lb < 0 else lb
        self.ub = ub
        self.objective = objective
        self.gpr = gpr_association
        self._bool_function = None

    def set_flux_bounds(self, lb, ub):
        self.lb, self.ub = lb, ub
        self.reversible = bool(lb < 0)

    def set_gpr_association(self, gpr_association):
        self.gpr = gpr_association

    def set_objective(self, value):
        self.objective = value

    def get_genes(self):
        if self.gpr is not None:
            return self.gpr.get_genes()
        else:
            return []

    def evaluate_gpr(self, active_genes):
        """ Boolean evaluation of the GPR association for a given set of active genes.

        Arguments:
            active_genes (list): list of active genes

        Returns:
            bool: is the reaction active
        """

        if self._bool_function is None:
            self._gpr_to_function()

        return self._bool_function(active_genes)

    def _gpr_to_function(self):

        if not self.gpr:
            rule = 'True'
        else:
            rule = ' ' + str(self.gpr).replace('(', '( ').replace(')', ' )') + ' '
            for gene in self.get_genes():
                rule = rule.replace(' ' + gene + ' ', ' x[\'' + gene + '\'] ')
        self._bool_function = eval('lambda x: ' + rule)

    def to_string(self, metabolite_names=None):
        """ Print a reaction to a text based representation.

        Arguments:
            metabolite_names (dict): replace metabolite id's with names (optional)

        Returns:
            str: reaction string
        """

        str_rxn = Reaction.to_string(self, metabolite_names)

        if self.lb != -inf and (self.reversible or self.lb != 0.0) or self.ub != inf:
            str_rxn += f" [{self.lb}, {self.ub}]"

        if self.objective:
            str_rxn += f' @{self.objective}'

        return str_rxn

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)


class CBModel(Model):
    """ This class implements a constraint-based model."""

    def __init__(self, model_id):
        """
        Arguments:
            model_id (string): a valid unique identifier
        """
        Model.__init__(self, model_id)
        self.genes = AttrOrderedDict()
        self._g_r_lookup = None
        self._biomass_reaction = None

    def update(self):
        Model.update(self)
        self._g_r_lookup = None
        self._biomass_reaction = None

    @property
    def biomass_reaction(self):
        if self._biomass_reaction is None:
            self._detect_biomass_reaction()

        return self._biomass_reaction

    @biomass_reaction.setter
    def biomass_reaction(self, r_id):
        if r_id not in self.reactions:
            raise RuntimeError(f"Reaction {r_id} is not in the model")
        self._biomass_reaction = r_id

    def _detect_biomass_reaction(self):

            matches = [r_id for r_id, rxn in self.reactions.items() if rxn.objective]

            if matches:
                self._biomass_reaction = matches[0]
                if len(matches) > 1:
                    warn("Ambiguous biomass reaction (model has multiple objectives).")

                id_name = self._biomass_reaction + self.reactions[self._biomass_reaction].name
                if not re.search("biomass|growth", id_name, re.IGNORECASE):
                    warn(f"Suspicious biomass identifier: {self._biomass_reaction}")
            else:
                raise RuntimeError(f"No biomass reaction identified from model objective.")

    def add_gene(self, gene, replace=True):
        """ Add a gene metabolite to the model.
        If a gene with the same id exists, it will be replaced.

        Arguments:
            gene (Gene): gene
            replace (bool): replace previous gene with same id (default: True)
       """

        if gene.id in self.genes and not replace:
            warn(f"Gene {gene.id} already exists, ignoring.")
        else:
            self.genes[gene.id] = gene

    def add_reaction_from_str(self, reaction_str, compartment=None):
        """ Parse a reaction from a string and add it to the model.

        Arguments:
            reaction_str (str): string representation a the reaction
            compartment (str): reaction compartment id (optional)

        Notes:
            If the metabolites specified in the reaction are not yet in the model, they will be automatically added.
            If the compartment id is not given, it will use the first available compartment.
        """

        if not self._parser:
            self._parser = ReactionParser()

        if compartment is None:
            compartment = list(self.compartments.keys())[0]

        r_id, reversible, stoichiometry, lb, ub, obj_coeff = \
            self._parser.parse_reaction(reaction_str, kind='cb')

        for m_id in stoichiometry:
            if m_id not in self.metabolites:
                self.add_metabolite(Metabolite(m_id, m_id, compartment=compartment))

        reaction = CBReaction(r_id, r_id, reversible, stoichiometry, None, lb, ub, obj_coeff)
        self.add_reaction(reaction)
        self._needs_update = True

        return r_id

    def add_ratio_constraint(self, r_num, r_den, ratio):
        """ Add a flux ratio constraint to the model.

        Arguments:
            r_num (str): id of the numerator
            r_den (str): id of the denominator
            ratio (float): ratio value

        Returns:
            str : identifier of the pseudo-metabolite
        """

        if r_num not in self.reactions or r_den not in self.reactions:
            raise KeyError(f"Invalid reactions in ratio {r_num}/{r_den}")

        pseudo_c_id = "pseudo"
        pseudo_m_id = f"ratio_{r_num}_{r_den}"

        if pseudo_c_id not in self.compartments:
            self.add_compartment(Compartment(pseudo_c_id))

        self.add_metabolite(Metabolite(pseudo_m_id, compartment=pseudo_c_id))
        self.reactions[r_num].stoichiometry[pseudo_m_id] = 1
        self.reactions[r_den].stoichiometry[pseudo_m_id] = -ratio
        return pseudo_m_id

    def get_reactions_by_gene(self, g_id):
        """ Get a list of reactions associated with a given gene.

        Args:
            g_id (str): gene id

        Returns:
            list: reactions catalyzed by any proteins (or subunits) encoded by this gene
        """
        g_r_lookup = self.gene_to_reaction_lookup()
        return g_r_lookup[g_id]

    def get_objective(self):
        return {r_id: rxn.objective for r_id, rxn in self.reactions.items() if rxn.objective}

    def remove_gene(self, gene_id):
        """ Remove a gene from the model.

        Arguments:
            gene_id (str) : gene id
        """
        self.remove_genes([gene_id])

    def remove_genes(self, genes, safe_delete=True):
        """ Remove a list of genes from the model.
            safe_delete (bool): also remove genes from reaction associations (default: True)

        Arguments:
            genes (list) : gene ids
        """

        if safe_delete:
            g_r_lookup = self.gene_to_reaction_lookup()

        for gene_id in genes:
            if gene_id in self.genes:
                del self.genes[gene_id]
            else:
                warn(f"No such gene '{gene_id}'")

            if safe_delete:
                for r_id in g_r_lookup[gene_id]:
                    self.reactions[r_id].gpr.remove_gene(gene_id)

    def remove_ratio_constraint(self, r_num, r_den):
        """ Remove a flux ratio constraint from the model.

        Arguments:
            r_num (str): id of the numerator
            r_den (str): id of the denominator

        """

        pseudo_m_id = f"ratio_{r_num}_{r_den}"
        if pseudo_m_id in self.metabolites:
            self.remove_metabolite(pseudo_m_id)
        else:
            raise RuntimeError(f"No ratio constraint for {r_num}/{r_den}")

    def set_flux_bounds(self, r_id, lb=None, ub=None):
        """ Define flux bounds for one reaction

        Arguments:
            r_id (str): reaction id
            lb (float): lower bound
            ub (float): upper bound
        """
        if r_id not in self.reactions:
            warn(f"Reaction {r_id} not found")
            return

        if lb is not None:
            self.reactions[r_id].lb = lb
            self.reactions[r_id].reversible = bool(lb < 0)

        if ub is not None:
            self.reactions[r_id].ub = ub

    def set_gpr_association(self, r_id, gpr, add_genes=True):
        """ Set GPR association for a given reaction:

        Arguments:
            r_id (str): reaction id
            gpr (GPRAssociation): GPR association
            add_genes (bool): check if associated genes need to be added to the model
        """

        if r_id not in self.reactions:
            raise KeyError(f"Reaction {r_id} not found")

        self.reactions[r_id].gpr = gpr

        if add_genes and gpr is not None:
            for gene_id in gpr.get_genes():
                if gene_id not in self.genes:
                    self.add_gene(Gene(gene_id))

    def set_objective(self, coefficients):
        """ Define objective coefficients for a list of reactions

        Arguments:
            coefficients (dict): dictionary of reactions and coefficients

        """
        for r_id, coeff, in coefficients.items():
            if r_id not in self.reactions:
                raise KeyError(f"Reaction {r_id} not found")
            self.reactions[r_id].objective = coeff

    def gene_to_reaction_lookup(self):
        """ Build a dictionary from genes to associated reactions.

        Returns:
            dict: gene to reaction mapping

        """
        if not self._g_r_lookup:
            self._g_r_lookup = {g_id: [] for g_id in self.genes}

            for r_id, rxn in self.reactions.items():
                genes = rxn.get_genes()
                for g_id in genes:
                    self._g_r_lookup[g_id].append(r_id)

        return self._g_r_lookup

    def evaluate_gprs(self, active_genes):
        """ Boolean evaluation of the GPR associations for a given set of active genes.

        Arguments:
            active_genes (list): list of active genes

        Returns:
            list: list of active reactions
        """
        genes_state = {gene: gene in active_genes for gene in self.genes}
        return [r_id for r_id, rxn in self.reactions.items() if rxn.evaluate_gpr(genes_state)]
