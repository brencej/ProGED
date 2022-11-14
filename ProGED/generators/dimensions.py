import numpy as np
import sympy as sp
from nltk import PCFG, Nonterminal
from nltk.grammar import read_grammar, standard_nonterm_parser, CFG, ProbabilisticProduction, _read_production

from diophantine import solve
from itertools import product, combinations_with_replacement, permutations

from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import unit_to_string, string_to_unit, units_dict


def unit_to_string (unit, unit_symbols=["m", "s", "kg", "T", "V"]):
    return "".join([unit_symbols[i]+str(unit[i]) for i in range(len(unit))])

def string_to_unit (unit_string, unit_symbols=["m", "s", "kg", "T", "V"]):
    u = []
    for i in range(len(unit_symbols)-1):
        split = unit_string.split(unit_symbols[i])[1].split(unit_symbols[i+1])
        u += [int(split[0])]
    u += [int(split[1])]
    return u

def units_dict (variables, units, target_variable_unit = [0,0,0,0,0], unit_symbols = ["m", "s", "kg", "T", "V"]):
    dimensionless = [0]*len(unit_symbols)
    dictunits = {}
    for i in range(len(variables)):
        unit_string = unit_to_string(units[i], unit_symbols)
        if unit_string in dictunits:
            dictunits[unit_string] += [variables[i]]
        else:
            dictunits[unit_string] = [variables[i]]
    if unit_to_string(dimensionless) not in dictunits:
        dictunits[unit_to_string(dimensionless, unit_symbols)] = []

    if unit_to_string(target_variable_unit) not in dictunits:
        dictunits[unit_to_string(target_variable_unit, unit_symbols)] = []
        
    for unit in units:
        if unit_to_string(unit) not in dictunits:
            dictunits[unit_to_string(unit, unit_symbols)] = []
    return dictunits

def check_rules (units, syms, rules):
    """ Checks whether given assignment of units to nonterminals follows all attribute rules for the production rule. 
    Arguments:
        units (list of lists): sequence of units in vector representation
        syms (dict): dictionary of symbols, used in attribute rules, formatted as {string: sp.MatrixSymbol}
        rules (list): list of attribute rules, represented as sympy matrix expressions
    Returns:
        (bool): False if any rules are broken
    """
    for rule in rules:
        sym_units = [sp.Matrix(unit) for unit in units]
        subs_pairs = dict(zip(list(syms.values()), sym_units))
        if not rule.subs(subs_pairs).doit().is_zero_matrix:
            return False
    return True

def expand_production(att_prod, units):
    prod = _read_production(att_prod[0], standard_nonterm_parser)[0]
    syms = {}; lhs_str = ""; rhs_str = ""

    a = str(prod.lhs()) + "1"
    syms["u" + a] = sp.MatrixSymbol("u" + a, len(units[0]), 1)
    lhs_str += str(a)

    for a0 in prod.rhs():
        if isinstance(a0, Nonterminal):
            n = 1; a = str(a0)
            while "u" + a + str(n) in syms:
                n += 1
            syms["u" + a + str(n)] = sp.MatrixSymbol("u" + a + str(n), len(units[0]), 1)
            rhs_str += a + str(n)
        else:
            rhs_str += " '" + a0 + "' "

    rules = []
    for rule_str in att_prod[2]:
        rules += [sp.sympify(rule_str, locals=syms)]


    allowed_comb = []
    for unit_comb in combinations_with_replacement(units, len(syms)):
        for unit_set in permutations(unit_comb):
            #print(str(unit_set))
            if check_rules(unit_set, syms, rules):
                if not unit_set in allowed_comb:
                    allowed_comb += [unit_set]

    dimprods = []
    for comb in allowed_comb:
        prod_str = lhs_str + " -> " + rhs_str + " [" + str(att_prod[1]) + "]"
        for (sym, unit) in zip(list(syms.keys()), comb):
            #print(prod_str)
            prod_str = prod_str.replace(sym[1:], sym[1:-1] + "_" + unit_to_string(unit))
        dimprods += [prod_str]

    # !TODO: add the correct probability procedure from Algorithm 1
    # correct_probabilities does not do the same thing!

    return dimprods

def correct_probabilities(grammar):
    """ Checks whether the probabilities for each nonterminal add up to 1 and renormalizes them if they don't. 
    Arguments:
        grammar (str): string representation of grammar in NLTK syntax
    Returns:
        (nltk.grammar.PCFG): PCFG with corrected probabilities
    """
    start, productions = read_grammar(grammar, standard_nonterm_parser, probabilistic=True)
    probs = {}
    prods = {}
    for production in productions:
        probs[production.lhs()] = probs.get(production.lhs(), 0) + production.prob()
        prods[production.lhs()] = prods.get(production.lhs(), []) + [production]
    productions2 = []
    for (lhs, p) in probs.items():
        if not ((1 - PCFG.EPSILON) < p < (1 + PCFG.EPSILON)):
            for i in range(len(prods[lhs])):
                productions2 += [ProbabilisticProduction(lhs, prods[lhs][i].rhs(), prob = prods[lhs][i].prob()/p)]
        else:
            productions2 += prods[lhs]
    return productions2

def remove_duplicate_units(units, unit_symbols=["m", "s", "kg", "T", "V"]):
    units = [unit_to_string(u, unit_symbols=unit_symbols) for u in units]
    units = list(set(units))
    return [string_to_unit(u, unit_symbols=unit_symbols) for u in units]


def dimensional_attribute_grammar_to_pcfg(attribute_productions, vars, units, target_unit, start = None, append_vars = True, unit_symbols = ["m", "s", "kg", "T", "V"]):
    dimensionless = [0]*len(unit_symbols)
    if not start:
        start = attribute_productions[0][0].split("->")[0].strip() 
    start += "_" + unit_to_string(target_unit, unit_symbols)

    # remove duplicates and add target unit and dimensionless unit
    all_units = remove_duplicate_units(units, unit_symbols)
    if target_unit not in all_units:
        all_units = [target_unit] + all_units
    if dimensionless not in all_units:
        all_units += [dimensionless]
    
    # expand each production
    dim_prods = []
    for att_prod in attribute_productions:
        dim_prods += expand_production(att_prod, all_units)
    
    if append_vars:
    # add the dimensional productions for variables
        dict_units = units_dict(vars, units, target_variable_unit=target_unit)
        for u, v in dict_units.items():
            for vi in v:
                dim_prods += ["V_" + u + " -> '" + vi + "' [" + str(1.0/len(v)) + "]"]
                
    dim_prods = correct_probabilities(dim_prods)
    return Nonterminal(start), dim_prods


def clumsy_solve(A: sp.Matrix, b: sp.Matrix):
    """Fixes a bug in diophantine.solve by expanding the system of equations to force multiple solutions."""
    try:  # First try to find 0 or infinite solutions.
        x = solve(A, b)
        return x
    except NotImplementedError:
        # Expand the system to get more than 1 solutions (infinite, 
        # since nontrivial kernel). Then drop the last element of 
        # the solution to get the solution of the original unexpanded 
        # system.
        A_inf = sp.Matrix.hstack(A, sp.Matrix.zeros(A.shape[0], 1))  # Expand system.
        x = solve(A_inf, b)  # infinite solutions so no error ...
        return [sp.Matrix(x[0][:-1])]  # Drop the last element of the vector.

def extend_units(units):
    ext_units = list(units)
    for unit in units:
        for i in range(len(unit)):
            for j in range(abs(unit[i])):
                u = [0]*i + [int(unit[i]/abs(unit[i])*(abs(unit[i])-j))] + unit[min([i+1, len(unit)]):]
                if u not in ext_units:
                    ext_units += [u]
                    
    for i in range(len(units[0])):
        if np.sum(np.abs(units)[:,i]) > 0:
            u = [0]*i + [1] + [0]*(len(units[0]) - i - 1)
            if u not in ext_units:
                ext_units += [u]
        
    return ext_units

def extend_units_dio(units_list, target_variable_unit):
    """Extends the units to facilitate generation of expressions by grammar.
    
    A system of diophantine equations Ax=b is solved, where A is a matrix of provided units transposed,
    b is the the target variable unit and x is a vector of integer weights. x represents the largest 
    multiple of a unit that needs to be included to be able to derive expressions.  
    """
    #target_unit = units_list[target_variable_index]
    units = list(units_list)
    #units.pop(target_variable_index)

    """diophantine removes zero-value rows, so we need to track those zeros and add them back later"""
    units_matrix = np.vstack(units_list + target_variable_unit).T
    #zeroind = np.where((units_matrix==0).all(axis=1))[0]
    #units_matrix = np.delete(units_matrix, zeroind, axis=0)

    """Define and solve the system of diophantine equations."""
    A = sp.Matrix(units_matrix[:, :-1])
    b = sp.Matrix(units_matrix[:, -1])
    if len(A) < 1:
        return units_list
    solutions = clumsy_solve(A, b)
    if len(solutions) < 1:
        print("Unable to extend units - found no solutions to diophantine equation.")
        return units_list

    """Convert from the coefficient basis to the units basis"""
    expanded_units = list(units)
    for solution in solutions:
        expanded_multipliers = []
        """Insert back the removed zeros"""
        #solution = np.array(sol).reshape((-1,))
        #for i in zeroind:
        #    solution = np.insert(solution, i, 0)
        """For each dimension, generate every integer multiplier up to the maxumal."""
        for u in solution:
            expanded_multipliers += [[np.sign(u)*ui for ui in range(0, abs(u)+1)]]
        """Generate the cartesian product of the integer multipliers between all dimensions."""
        expanded_combinations = product(*expanded_multipliers)
        """Obtain units by computing sums of weighted units."""
        for comb in expanded_combinations:
            unit = list(np.dot(comb, units))
            if unit not in expanded_units:
                expanded_units += [unit]
    return expanded_units

def construct_dimensional_universal_grammar(variables,
                                  units,
                                  target_variable_unit,
                                  constant_symbol = "C",
                                  p_sum = [0.2, 0.2, 0.6],
                                  p_mul = [0.2, 0.2, 0.6],
                                  p_rec = [0.2, 0.4, 0.4],
                                  p_fun = [0.6, 0.1, 0.1, 0.1, 0.1],
                                  functions=["sin", "cos", "sqrt", "exp"], 
                                  extended_units = None,
                                  unit_symbols = ["m", "s", "kg", "T", "V"]
                                  ):

    if isinstance(extended_units, list):
        units += extended_units
    elif isinstance(extended_units, str):
        if extended_units == "heuristic":
            units = extend_units(units)
        elif extended_units == "diophantine":
            units = extend_units_dio(units, target_variable_unit)
            print(units)
        else:
            raise ValueError("Dimensional grammar construction: choice of unit extension not recognized. Supported inputs: None, list of units, 'heuristic' and 'diophantine'")
    

    prods = []
    prods += [["E -> E '+' F", p_sum[0], ["uE1 - uE2", "uE1 - uF1"]]]
    prods += [["E -> E '-' F", p_sum[1], ["uE1 - uE2", "uE1 - uF1"]]]
    prods += [["E -> F", p_sum[2], ["uE1 - uF1"]]]
    prods += [["F -> F '*' T", p_mul[0], ["uF2 + uT1 - uF1"]]]
    prods += [["F -> F '/' T", p_mul[1], ["uF2 - uT1 - uF1"]]]
    prods += [["F -> T", p_mul[2], ["uT1 - uF1"]]]
    prods += [["T -> R", p_rec[0], ["uT1 - uR1"]]]
    prods += [["T -> '" + constant_symbol + "'", p_rec[1], ["uT1"]]]
    prods += [["T -> V", p_rec[2], ["uV1 - uT1"]]]
    prods += [["R -> '(' E ')'", p_fun[0], ["uR1 - uE1"]]]
    for (i,f) in enumerate(functions):
        prods += [["R -> '"+f+"(' E ')'", p_fun[i+1], ["uR1 - uE1", "uR1"]]]

    pcfg_start, pcfg_prods = dimensional_attribute_grammar_to_pcfg(prods, variables, units, target_variable_unit, append_vars = True, unit_symbols = unit_symbols)
    return GeneratorGrammar(PCFG(pcfg_start, pcfg_prods))

def construct_dimensional_polynomial_grammar(variables,
                                  units,
                                  target_variable_unit,
                                  constant_symbol = "C",
                                  p_sum = [0.4, 0.6],
                                  p_mul = [0.4, 0.4, 0.2],
                                  extended_units = None,
                                  unit_symbols = ["m", "s", "kg", "T", "V"]
                                  ):

    if isinstance(extended_units, list):
        units += extended_units
    elif isinstance(extended_units, str):
        if extended_units == "heuristic":
            units = extend_units(units)
        elif extended_units == "diophantine":
            units = extend_units_dio(units, target_variable_unit)
            print(units)
        else:
            raise ValueError("Dimensional grammar construction: choice of unit extension not recognized. Supported inputs: None, list of units, 'heuristic' and 'diophantine'")
    

    prods = []
    prods += [["P -> P '+' M", p_sum[0], ["uP1 - uP2", "uP1 - uM1"]]]
    prods += [["P -> M", p_sum[1], ["uP1 - uM1"]]]
    prods += [["M -> M '*' V", p_mul[0], ["-uM1 + uM2 + uV1"]]]
    prods += [["M -> V", p_mul[1], ["uM1 - uV1"]]]
    prods += [["M -> '" + constant_symbol + "'", p_mul[2], ["uM1"]]]

    pcfg_start, pcfg_prods = dimensional_attribute_grammar_to_pcfg(prods, variables, units, target_variable_unit, append_vars = True, unit_symbols = unit_symbols)
    return GeneratorGrammar(PCFG(pcfg_start, pcfg_prods))