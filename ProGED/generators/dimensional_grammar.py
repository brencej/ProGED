# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
from nltk import PCFG, Nonterminal
from nltk.grammar import read_grammar, standard_nonterm_parser, ProbabilisticProduction
from itertools import combinations_with_replacement, permutations

from ProGED.generators.grammar import GeneratorGrammar

def unit_to_string (unit, unit_symbols=["m", "s", "kg", "T", "V"]):
    """Transforms from vector representation of a unit to string representation.
    Example: [1, -2, 1, 0, 0] -> 'm1s-2kg1T0V0'
    Arguments:
        unit (list): contains exponents of base units that make up the given unit
        unit_symbols (list): contains the strings representing each base unit. Length should match the length of units.
    Returns:
        (string) 
    """
    return "".join([unit_symbols[i]+str(unit[i]) for i in range(len(unit))])

def string_to_unit (unit_string, unit_symbols=["m", "s", "kg", "T", "V"]):
    """Transforms from string representation of a unit to vector representation.
    Example:  'm1s-2kg1T0V0' -> [1, -2, 1, 0, 0] 
    Arguments:
        unit_string (string): string representation of a unit, see example
        unit_symbols (list): contains the strings representing each base unit. Length should match the length of units.
    Returns:
        (list): contains exponents of base units that make up the given unit
    """
    u = []
    for i in range(len(unit_symbols)-1):
        split = unit_string.split(unit_symbols[i])[1].split(unit_symbols[i+1])
        u += [int(split[0])]
    u += [int(split[1])]
    return u

def units_dict (variables, units, dimensionless = [0,0,0,0,0], target_variable_unit = [0,0,0,0,0]):
    """ Constructs a dictionary, which has units as keys and lists of variables as values. 
    The target variable unit will not be included unless a variable shares its unit. 
    Arguments:
        variables (list of strings): list of variables
        units (list): list of units that correspond to the variables. Units should be in vector representation.
        dimensionless (list): the unit we treat as dimensionless.
        target_variable_unit (list): the unit of the target variable for equation discovery.
    Returns: 
    """
    dictunits = {}
    for i in range(len(variables)):
        unit_string = unit_to_string(units[i])
        if unit_string in dictunits:
            dictunits[unit_string] += [variables[i]]
        else:
            dictunits[unit_string] = [variables[i]]
    if unit_to_string(dimensionless) not in dictunits:
        dictunits[unit_to_string(dimensionless)] = []
    #if unit_to_string(unit_to_string(units[target_variable_unit_index])) not in dictunits:
    #    dictunits[unit_to_string(units[target_variable_unit_index])] = []
    if unit_to_string(target_variable_unit) not in dictunits:
        dictunits[unit_to_string(target_variable_unit)] = []
        
    for unit in units:
        if unit_to_string(unit) not in dictunits:
            dictunits[unit_to_string(unit)] = []
    return dictunits

def normalize_grammar_distributions(grammar):
    """Parses the string representation of a PCFG using NLTK. If the probabilities of all productions with
    the same LHS do not sum up to 1, those probabilities are renormalized so that they do sum up to 1.
    Arguments:
        grammar (str): string representation of a PCFG in NLTK notation
    Returns:
        NLTK.Nonterminal: start symbol
        list: list of renormalized productions
    """
    start, productions = read_grammar(grammar, standard_nonterm_parser, probabilistic=True)
    #g = CFG(start, productions, True)
    probs = {}
    prods = {}
    #print(productions)
    for production in productions:
        probs[production.lhs()] = probs.get(production.lhs(), 0) + production.prob()
        prods[production.lhs()] = prods.get(production.lhs(), []) + [production]
    #return probs, productions, prods
    productions2 = []
    for (lhs, p) in probs.items():
        if not ((1 - PCFG.EPSILON) < p < (1 + PCFG.EPSILON)):
            #raise ValueError("Productions for %r do not sum to 1" % lhs)
            for i in range(len(prods[lhs])):
                productions2 += [ProbabilisticProduction(lhs, prods[lhs][i].rhs(), prob = prods[lhs][i].prob()/p)]
        else:
            productions2 += prods[lhs]
    return start, productions2

def units_operation (rhs, *args, dimensionless = [0,0,0,0,0]):
    """ Defines the unit transformations for various arithmetic operations.
    If we want to extend the functionality of dimensional grammar generation, 
    we can extend the supported operations in this function.
    Arguments:
        rhs (Sympy expression): the RHS of a grammar production, represented as a Sympy expression tree
        args (list): list of units passed to rhs, in vector representation
        dimensionless (list): the unit interpreted as dimensionless
    Returns:
        list: list of transformed units in vector representation
    """
    if isinstance(rhs, sp.core.add.Add):
        for i in range(1, len(args)):
            if args[i] != args[0]:
                raise ValueError()
        return args[0]
    
    if isinstance(rhs, sp.core.mul.Mul):
        return list(np.sum(args, axis=0))
    
    if isinstance(rhs, sp.core.power.Pow):
        if isinstance(rhs.args[1], sp.core.numbers.Rational):
            return [float(rhs.args[1])*a for a in args[0]]
        # if not rational exponent, the next elif takes care of it
    
    # All special functions that accept only dimensionless - most of them
    if isinstance(rhs, (sp.functions.elementary.trigonometric.TrigonometricFunction,
                          sp.functions.elementary.trigonometric.InverseTrigonometricFunction,
                          sp.functions.elementary.hyperbolic.HyperbolicFunction,
                          sp.functions.elementary.hyperbolic.InverseHyperbolicFunction,
                          sp.core.power.Pow,
                          sp.functions.elementary.exponential.ExpBase)):
        if args[0] == dimensionless:
            return dimensionless
        else:
            raise ValueError("Function of type " + str(type(rhs)) + " receives only a dimensionless unit.")
            
    # effectively else, if no if has triggered 
    raise SyntaxError("Function " + str(type(rhs)) + " not recognized.")
            

def unit_tree (rhs, units, dimensionless = [0,0,0,0,0]):
    """Computes the transformation of input units, according to a given RHS of a grammar production.
    Arguments:
        rhs (Sympy expression): the RHS of a grammar production, represented as a Sympy expression tree
        units (list): list of units that correspond to nonterminals in rhs from left to right
        dimensonless (list): unit to be interpereted as dimensionless
    Returns:
        list: resulting unit, as transformed by the arithmetic operations in the RHS
        int: number of units from input units, consumed in this iteration. Utility variable for recursion.
    """
    if isinstance(rhs, sp.core.Symbol):
        return units[0], 1
    elif isinstance(rhs, sp.core.numbers.Number):
        return dimensionless, 0
    else:
        units1 = []
        n = 0
        for arg in rhs.args:
            unit1, n1 = unit_tree(arg, units[n:])
            units1 += [unit1]; n += n1
        return units_operation(rhs, *units1), n
    
def correct_unit_combinations(prod, units):
    """Brute-force approach for finding all unit combinations possible for a given production.
    Generates all permutations of all combinations of matching units to nonterminals, computes the 
    unit transformation for each of them, and keeps only those for which the resulting unit matches the LHS unit.
    Due to poor computational scaling, this should be used only with small numbers of nonterminals and units.
    Arguments:
        prod (Sympy expression): RHS of a grammar production, represented as Sympy expression
        units (list): list of units in vector representation
    Returns:
        (list): list of allowed unit combinations. Each element is a list of units, where the first one belongs 
            to the LHS symbol and the rest correspond to nonterminals in the RHS from left to right. 
    """
    rhs = prod
    
    symbols = []
    for item in sp.preorder_traversal(rhs):
        if isinstance(item, sp.core.Symbol):
            symbols += [item]
    
    allowed_comb = []
    for unit in units:
        allowed_comb_for_unit = []
        for unit_comb in combinations_with_replacement(units, len(symbols)):
            for unit_set in permutations(unit_comb):
                try:
                    out_unit = unit_tree(rhs, unit_set)[0]
                    if out_unit == unit:
                        comb1 = [unit]+list(unit_set)
                        if not comb1 in allowed_comb_for_unit:
                            allowed_comb_for_unit += [comb1]
                except ValueError:
                    pass
        allowed_comb += [allowed_comb_for_unit]
    
    return allowed_comb

def unitfy_nonterminals_prepare(prod, units, n = 0, symbols = {}):
    """ Recursive utility function, used by unitfy_nonterminals. 
    """
    if isinstance(prod, sp.core.Symbol):
        c = str(prod)+str(n)
        symbols[c] = str(prod) + "_" + unit_to_string(units[n])
        return sp.sympify(c), n+1, symbols
    
    args2 = [];
    for arg in prod.args:
        arg2, n2, symb2 = unitfy_nonterminals_prepare(arg, units, n = n, symbols = symbols)
        args2 += [arg2]; n = n2; symbols = symb2
    return type(prod)(*args2), n, symbols

def unitfy_nonterminals(prod, units):
    """ Extends nonterminals to nonterminals with units. 
    Arguments:
        prod (Sympy expression): RHS of a grammar production, represented as Sympy expression
        units (list): list of units that correspond to nonterminals from left to right
    Returns:
        (string): production with renamed nonterminals in string representation
    """
    expr, n, symbols = unitfy_nonterminals_prepare(prod, units, n = 0, symbols = {})
    s = str(expr)
    # Add ' to terminals by going through nonterminals and placing ' in between
    s2 = str(s)
    sym_first, sym_last = False, False
    for key in symbols:
        split = s.strip().split(key)
        a = str(key)
        if len(split[0]) > 0:
            a = "'" + a
        if len(split[1]) > 0:
            a = a + "'"
        s2 = s2.replace(key, a)
        # check whether key is the first or last symbol, so we can later decide if we need '
        i = s.find(key)
        if i == 0:
            sym_first = True
        if i + len(key) == len(s):
            sym_last = True
    if not sym_first:
        s2 = "'" + s2
    if not sym_last:
        s2 = s2 + "'"
    s = s2
    # replace the placeholder symbols with the nonterminal+unit symbols
    for key in symbols:
        s = s.replace(key, symbols[key])
    return s

def expand_production(prod, units):
    """Expands grammar production into a dimensional grammar production.
    Parses the production with Sympy, computes all allowed combinations of unit assignments according
    to the arightmetic operation on the RHS and transforms nonterminals accordingly.
    This is a brute-force approach, so it should be used only with small numbers of nonterminals and units.
    The production probability is divided among the newly created dimensional productions.
    Arguments:
        prod (NLTK.grammar.ProbabilisticProduction): grammar production to be expanded
        units (list): list of units in vector representation
    Returns:
        (list): list of dimensional productions in string representation
    """ 
    left = prod.lhs()
    right = prod.rhs()
    str_right = ""
    for item in right:
        str_right += str(item).strip("'")
    right = sp.sympify(str_right, evaluate = False)
    
    allowed_comb = correct_unit_combinations(right, units)
    
    prods = []
    for combs in allowed_comb:
        for comb in combs:
            s = str(left) + "_" + unit_to_string(comb[0])
            s += " -> "
            s += unitfy_nonterminals(right, comb[1:])
            s += " [" + str(prod.prob() / len(combs)) + "]"
            prods += [s]
    return prods

def transform_grammar_to_dimensional(grammar, vars, units, target_unit, dimensionless = [0,0,0,0,0]):
    """ 
    Transforms a probabilistic context-free grammar (PCFG) into a dimensional PCFG.
    Each production is expanded into a number of dimensional productions, 
    equal to the number possible unit combinations, resulting from the arithemetic operations 
    on the RHS of the production. 
    This is a brute-force approach, so it should be used only with small numbers of nonterminals and units.
    Arguments:
        grammar (NLTK.PCFG or ProGED.GeneratorGrammar): thh grammar to be transformed
        vars (list): list of variables 
        units (list): list of units of length at least as vars, in vector representation
        target_unit (list): target variable unit in vector representation
        dimensionless (list): unit to be interpreted as dimensionless
    Returns:
        (NLTK.PCFG): transformed grammar
    """
    if isinstance(grammar, GeneratorGrammar):
        grammar = grammar.grammar
    
    # remove duplicates and add target unit and dimensionless unit
    all_units = list(map(string_to_unit, list(set(map(unit_to_string, units)))))
    if target_unit not in all_units:
        all_units = [target_unit] + all_units
    if dimensionless not in all_units:
        all_units += [dimensionless]
    
    # expand each production
    dim_prods = []
    for prod in grammar.productions():
        dim_prods += expand_production(prod, all_units)
    
    # add the dimensional productions for variables
    dict_units = units_dict(vars, units, target_variable_unit=target_unit)
    for u, v in dict_units.items():
        for vi in v:
            dim_prods += ["V_" + u + " -> '" + vi + "' [" + str(1.0/len(v)) + "]"]

    start_fixed, prods_fixed = normalize_grammar_distributions("\n".join(dim_prods))
    return PCFG(start_fixed, prods_fixed)
    
if __name__ == "__main__":
    from ProGED.generators.grammar_construction import grammar_from_template
    grammar = grammar_from_template("poly", {})