# -*- coding: utf-8 -*-

import numpy as np

from ProGED.generators.grammar import GeneratorGrammar

import sympy as sp
from diophantine import solve
from itertools import product

def grammar_from_template (template_name, generator_settings, repeat_limit = 100, depth_limit = 100, string = False):
    if template_name in GRAMMAR_LIBRARY:
        grammar_str = GRAMMAR_LIBRARY[template_name](**generator_settings)
        if string:
            return grammar_str
        return GeneratorGrammar(grammar_str, repeat_limit = repeat_limit, depth_limit = depth_limit)

def construct_right (right = "a", prob = 1):
    return right + " [" + str(prob) + "]"

def construct_production (left = "S", items = ["a"], probs=[1]):
    if not items:
        return ""
    else:
        return "\n" + left + " -> " + construct_right_distribution (items=items, probs=probs)

def construct_right_distribution (items=[], probs=[]):
    p = np.array(probs)/np.sum(probs)
    S = construct_right(right=items[0], prob=p[0])
    for i in range(1, len(items)):
        S += " | " + construct_right(right=items[i], prob=p[i])
    return S

def construct_grammar_trigonometric (probs1 = [0.8,0.2], probs2=[0.4,0.4,0.2], 
                                     variables = ["'x'", "'y'"], p_vars = [0.5, 0.5],
                                     functions = ["'sin'", "'cos'", "'tan'"]):
    
    grammar = construct_production(left="S", items=["T1" + "'('" + "V" + "')'",
                                                    "T1" + " " + "T2" + "'('" + "V" + "')'"], probs=probs1)
    grammar += construct_production(left="T1", items=functions, probs=probs2)
    grammar += construct_production(left="T2", items=["'h'"], probs=[1])
    grammar += construct_production(left = "V", items=variables, probs=p_vars)
    return grammar
    
def construct_grammar_function (functions=["'sin'", "'cos'"], probs=[0.5,0.5], string=True):
    grammar = construct_production(left="S", items=["A'(''x'')'"], probs=[1])
    grammar += construct_production(left="A", items=functions, probs=probs)
    return grammar
    
def construct_grammar_polytrig (p_more_terms=[0.7,0.15,0.15], p_higher_terms=0.5, p_vars = [0.5,0.3,0.2], 
                                variables = ["'x'", "'v'", "'a'", "'sin(C*x + C)'"]):
    grammar = construct_production(left="S", items=["'C' '+' S2"], probs=[1])
    grammar += construct_production(left="S2", items=["'C' '*' T '+' S2", "'C' '*' T", "'C'"], probs=p_more_terms)
    grammar += construct_production(left="T", items=["T '*' V", "V"], probs=[p_higher_terms, 1-p_higher_terms])
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar

def construct_grammar_polynomial (p_S = [0.4, 0.6], p_T = [0.4, 0.6], p_vars = [1], p_R = [1, 0], p_F = [1],
                                  functions = ["'exp'"], variables = ["'x'"]):
    grammar = construct_production(left="S", items=["S '+' R", "R"], probs=p_S)
    grammar += construct_production(left="R", items=["T", "'C' '*' F '(' T ')'"], probs=p_R)
    grammar += construct_production(left="T", items=["T '*' V", "'C'"], probs=p_T)
    grammar += construct_production(left="F", items=functions, probs=p_F)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar

def construct_grammar_simplerational (p_S = [0.2, 0.8], p_P = [0.4, 0.3, 0.3], p_R = [0.4, 0.6], p_M = [0.4, 0.6], 
                                      p_F = [1], p_vars = [1], functions = ["'exp'"], variables = ["'x'"]):
    grammar = construct_production(left="S", items=["P '/' R", "P"], probs=p_S)
    grammar += construct_production(left="P", items=["P '+' 'C' '*' R", "'C' '*' R", "'C'"], probs=p_P)
    grammar += construct_production(left="R", items=["F '(' 'C' '*' M ')'", "M"], probs=p_R)
    grammar += construct_production(left="M", items=["M '*' V", "V"], probs=p_M)
    grammar += construct_production(left="F", items=functions, probs=p_F)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar

def construct_grammar_rational (p_S = [0.4, 0.6], p_T = [0.4, 0.6], p_vars = [1], p_R = [0.6, 0.4], p_F = [1],
                                  functions = ["'exp'"], variables = ["'x'"]):
    grammar = construct_production(left="S", items=["'(' E ')' '/' '(' E ')'"], probs=[1])
    grammar += construct_production(left="E", items=["E '+' R", "R"], probs=p_S)
    grammar += construct_production(left="R", items=["T", "'C' '*' F '(' T ')'"], probs=p_R)
    grammar += construct_production(left="T", items=["T '*' V", "'C'"], probs=p_T)
    grammar += construct_production(left="F", items=functions, probs=p_F)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar

def construct_grammar_universal (p_sum=[0.2, 0.2, 0.6], p_mul = [0.2, 0.2, 0.6], p_rec = [0.2, 0.4, 0.4], 
                                 variables=["'x'", "'y'"], p_vars=[0.5,0.5],
                                 functions=["sin", "cos", "sqrt", "exp"], p_functs=[0.6, 0.1, 0.1, 0.1, 0.1]):
    #grammar = construct_production(left="S", items=["E '+' 'C'"], probs=[1])
    grammar = construct_production(left="S", items=["S '+' F", "S '-' F", "F"], probs=p_sum)
    grammar += construct_production(left="F", items=["F '*' T", "F '/' T", "T"], probs=p_mul)
    grammar += construct_production(left="T", items=["R", "'C'", "V"], probs=p_rec)
    grammar += construct_production(left="R", items=["'(' S ')'"] + ["'"+f+"(' S ')'" for f in functions], probs=p_functs)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


def unit_to_string (unit, unit_symbols=["m", "s", "kg", "T", "V"]):
    return "".join([unit_symbols[i]+str(unit[i]) for i in range(len(unit))])

def string_to_unit (unit_string, unit_symbols=["m", "s", "kg", "T", "V"]):
    u = []
    for i in range(len(unit_symbols)-1):
        split = unit_string.split(unit_symbols[i])[1].split(unit_symbols[i+1])
        u += [int(split[0])]
    u += [int(split[1])]
    return u

def units_dict (variables, units, dimensionless = [0,0,0,0,0], target_variable_unit = [0,0,0,0,0]):
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

def unit_conversions(units_dict, order=1, unit_symbols = ["m", "s", "kg", "T", "V"]):
    conversions = {}
    #units = np.array([np.fromstring(unit.strip("[").strip("]").strip(), sep=",", dtype=int) for unit in list(units_dict.keys())])
    units = np.array([string_to_unit(unit, unit_symbols = unit_symbols) for unit in list(units_dict.keys())])
    for i in range(len(units)):
        conversions_mul = []
        conversions_div = []
        for j in range(len(units)):
            for k in range(len(units)):
                if np.array_equal(units[i], units[j] + units[k]):
                    if [j,k] not in conversions_mul and [k,j] not in conversions_mul:
                        conversions_mul += [[j,k]]
                if np.array_equal(units[i], units[j] - units[k]):
                    if [j,k] not in conversions_div:
                        conversions_div += [[j,k]]
                if np.array_equal(units[i], units[k]- units[j]):
                    if [k,j] not in conversions_div:
                        conversions_div += [[k,j]]
        conversions[str(i)+"*"] = conversions_mul
        conversions[str(i)+"/"] = conversions_div
    return conversions, units

def probs_uniform(items, A=1):
    if len(items) > 0:
        return [A/len(items)]*len(items)
    else:
        return []
    
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

def extend_units_dio(units_list, target_variable_index):
    """Extends the units to facilitate generation of expressions by grammar.
    
    A system of diophantine equations Ax=b is solved, where A is a matrix of provided units transposed,
    b is the the target variable unit and x is a vector of integer weights. x represents the largest 
    multiple of a unit that needs to be included to be able to derive expressions.  
    """
    target_unit = units_list[target_variable_index]
    units = list(units_list)
    units.pop(target_variable_index)

    """diophantine removes zero-value rows, so we need to track those zeros and add them back later"""
    units_matrix = np.vstack(units_list).T
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
    
def construct_grammar_universal_dim (variables=["'U'", "'d'", "'k'"],
                                     p_vars = [0.34, 0.33, 0.33],
                                     p_sum = [0.2, 0.2, 0.6],
                                     p_mul = [0.2, 0.2, 0.6],
                                     p_rec=[0.2, 0.4, 0.4], # recurse vs terminate
                                     functions=["sin", "cos", "sqrt", "exp"], p_functs=[0.6, 0.1, 0.1, 0.1, 0.1],
                                     units = [[2,-2,1,0,0], [1,0,0,0,0], [-1,0,0,0,0], [2,-2,1,0,0]], 
                                     target_variable_unit_index = -1,
                                     dimensionless = [0,0,0,0,0],
                                     extended_units = None):
    target_variable_unit = list(units[target_variable_unit_index])
    
    if isinstance(extended_units, list):
        units += extended_units
    elif isinstance(extended_units, str):
        if extended_units == "heuristic":
            units = extend_units(units)
        elif extended_units == "diophantine":
            units = extend_units_dio(units, target_variable_unit_index)
            print(units)
        else:
            raise ValueError("Dimensional grammar construction: choice of unit extension not recognized. Supported inputs: None, list of units, 'heuristic' and 'diophantine'")
    
    dictunits = units_dict(variables, units, dimensionless = dimensionless, target_variable_unit = target_variable_unit)
    conversions, unique_units = unit_conversions(dictunits)
    strunits = [unit_to_string(unit) for unit in unique_units]
    
    grammar = construct_production(left="S", items=["E_" + unit_to_string(target_variable_unit)], probs=[1.0])
    for i in range(len(unique_units)):          
        right_sum = ["E_" + strunits[i] + "'+'" + "F_" + strunits[i]]
        right_sub = ["E_" + strunits[i] + "'-'" + "F_" + strunits[i]]
        right_Fid = ["F_" + strunits[i]]
        grammar += construct_production(left="E_" + strunits[i], 
                                        items = right_sum + right_sub + right_Fid,
                                        probs = p_sum)
        
        right_mul = ["F_" + strunits[conv[0]] + "'*'" + "T_" + strunits[conv[1]] for conv in conversions[str(i)+"*"]]
        right_div = ["F_" + strunits[conv[0]] + "'/'" + "T_" + strunits[conv[1]] for conv in conversions[str(i)+"/"]]
        right_Tid = ["T_" + strunits[i]]
        probs_mul = probs_uniform(right_mul, A=p_mul[0])
        probs_div = probs_uniform(right_div, A=p_mul[1])
        grammar += construct_production(left="F_" + strunits[i], 
                                        items = right_mul + right_div + right_Tid,
                                        probs = probs_mul + probs_div + [p_mul[2]])
        
        if strunits[i] == unit_to_string(dimensionless):
            right_recur = ["F"]
            right_const = ["'C'"]
        else:
            right_recur = ["'('" + "E_" + strunits[i] + "')'"]
            #right_recur = ["E_" + strunits[i]]
            right_const = ["C_" + strunits[i]]
        right_var = dictunits[unit_to_string(unique_units[i])]
        probs_vars = probs_uniform(dictunits[strunits[i]], A=p_rec[1])
        probs_rec = [p_rec[0]] + probs_vars + [p_rec[2]]
        # if len(probs_vars) > 0:
        #     right_const = ["'C'"]
        # else:
        #     #probs_rec = [1.0]
        #     right_const = ["'1'"]
        grammar += construct_production(left="T_" + strunits[i], 
                                        items = right_recur + right_var + right_const,
                                        probs = probs_rec)
        
        
        if strunits[i] == unit_to_string(dimensionless):            
            right_F = ["'('" + "E_" + strunits[i] + "')'"] + ["'"+f+"('" + "E_"+strunits[i] + "')'" for f in functions]
            grammar += construct_production(left = "F", 
                                            items=right_F,
                                            probs=p_functs)

    return grammar

GRAMMAR_LIBRARY = {
    "universal": construct_grammar_universal,
    "universal-dim": construct_grammar_universal_dim,
    "rational": construct_grammar_rational,
    "simplerational": construct_grammar_simplerational,
    "polytrig": construct_grammar_polytrig,
    "trigonometric": construct_grammar_trigonometric,
    "polynomial": construct_grammar_polynomial}



if __name__ == "__main__":
    print("--- grammar_construction.py test ---")
    np.random.seed(0)
    from nltk import PCFG
    grammar = grammar_from_template("universal", {"variables":["'phi'", "'theta'", "'r'"], "p_vars":[0.2,0.4,0.4]})
    # Testing some grammar generation:
    grammar1 = grammar_from_template("trigonometric", {})
    # Grammar template without variables argument (proudces error):
    # grammar2 = grammar_from_template("trigonometric", {"variables":["'phi'", "'theta'", "'r'"]})
    grammar3 = grammar_from_template("function", {"variables":["'phi'", "'theta'", "'r'"]})
    grammar4 = grammar_from_template("trigonometric", {"probs1":[0.8,0.2], "probs2":[0.4,0.4,0.2]  })
    grammar5 = grammar_from_template("function", {"functions":["'sin'", "'cos'"], "probs":[0.5,0.5]})
    for i, grammar_ in enumerate([grammar, grammar1, grammar3, grammar4, grammar5]):
        print(f"grammar {i}: {grammar_}")
    print(grammar)
    for i in range(5):
        print(grammar.generate_one())
    print("test", construct_production("s", [], []))


