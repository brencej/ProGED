# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:04:27 2021

@author: Jure
"""

import numpy as np
from ProGED.generators.grammar_construction import construct_production

def unit_to_string (unit, unit_symbols=["m", "s", "kg", "T", "V"]):
    return "".join([unit_symbols[i]+str(unit[i]) for i in range(len(unit))])

def string_to_unit (unit_string, unit_symbols=["m", "s", "kg", "T", "V"]):
    u = []
    for i in range(len(unit_symbols)-1):
        split = unit_string.split(unit_symbols[i])[1].split(unit_symbols[i+1])
        u += [int(split[0])]
    u += [int(split[1])]
    return u

def units_dict (variables, units, dimensionless = [0,0,0,0,0]):
    dictunits = {}
    for i in range(len(variables)):
        unit_string = unit_to_string(units[i])
        if unit_string in dictunits:
            dictunits[unit_string] += [variables[i]]
        else:
            dictunits[unit_string] = [variables[i]]
    if unit_to_string(dimensionless) not in dictunits:
        dictunits[unit_to_string(dimensionless)] = ["'C'"]
    return dictunits

def unit_conversions(units_dict, order=1):
    conversions = {}
    #units = np.array([np.fromstring(unit.strip("[").strip("]").strip(), sep=",", dtype=int) for unit in list(units_dict.keys())])
    units = np.array([string_to_unit(unit) for unit in list(units_dict.keys())])
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
    
def construct_grammar_universal_dim_direct (variables=["'U'", "'d'", "'k'", "'A'"],
                                     p_recursion=[0.1, 0.9], # recurse vs terminate
                                     p_operations=[0.2, 0.3, 0.4, 0.1], # sum, sub, mul, div
                                     p_constant=[0.2, 0.8], # constant vs variable
                                     functions=["sin", "cos", "sqrt", "exp"], p_functs=[0.6, 0.1, 0.1, 0.1, 0.1],
                                     units = [[2,-2,1,0,0], [1,0,0,0,0], [-1,0,0,0,0], [0,0,0,0,0], [2,-2,1,0,0]], 
                                     target_variable_unit_index = -1,
                                     dimensionless = [0,0,0,0,0]):
    target_variable_unit = units[target_variable_unit_index]
    dictunits = units_dict(variables, units)
    conversions, unique_units = unit_conversions(dictunits)
    strunits = [unit_to_string(unit) for unit in unique_units]
    
    grammar = construct_production(left="S", items=[unit_to_string(target_variable_unit)], probs=[1.0])
    for i in range(len(unique_units)):
        if strunits[i] == unit_to_string(dimensionless):
            grammar += construct_production(left=strunits[i], 
                                            items=["F"] + ["'"+f+"(' F ')'" for f in functions],
                                            probs=p_functs)
            left_item = "F"
        else:
            left_item = strunits[i]
            
        right_sum = ["'('" + strunits[i] + "')'" + "'+'" + "'('" + strunits[i] + "')'"]
        right_sub = ["'('" + strunits[i] + "')'" + "'-'" + "'('" + strunits[i] + "')'"]
        right_mul = ["'('" + strunits[conv[0]] + "')'" + "'*'" + "'('" + strunits[conv[1]] + "')'" for conv in conversions[str(i)+"*"]]
        right_div = ["'('" + strunits[conv[0]] + "')'" + "'/'" + "'('" + strunits[conv[1]] + "')'" for conv in conversions[str(i)+"/"]]
        right_var = dictunits[unit_to_string(unique_units[i])]
        right_const = ["'C'"]
        right_recur = right_sum + right_sub + right_mul + right_div 
        right_terminal = right_const + right_var
        right = right_recur + right_terminal
        
        probs_mul = probs_uniform(right_mul, A=p_operations[2])
        probs_div = probs_uniform(right_div, A=p_operations[3])
        probs_recur = np.hstack([p_operations[:2], probs_mul, probs_div])
        probs_vars = probs_uniform(dictunits[strunits[i]], A=p_constant[1])
        probs_terminal = np.hstack([[p_constant[0]], probs_vars])
        probs = np.hstack([p_recursion[0]*probs_recur, p_recursion[1]*probs_terminal])

        #probs = [0.4/len(right_recur)]*len(right_recur) + [0.6/len(right_terminal)]*len(right_terminal)
        
        grammar += construct_production(left=left_item, 
                                        items=right,
                                        probs = probs)

    return grammar

def construct_grammar_universal_dim (variables=["'U'", "'d'", "'k'", "'A'"],
                                     p_sum = [0.2, 0.2, 0.6],
                                     p_mul = [0.2, 0.2, 0.6],
                                     p_rec=[0.2, 0.4, 0.4], # recurse vs terminate
                                     functions=["sin", "cos", "sqrt", "exp"], p_functs=[0.6, 0.1, 0.1, 0.1, 0.1],
                                     units = [[2,-2,1,0,0], [1,0,0,0,0], [-1,0,0,0,0], [0,0,0,0,0], [2,-2,1,0,0]], 
                                     target_variable_unit_index = -1,
                                     dimensionless = [0,0,0,0,0]):
    target_variable_unit = units[target_variable_unit_index]
    dictunits = units_dict(variables, units)
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
        else:
            right_recur = ["'('" + "E_" + strunits[i] + "')'"]
        right_var = dictunits[unit_to_string(unique_units[i])]
        right_const = ["'C'"]
        probs_vars = probs_uniform(dictunits[strunits[i]], A=p_rec[1])
        grammar += construct_production(left="T_" + strunits[i], 
                                        items = right_recur + right_var + right_const,
                                        probs = [p_rec[0]] + probs_vars + [p_rec[2]])
        
        if strunits[i] == unit_to_string(dimensionless):
            right_F = ["'('" + "E_" + strunits[i] + "')'"] + ["'"+f+"('" + "E_"+strunits[i] + "')'" for f in functions]
            grammar += construct_production(left = "F", 
                                            items=right_F,
                                            probs=p_functs)

    return grammar