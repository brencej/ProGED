# -*- coding: utf-8 -*-

import numpy as np

from ProGED.generators.grammar import GeneratorGrammar

def grammar_from_template (template_name, generator_settings):
    if template_name in GRAMMAR_LIBRARY:
        grammar_str = GRAMMAR_LIBRARY[template_name](**generator_settings)
        return GeneratorGrammar(grammar_str)

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

def construct_grammar_polynomial (p_S = [0.4, 0.6], p_T = [0.4, 0.6], p_vars = [1], p_R = [0.6, 0.4], p_F = [1],
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


GRAMMAR_LIBRARY = {
    "universal": construct_grammar_universal,
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
