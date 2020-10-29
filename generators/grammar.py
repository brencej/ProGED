# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:00:39 2020

@author: Jure
"""

import numpy as np
from nltk import PCFG
from nltk.grammar import Nonterminal

from generators.base_generator import BaseExpressionGenerator

class GeneratorGrammar (BaseExpressionGenerator):
    def __init__ (self, grammar):
        self.generator_type = "PCFG"
    
        if isinstance(grammar, str):
            self.grammar = PCFG.fromstring(grammar)
        elif isinstance(grammar, type(PCFG.fromstring("S -> 'x' [1]"))):
            self.grammar = grammar
        else:
            raise TypeError ("Unknown grammar specification. \n"\
                             "Expected: string or nltk.grammar.PCFG object.\n"\
                             "Input: " + str(grammar))
                
        self.start_symbol = self.grammar.start()
    
    def generate_one (self):
        return generate_sample(self.grammar, items=[self.start_symbol])
    
    def code_to_expression (self, code):
        return code_to_sample(code, self.grammar, items=[self.start_symbol])

    def count_trees(self, start, height):
        """Counts all trees of height <= height."""
        if not isinstance(start, Nonterminal):
            return 1
        elif height == 0:
            return 0
        else:
            counter = 0
            prods = self.grammar.productions(lhs=start)
            for prod in prods:
                combinations = 1
                for symbol in prod.rhs():
                    combinations *= self.count_trees(symbol, height-1)
                counter += combinations
            return counter

    def count_coverage(self, start, height):
        """Counts total probability of all parse trees of height <= height."""
        if not isinstance(start, Nonterminal):
            return 1
        elif height == 0:
            return 0
        else:
            coverage = 0
            prods = self.grammar.productions(lhs=start)
            for prod in prods:
                subprobabs = prod.prob()
                for symbol in prod.rhs():
                    subprobabs *= self.count_coverage(symbol, height-1)
                coverage += subprobabs
            return coverage

    def __str__ (self):
        return str(self.grammar)
    
    def __repr__ (self):
        return str(self.grammar)
    
    
    
def generate_sample(grammar, items=[Nonterminal("S")]):
    """Samples PCFG once. 
    Input:
        grammar - PCFG object from NLTK library
        items - list containing start symbol as Nonterminal object. Default: [Nonterminal("S")]
    Output:
        frags - sampled string in list form. Call "".join(frags) to get string.
        probab - parse tree probability
        code - parse tree encoding. Use code_to_sample to recover the expression and productions.
    """
#    print(items)
    frags = []
    probab = 1
    code = ""
    if len(items) == 1:
        if isinstance(items[0], Nonterminal):
            prods = grammar.productions(lhs=items[0])
            probs = [p.prob() for p in prods]
            prod_i = np.random.choice(list(range(len(prods))), p = probs)
            frag, p, h = generate_sample(grammar, prods[prod_i].rhs())
            frags += frag
            probab *= p * probs[prod_i]
            code += str(prod_i) + h
        else:
            frags += [items[0]]
    else:
        for item in items:
            frag, p, h = generate_sample(grammar, [item])
            frags += frag
            probab *= p
            code += h
    return frags, probab, code

def code_to_sample (code, grammar, items=[Nonterminal("S")]):
    """Reconstructs expression and productions from parse tree encoding.
    Input:
        code - parse tree encoding in string format, as returned by generate sample
        grammar - PCFG object that was used to generate the code
        items - list containing start symbol for the grammar. Default: [Nonterminal("S")]
    Output:
        frags - expression in list form. Call "".join(frags) to get string.
        productions - list of used productions in string form. The parse tree is ordered top to bottom, left to right.
        code0 - auxilary variable, used by the recursive nature of the function. Should be an empty string. If not, something went wrong."""
    code0 = code
    frags = []
    productions=[]
    if len(items) == 1:
        if isinstance(items[0], Nonterminal):
            prods = grammar.productions(lhs=items[0])
            prod = prods[int(code0[0])]
            productions += [prod]
            frag, productions_child, code0 = code_to_sample(code0[1:], grammar, prod.rhs())
            frags += frag
            productions += productions_child
        else:
            frags += [items[0]]
    else:
        for item in items:
            frag, productions_child, code0 = code_to_sample (code0, grammar, [item])
            frags += frag
            productions += productions_child
    #print(frags, code0)
    return frags, productions, code0

def sample_improper_grammar (grammar):
    try:
        return generate_sample(grammar)
    except RecursionError:
        return []
    
if __name__ == "__main__":
    print("--- generators.grammar.py test ---")
    np.random.seed(0)
    grammar = GeneratorGrammar("E -> 'x' [0.7] | E '*' 'x' [0.3]")
    for i in range(5):
        f, p, c = grammar.generate_one()
        print(f, p, c)
        print(code_to_sample(c, grammar.grammar, [grammar.start_symbol]))
        print(grammar.count_trees(grammar.start_symbol,i))
        print(grammar.count_coverage(grammar.start_symbol,i))
    
    
