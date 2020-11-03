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
    
    
    
def generate_sample_alternative(grammar, start):
    """Alternative implementation of generate_sample. Just for example."""
    if not isinstance(start, Nonterminal):
        return [start], 1, ""
    else:
        prods = grammar.productions(lhs=start)
        probs = [p.prob() for p in prods]
        prod_i = np.random.choice(list(range(len(prods))), p = probs)
        frags = []
        probab = probs[prod_i]
        code = str(prod_i)
        for symbol in prods[prod_i].rhs():
            frag, p, h = generate_sample_alternative(grammar, symbol)
            frags += frag
            probab *= p
            code += h
        return frags, probab, code
    
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
        np.random.seed(i)
        f, p, c = grammar.generate_one()
        print((f, p, c))
        np.random.seed(i)
        print(generate_sample_alternative(grammar.grammar, grammar.start_symbol))
        print(code_to_sample(c, grammar.grammar, [grammar.start_symbol]))
        print(grammar.count_trees(grammar.start_symbol,i))
        print(grammar.count_coverage(grammar.start_symbol,i))
    print("\n-- testing different grammars: --\n")
    pgram0 = GeneratorGrammar("""
        S -> 'a' [0.3]
        S -> 'b' [0.7]
    """)
    pgram1 = GeneratorGrammar("""
        S -> A B [0.8]
        S -> 's' [0.2]
        A -> 'a' [1]
        B -> 'b' [0.3]
        B -> C D [0.7]
        C -> 'c' [1]
        D -> 'd' [1]
    """)
    pgramSS = GeneratorGrammar("""
        S -> S S [0.3]
        S -> 'a' [0.7]  
    """)
    def pgramSSparam(p=0.3):
        return GeneratorGrammar(f"""
                S -> S S [{p}]
                S -> 'a' [{1-p}]  
    """)
    pgrama = GeneratorGrammar("""
        S -> A B [0.7]
        S -> 'a' [0.1]
        S -> 'b' [0.1]
        S -> 'c' [0.1]
        A -> A1 A2 [0.3]
        A -> 'aq' [0.5]
        A -> 'bq' [0.2]
        B -> 'aw' [0.1]
        B -> 'bw' [0.9]
        A2 -> 'ak' [0.4]
        A2 -> 'bk' [0.6]
        A1 -> A11 A12 [1]
        A11 -> 'ar' [0.8]
        A11 -> 'br' [0.2]
        A12 -> 'af' [0.3]
        A12 -> 'bf' [0.7]
    """)
    pgramw = GeneratorGrammar("""
    S -> A B A2 A1 B2 B1 [0.7]
    A -> A1 A2 [0.3]
    B -> B1 B2 [0.1]
    S -> 'a' [0.1]
    S -> 'b' [0.1]
    S -> 'c' [0.1]
    A -> 'aq' [0.5]
    A -> 'bq' [0.2]
    B -> 'aw' [0.1]
    B -> 'bw' [0.8]
    A2 -> 'ak' [0.4]
    A2 -> 'bk' [0.6]
    A1 -> 'ar' [0.8]
    A1 -> 'br' [0.2]
    B2 -> 'af' [1]
    B1 -> 'bf' [1]
""")
    p=0.6
    for gramm in [grammar, pgram0, pgram1, pgrama, pgramw, pgramSS, pgramSSparam(p) ]:
        print(f"\nFor grammar:\n {gramm}")
        for i in range(0,5):
            print(gramm.count_trees(gramm.start_symbol,i), f" = count trees of height <= {i}")
            print(gramm.count_coverage(gramm.start_symbol,i), f" = total probability of height <= {i}")
    print(f"Chi says: limit probablity = 1/p - 1, i.e. p={p} => prob={1/p-1}")
        
        
