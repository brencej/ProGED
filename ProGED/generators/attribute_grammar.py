import numpy as np
from nltk import PCFG, Nonterminal
from nltk.grammar import standard_nonterm_parser, ProbabilisticProduction, _read_production

import ProGED as pg
from ProGED.generators.base_generator import BaseExpressionGenerator

from ProGED.generators.base_generator import ProGEDMaxAttemptError
from ProGED.generators.grammar import ProGEDDeadEndError, ProGEDDepthError


from nltk.grammar import ProbabilisticProduction, _read_cfg_production, PCFG, Nonterminal
from scipy.stats import norm as scipy_norm
from copy import deepcopy

class AttributedProduction(ProbabilisticProduction):
    def __init__(self, lhs, rhs, prob, attribute_rules):
        ProbabilisticProduction.__init__(self, lhs, rhs, prob=prob)
        default_attribute_rules = ["", "True", "", "True"]
        self.attribute_rules = attribute_rules + default_attribute_rules[len(attribute_rules):]
        
    def to_nltk(self):
        return ProbabilisticProduction(self.lhs(), self.rhs(), prob=self.prob())
    
    def __str__(self):
        return super().__str__() + " " + str(self.attribute_rules).replace("[", "{").replace("]", "}")

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self._lhs == other._lhs
            and self._rhs == other._rhs
            and self.prob() == other.prob()
            and self.attribute_rules == other.attribute_rules
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self._lhs, self._rhs, self.prob(), str(self.attribute_rules)))

class AttributePCFG(PCFG, BaseExpressionGenerator):
    def __init__(self, start, productions, attributes=None, externals = None, update_externals = None, depth_limit=100, repeat_limit = 100, sampling_max_tries=10):
        PCFG.__init__(self, start, productions, False)
        self.attributes = attributes
        self.depth_limit = depth_limit
        self.repeat_limit = repeat_limit
        self.sampling_max_tries = sampling_max_tries
        self.update_externals = update_externals

        if not externals:
            self.original_externals = {}
        else:
            self.original_externals = deepcopy(externals)

        nonterminals = {}
        for production in self.productions():
            nt = str(production.lhs())
            nonterminals[nt] = Nonterminal(nt)
            nonterminals[nt].frozen = False
        self.original_externals.update(nonterminals)
    
    def generate_one (self, depth_limit = None, repeat_limit = None, seed = None):
        if seed:
            np.random.seed(seed)
        if not depth_limit:
            depth_limit = self.depth_limit
        if not repeat_limit:
            repeat_limit = self.repeat_limit
        if self.update_externals:
                self.original_externals.update(self.update_externals(self.original_externals))

        i=0
        while i < repeat_limit:
            try:
                externals = deepcopy(self.original_externals)
                res = generate_sample_attributed(self, start = self._start, externals = externals, depth = 0, 
                                                   depth_limit = depth_limit)#, i
                 
                return res[0], res[1], res[2], [deepcopy(externals), res[3]]
            
            except (ProGEDDepthError, ProGEDDeadEndError, ValueError):
                i += 1

        if i >= repeat_limit:
            #print(i)
            raise ProGEDMaxAttemptError("The maximum number of attempts to generate a valid expression has been exceeded.")
        
    def to_nltk(self):
        prods = [pr.to_nltk() for pr in self.productions()]
        return PCFG(self.start(), prods)
    
    @classmethod
    def fromstring(cls, str, **kwargs):
        prods = read_attribute_grammar(str)
        start = Nonterminal(str[0][0].split("->")[0].strip())
        return cls(start, prods, **kwargs)

def read_attribute_grammar(grammar):
    attProds = []
    #prods = {}; probs = {}; atts = {}
    for att_prod in grammar:
        prod = _read_cfg_production(att_prod[0])[0]
        attProds += [AttributedProduction(prod.lhs(), prod.rhs(), att_prod[1], att_prod[2])]
    return attProds
    

def generate_sample_attributed(grammar, start, externals = {}, depth = 0, depth_limit = 100):
    #print(start)
    if depth > depth_limit:
        raise ProGEDDepthError(f"Recursion depth exceeded: {depth} / {depth_limit}. \nRaise the depth_limit, lower the probability of recursion or except this error and repeat the sampling.")
    if not isinstance(start, Nonterminal):
        return [start], 1, "", []
    else:
        if externals[str(start)].frozen:
            return externals[str(start)].subtree
        
        prods = grammar.productions(lhs=start)
        if len(prods) < 1:
            raise ProGEDDeadEndError("A dead end nonterminal has been reached while generating an expression from a grammar. Either check the grammar for mistakes or except this error and repeat the sampling.")
        
        prods = grammar.productions(lhs = start)
        good_prods = []; good_probs = []
        for prod_ind, prod in enumerate(prods):
            rules = prod.attribute_rules
            # enumerate nonterminals
            lhs_str = str(prod.lhs()) + "1" 
            syms = {lhs_str: start}
            if externals:
                syms.update(externals) 
            rhs_str = ""
            for a0 in prod.rhs():
                if isinstance(a0, Nonterminal):
                    n = 1; a = str(a0)
                    while "" + a + str(n) in syms:
                        n += 1
                    syms["" + a + str(n)] = Nonterminal(a)
                    rhs_str += a + str(n) + " "
                else:
                    rhs_str += " '" + a0 + "' "
            rhs_str = rhs_str.strip(" ")
            #print(syms)
            enumerated_prod = _read_production(lhs_str + " -> " + rhs_str, standard_nonterm_parser)[0]
            #print(enumerated_prod)
            
            # exec constructive attribute rules
            exec(rules[0], globals(), syms)
            # eval checking rules
            if eval(rules[1], globals(), syms):
                good_prods += [(enumerated_prod, syms, rules, prod_ind)]
                good_probs += [prod.prob()]

        if len(good_prods) == 0:
            raise ProGEDDeadEndError("Dead end error while selecting production")
            
        prod_i = np.random.choice(range(len(good_prods)), 1, p=np.array(good_probs)/np.sum(good_probs))[0]
        enumerated_prod, syms, rules, prod_ind = good_prods[prod_i]
        probab = good_probs[prod_i]

        frags = []; code = str(prod_ind); all_prods = [grammar.productions(lhs = start)[prod_i]]
        for symbol in enumerated_prod.rhs():
            if str(symbol) in syms:
                s = syms[str(symbol)]
            else:
                s = symbol

            frag, p, h, prods = generate_sample_attributed(grammar, s, depth = depth + 1, externals=externals,
                                                           depth_limit=depth_limit)
            frags += frag
            probab *= p
            if len(h) > 0:
                code += "_" + h
            all_prods += [prods]
            if str(symbol) in syms:
                s.expr = frag
        
        # exec final attribute rules, if they exist
        exec(rules[2], globals(), syms)
        if not eval(rules[3], globals(), syms):
            raise ProGEDDeadEndError("After-selection attribute rule evaluates false.")
            
        # handle freezing: if it was already frozen, we would never have entered this 'else'; if it was just frozen, we store the subtree
        if externals[str(start)].frozen:
            externals[str(start)].subtree = (frags, probab, code, all_prods)
            
        return frags, probab, code, all_prods

    
def get_discrete_normal_sampler(lower_bound = -5, upper_bound = 5, scale = 2):
    x = np.arange(lower_bound, upper_bound+1)
    prob = scipy_norm.cdf(x + 0.5, scale = scale) - scipy_norm.cdf(x - 0.5, scale = scale)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    return lambda n: np.random.choice(x, size = n, p = prob)

class GaussianMixtureUnitGenerator:
    def __init__(self, N, lower_bound=-5, upper_bound=5, scale=1, update_rate=0.5, update_scale_ratio = 1/4):
        self.N = N
        self.scale=scale
        self.x = np.arange(lower_bound, upper_bound+1)
        self.update_rate = update_rate
        self.update_scale_ratio = update_scale_ratio

        self.probs = []
        for _ in range(self.N):
            prob = scipy_norm.cdf(self.x+0.5, loc=0, scale=scale) - scipy_norm.cdf(self.x-0.5, loc=0, scale=scale)
            prob = prob / prob.sum()
            self.probs += [prob]

        self.units_history = []
        self.generated_units = []

    def gen_u(self):
        self.generated_units += [np.array([np.random.choice(self.x, p=p) for p in self.probs])]
        return self.generated_units[-1]

    def update(self, grammar, info):
        if info == "start":
            self.generated_units = []

        elif isinstance(info, tuple):
            self.units_history += [self.generated_units]

            probs = []
            for i in range(self.N):
                alpha = self.update_rate/len(self.generated_units)
                prob = (1-alpha)*self.probs[i]
                for j in range(len(self.generated_units)):
                    ui = self.generated_units[j][i]
                    prob += alpha*(scipy_norm.cdf(self.x+0.5, loc=ui, scale=self.scale*self.update_scale_ratio) - scipy_norm.cdf(self.x-0.5, loc=ui, scale=self.scale*self.update_scale_ratio))
                prob = prob / prob.sum()
                probs += [prob]
            self.probs = probs

# if __name__ == "__main__":
#     #prods = [["P -> P '+' M", 0.5, ["P2.u = P1.u", "M1.u = P1.u", "True"]]]
#     #prods += [["P -> M", 0.5, ["M1.u = P1.u", "True"]]]
#     #prods += [["M -> 'v'", 0.5, ["M1.u == (1,-1)"]]]
#     #prods += [["M -> 't'", 0.5, ["M1.u == (0,1)"]]]

#     #variables = ["v", "t"]
#     #units = np.array([[1,0], [1,-1], [0,1]])
#     #variables = ["x1", "x2"]
#     #units = np.array([[1,0], [1,0], [1,0]])
#     variables = ["a", "t"]
#     units = np.array([[1,-2], [0,1]])
#     zero = np.zeros(2)

#     sampler = get_discrete_normal_sampler(scale=1.5)
#     def gen_u():
#         #return units[np.random.randint(0, len(units))]
#         #return units[0]
#         un = sampler(units.shape[1])
#         #print(un)
#         return un
    
#     class UnitGenerator:
#         def __init__(self, N, lower_bound=-5, upper_bound=5, scale=1):
#             self.N = N; self.lower_bound=lower_bound; self.upper_bound=upper_bound; self.scale=scale
#             self.sampler = get_discrete_normal_sampler(self.lower_bound, self.upper_bound, self.scale)

#         def gen_u(self):
#             return self.sampler(self.N)
    
#         def update(self):
#             print("Update, yay!")

#     start = Nonterminal("E")
#     start.u = np.array([1,0])

#     #prods = [["P -> M '*' M", 1.0, ["u = gen_u()", "M1.u = u", "M2.u = P1.u - u", "True"]]]
#     #prods += [["P -> M", 0, ["M1.u = P1.u", "True"]]]
#     #prods += [["M -> 'v'", 0.5, ["np.all(M1.u == units[0])"]]]
#     #prods += [["M -> 't'", 0.5, ["np.all(M1.u == units[1])"]]]

#     # prods = [["P -> P '*' M", 0.5, ["u = gen_u()", "P2.u = u", "M1.u = P1.u - u", "True"]]]
#     # prods += [["P -> M", 0.5, ["M1.u = P1.u", "True"]]]
#     # prods += [["M -> 'v'", 0.5, ["np.all(M1.u == units[0])"]]]
#     # prods += [["M -> 't'", 0.5, ["np.all(M1.u == units[1])"]]]

#     prods = [["E -> E '+' F", 0.4, ["E2.u = E1.u", "F1.u = E1.u", "True"]]]
#     prods += [["E -> E '-' F", 0, ["E2.u = E1.u", "F1.u = E1.u", "True"]]]
#     prods += [["E -> F", 0.6, ["F1.u = E1.u", "True"]]]
#     prods += [["F -> F '*' T", 0.4, ["u = gen_u()", "F2.u = u", "T1.u = F1.u - u", "True"]]]
#     prods += [["F -> F '/' T", 0, ["u = gen_u()", "F2.u = u", "T1.u = u - F1.u", "True"]]]
#     prods += [["F -> T", 0.6, ["T1.u = F1.u", "True"]]]
#     prods += [["T -> R", 0, ["R1.u = T1.u", "True"]]]
#     prods += [["T -> V", 1, ["V1.u = T1.u", "True"]]]
#     prods += [["T -> 'C'", 0, ["np.all(T1.u == zero)"]]]
#     prods += [["R -> '(' E ')'", 0.5, ["E1.u = R1.u", "True"]]]
#     prods += [["R -> 'sqrt(' E ')'", 0.2, ["E1.u = R1.u * 2", "True"]]]
#     prods += [["R -> 'sin(' E ')'", 0.1, ["E1.u = R1.u", "np.all(R1.u == zero)"]]]
#     prods += [["R -> 'cos(' E ')'", 0.1, ["E1.u = R1.u", "np.all(R1.u == zero)"]]]
#     prods += [["R -> 'tan(' E ')'", 0.1, ["E1.u = R1.u", "np.all(R1.u == zero)"]]]
#     for i, v in enumerate(variables):
#         prods += [["V -> '" + v + "'", 1/len(variables), ["np.all(V1.u == units["+str(i)+"])"]]]

#     grammar = read_attribute_grammar(prods)
#     atts = {"u": units}

#     grammar = AttributePCFG(start, grammar, attributes = atts)
#     models = pg.generate.generate_models(grammar, symbols = {"x": ["'"+v+"'" for v in variables], "const": "'C'"}, strategy_settings = {"N": 1, "max_repeat": 10})

#     print(models)

if __name__ == "__main__":
    import pandas as pd

    def make_simplestG(nC, nL, nG):
        state_var = [f"uc{i}" for i in range(nC)] + [f"il{i}" for i in range(nL)]
        exo_var = [f"ug{i}" for i in range(nG)]
        var = state_var + exo_var

        grammar = [["S -> " + " ',' ".join(["E" for _ in range(nC+nL)]), 1.0, []]]
        grammar += [["E -> E '+' F", 0.5, []]]
        grammar += [["E -> F", 0.5, []]]
        grammar += [["F -> 'C' '*' V ", 1.0, []]]
        for v in var:
            grammar += [[f"V -> '{v}'", 1/len(var), []]]

        return grammar, state_var, exo_var


    def make_G1(nC, nL, nG, maxR = 3):
        state_var = [f"uc{i}" for i in range(nC)] + [f"il{i}" for i in range(nL)]
        exo_var = [f"ug{i}" for i in range(nG)]
        var = state_var + exo_var

        units = {"i": np.array([2,-3,1,0,-1]), "u": np.array([0,0,0,0,1]), "C":np.array([2,-2,1,0,-2]), "L":np.array([-2,4,-1,0,2]), 
                "R":np.array([-2,3,-1,0,2]), "du": np.array([0,-1,0,0,1]), "di": np.array([2,-4,1,0,-1])}
        u = [units["u"]]*nC + [units["i"]]*nL + [units["u"]]*nG
        du = [units["du"]]*nC + [units["di"]]*nL
        const_u = []
        if nC>0: const_u += [units["C"]]
        if nL>0: const_u += [units["L"]]
        if maxR>0: const_u += [units["R"]]
        
        def gen_Vu():
            return u[np.random.randint(0,len(u))]
        def gen_Cu():
            return const_u[np.random.randint(0,len(const_u))]

        grammar = [["S -> " + " ',' ".join(["E" for _ in range(nC+nL)]), 1.0, ["; ".join([f"E{i+1}.u=du[{i}]" for i in range(nC+nL)]), "True"]]]

        grammar += [["E -> E '+' F", 0, ["E2.u=E1.u; F1.u=E1.u", "True"]]]
        grammar += [["E -> E '-' F", 0, ["E2.u=E1.u; F1.u=E1.u", "True"]]]
        grammar += [["E -> F", 1, ["F1.u=E1.u", "True"]]]

        grammar += [["F -> C '*' V ", 1.0, ["V1.u=gen_Vu(); C1.u=F1.u-V1.u", "True"]]]
        for i,v in enumerate(var):
            grammar += [[f"V -> '{v}'", 1/len(var), ["", f"np.all(V1.u==u[{i}])"]]]

        grammar += [["C -> C '+' CF", 0, ["C2.u=C1.u; CF1.u=C1.u", "True"]]]
        grammar += [["C -> C '-' CF", 0, ["C2.u=C1.u; CF1.u=C1.u", "True"]]]
        grammar += [["C -> CF", 1, ["CF1.u=C1.u", "True"]]]
        grammar += [["CF -> CF '*' CT", 0.2, ["CT1.u=gen_Cu(); CF2.u=CF1.u-CT1.u", "True"]]]
        grammar += [["CF -> CF '/' CT", 0.2, ["CT1.u=gen_Cu(); CF2.u=CF1.u+CT1.u", "True"]]]
        grammar += [["CF -> CT", 0.6, ["CT1.u=CF1.u", "True"]]]
        grammar += [["CT -> '(' C ')'", 0, ["C1.u=CT1.u", "True"]]]
        grammar += [["CT -> CV", 1, ["CV1.u=CT1.u", "True"]]]
        p_const = 1/(nC + nL + maxR)
        for i in range(nC):
            grammar += [[f"CV -> 'C{i}'", p_const, ["", "np.all(CV1.u==units['C'])"]]]
        for i in range(nL):
            grammar += [[f"CV -> 'L{i}'", p_const, ["", "np.all(CV1.u==units['L'])"]]]
        for i in range(maxR):
            grammar += [[f"CV -> 'R{i}'", p_const, ["", "np.all(CV1.u==units['R'])"]]]

        externals = {"gen_Vu":gen_Vu, "gen_Cu":gen_Cu, "units":units, "u":u, "du":du}
            
        return grammar, state_var, exo_var, externals
    
    # nC = 2
    # nL = 1
    # nG = 0

    # grammar, state_var, exo_var, externals = make_G1(nC, nL, nG, maxR=3)
    # var = state_var + exo_var

    # prods = read_attribute_grammar(grammar)
    # start = Nonterminal("S")
    # grammar = AttributePCFG(start, prods, externals=externals, repeat_limit=1000)

    # data = pd.DataFrame({v: [0] for v in var})
    # ED = pg.EqDisco(data=data,
    #                 lhs_vars = state_var,
    #                 rhs_vars = var,
    #                 constant_symbol="Z",
    #                 sample_size=20,
    #                 generator=grammar)
    
    # print(ED.generate_models())



    MR = 2
    var = ["a", "b", "c", "d"]

    def verify(term_v_i, term_eq_i):
        for i in range(len(term_v_i)):
            for vi in term_v_i[i]:
                if i not in term_eq_i[vi]:
                    return False            
        return True

    start = Nonterminal("S")
    prods = [["S -> E1 ',' E2 ',' E3 ',' E4", 1.0, [f"S.term_eq_i=[]; S.term_v_i=[]", "True", "", f"verify(S.term_v_i, S.term_eq_i)"]]]
    for i in range(1, 5):
        prods += [[f"E{i} -> E", 1.0, [f"S.term_eq_i += [[]]", "True"]]]
    prods += [["E -> E '+' M", 0.5, ["", "True"]]]
    prods += [["E -> M", 0.5, ["", "True"]]]
    for i in range(MR):
        prods += [[f"M -> M{i}", 1/MR, ["", "True", f"S.term_eq_i[-1] += [{i}]", "True"]]]
        prods += [[f"M{i} -> Mi", 1.0, [f"Mi1.i={i}; S.term_v_i += [[]]", "True", f"M{i}.frozen = True", "True"]]]
    prods += [["Mi -> Mi '*' F", 0.5, ["Mi2.i=Mi1.i; F1.i=Mi1.i", "True", "", "True"]]]
    prods += [["Mi -> 'C' '*' F", 0.4, ["F1.i=Mi1.i", "True", "", "True"]]]
    prods += [["Mi -> 'C' '*' 'z' '*' F", 0.1, ["F1.i=Mi1.i", "True", "", "True"]]]
    prods += [["F -> V '**' 'K' Ki Kv", 1.0, ["V1.i=F1.i; Ki1.i=F1.i; Kv1.V=V1", "True"]]]
    for i in range(MR):
        prods += [[f"Ki -> '{i+1}'", 1/MR, ["", f"Ki1.i=={i}"]]]
    for j,v in enumerate(var):
        prods += [[f"Kv -> '{v}'", 1/len(var), ["", f"'{v}' in Kv1.V.expr"]]]
        prods += [[f"V -> '{v}'", 1/len(var), ["", f"{j} not in S.term_v_i[-1]", f"S.term_v_i[-1]+=[{j}]", "True"]]]

    externals = {"var": var, "verify": verify}
    grammar = read_attribute_grammar(prods)
    grammar = AttributePCFG(start, grammar, repeat_limit=100, externals=externals)

    ED = pg.EqDisco(lhs_vars = var, rhs_vars = var+["z"], generator=grammar, sample_size=2, constant_symbol="C")
    np.random.seed(0)
    print(ED.generate_models())