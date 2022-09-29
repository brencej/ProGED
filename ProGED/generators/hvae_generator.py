from enum import Enum

from ProGED.generators.base_generator import BaseExpressionGenerator
from ProGED.equation_discoverer import EqDisco

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


class SymType(Enum):
    Var = 1
    Const = 2
    Operator = 3
    Fun = 4


universal_symbols = [{"symbol": 'x1', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'x2', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'x3', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'x4', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'x5', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'C', "type": SymType.Const, "precedence": 5},
                     {"symbol": '+', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '-', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '*', "type": SymType.Operator, "precedence": 1},
                     {"symbol": '/', "type": SymType.Operator, "precedence": 1},
                     {"symbol": '^', "type": SymType.Operator, "precedence": 2},
                     {"symbol": 'sin', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'cos', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5},
                     {"symbol": 'exp', "type": SymType.Fun, "precedence": 5}]


class GeneratorHVAE(BaseExpressionGenerator):
    def __init__(self, model, variables, symbols):
        self.generator_type = "HVAE"
        self.decoding_dict = symbols
        self.precedence = {t["symbol"]: t["precedence"] for t in symbols}
        self.constant = [t["symbol"] for t in symbols if t["type"]==SymType.Const][0]
        self.variables = variables
        if isinstance(model, str):
            self.model = torch.load(model)
            self.model.eval()
        else:
            self.model = model
        self.input_mean = torch.zeros(1, 1, next(self.model.decoder.parameters()).size(0))
        a = 0

    @staticmethod
    def train_and_init(equations, variables, symbols, representation_size=64, hidden_size=64, batch_size=32, epochs=20,
                       verbose=True, model_path=None):
        s_for_tokenization = {t["symbol"]: t for i, t in enumerate(symbols)}
        trees = [tokens_to_tree(e, s_for_tokenization) for e in equations]
        model = HVAE(len(symbols), hidden_size, representation_size)
        dataset = TreeDataset(symbols, train=trees, test=[])
        sampler = TreeSampler(batch_size, len(dataset))

        def collate_fn(batch):
            return batch

        trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        iter_counter = 0
        lmbda = (np.tanh(-4.5) + 1) / 2

        midpoint = len(dataset) // (2 * batch_size)

        for epoch in range(epochs):
            total = 0
            num_trees = 0
            with tqdm(total=len(dataset), desc=f'Testing - Epoch: {epoch + 1}/{epochs}', unit='chunks') as prog_bar:
                for i, trees in enumerate(trainloader):
                    total_loss = 0
                    for t in trees:
                        mu, logvar, outputs = model(t)
                        loss, b, k = outputs.loss(mu, logvar, lmbda, criterion)
                        total_loss += loss
                        total += loss.detach().item()
                    num_trees += batch_size
                    optimizer.zero_grad()
                    total_loss = total_loss / batch_size
                    total_loss.backward()
                    optimizer.step()
                    prog_bar.set_postfix(**{'run:': "HVAE",
                                            'loss': total / num_trees,
                                            'BCE': b.item(),
                                            'KLD': k.item()})
                    prog_bar.update(batch_size)

                    iter_counter += 1
                    if iter_counter < 2400:
                        lmbda = (np.tanh((iter_counter - 4500) / 1000) + 1) / 2

                    if verbose and i == midpoint:
                        z = model.encode(trees[0])[0]
                        decoded_tree = model.decode(z, symbols)
                        print("\nO: {}".format(str(trees[0])))
                        print("P: {}".format(str(decoded_tree)))

                    for t in trees:
                        t.clear_prediction()

        if model_path is not None:
            torch.save(model, model_path)
            return GeneratorHVAE(model_path, variables, symbols)
        else:
            return GeneratorHVAE(model, variables, symbols)

    def generate_one(self):
        inp = torch.normal(self.input_mean)
        tree = self.model.decode(inp, self.decoding_dict)
        # print(str(tree))
        tree.change_redundant_variables(self.variables, self.constant)
        return tree.to_list(with_precedence=True, precedence=self.precedence), 0, str(inp.tolist())

    def decode_latent(self, latent):
        tree = self.model.decode(latent, self.decoding_dict)
        tree.change_redundant_variables(self.variables, self.constant)
        return tree.to_list(with_precedence=True, precedence=self.precedence)


class Node:
    def __init__(self, symbol=None, right=None, left=None, target=None):
        self.symbol = symbol
        self.right = right
        self.left = left
        self.target = target
        self.prediction = None

    def __str__(self):
        if self.left is None and self.right is None:
            return self.symbol
        elif self.left is not None and self.right is None:
            return f"[{self.symbol}{str(self.left)}]"
        else:
            return f"[{str(self.left)}{self.symbol}{str(self.right)}]"

    def height(self):
        hl = self.left.height() if self.left is not None else 0
        hr = self.right.height() if self.right is not None else 0
        return max(hl, hr) + 1

    def to_list(self, notation="infix", with_precedence=False, precedence=None):
        if with_precedence and precedence is None:
            raise Exception("Should add a dictionary with precendence or list without precedence")
        left = [] if self.left is None else self.left.to_list(notation, with_precedence, precedence)
        right = [] if self.right is None else self.right.to_list(notation, with_precedence, precedence)
        if notation == "prefix":
            return [self.symbol] + left + right
        elif notation == "postfix":
            return left + right + [self.symbol]
        elif not with_precedence:
            if len(left) > 0 and len(right) == 0:
                return [self.symbol] + ["("] + left + [")"]
            return left + [self.symbol] + right
        else:
            if len(left) > 0 and len(right) == 0:
                return [self.symbol] + ["("] + left + [")"]

            if self.left is not None and precedence[self.symbol] > precedence[self.left.symbol]:
                left = ["("] + left + [")"]
            if self.right is not None and precedence[self.symbol] > precedence[self.right.symbol]:
                right = ["("] + right + [")"]
            return left + [self.symbol] + right

    def to_dict(self):
        d = {'s': self.symbol}
        if self.left is not None:
            d['l'] = self.left.to_dict()
        if self.right is not None:
            d['r'] = self.right.to_dict()
        return d

    def to_vector(self, symbol_dict, n_symbols):
        mat = []
        vec = torch.zeros(1, n_symbols)
        vec[0, symbol_dict[self.symbol]] = 1.0
        mat.append(vec)
        if self.left is not None:
            mat.append(self.left.to_vector(symbol_dict, n_symbols))
        if self.right is not None:
            mat.append(self.right.to_vector(symbol_dict, n_symbols))
        return torch.cat(mat)

    @staticmethod
    def from_dict(d):
        left = None
        right = None
        if "l" in d:
            left = Node.from_dict(d["l"])
        if 'r' in d:
            right = Node.from_dict(d["r"])
        return Node(d["s"], right=right, left=left)

    @staticmethod
    def to_matrix(tree, matrix_type="prediction"):
        reps = []
        if tree.left is not None:
            reps.append(Node.to_matrix(tree.left, matrix_type))

        if matrix_type == "target":
            reps.append(torch.Tensor([torch.argmax(tree.target[0, 0, :])]).long())
        else:
            reps.append(tree.prediction[0, :, :])

        if tree.right is not None:
            reps.append(Node.to_matrix(tree.right, matrix_type))

        return torch.cat(reps)

    def create_target_vector(self, symbol_dict, n_symbols):
        target = torch.zeros(n_symbols).float()
        target[symbol_dict[self.symbol]] = 1.0
        self.target = Variable(target[None, None, :])
        if self.left is not None:
            self.left.create_target_vector(symbol_dict, n_symbols)
        if self.right is not None:
            self.right.create_target_vector(symbol_dict, n_symbols)

    def loss(self, mu, logvar, lmbda, criterion):
        pred = Node.to_matrix(self, "prediction")
        target = Node.to_matrix(self, "target")
        BCE = criterion(pred, target)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return BCE + KLD, BCE, KLD

    def trim_to_height(self, max_height, types, const_symbol="c"):
        if max_height == 1 and types[self.symbol] is not SymType.Const and types[self.symbol] is not SymType.Var:
            self.symbol = const_symbol
            self.left = None
            self.right = None
        if self.left is not None and max_height > 1:
            self.left.trim_to_height(max_height-1, types, const_symbol)
        if self.right is not None and max_height > 1:
            self.right.trim_to_height(max_height-1, types, const_symbol)

    def change_redundant_variables(self, variables, constant):
        has_child = False
        if self.right is not None:
            self.right.change_redundant_variables(variables, constant)
            has_child = True
        if self.left is not None:
            self.left.change_redundant_variables(variables, constant)
            has_child = True
        if not has_child and self.symbol not in variables:
            self.symbol = constant

    def clear_prediction(self):
        if self.left is not None:
            self.left.clear_prediction()
        if self.right is not None:
            self.right.clear_prediction()
        self.prediction = None


def tokens_to_tree(tokens, symbols):
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
    """
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token in symbols and (symbols[token]["type"] is SymType.Var or symbols[token]["type"] is SymType.Const):
            out_stack.append(Node(token))
        elif token in symbols and symbols[token]["type"] is SymType.Fun:
            operator_stack.append(token)
        elif token in symbols and symbols[token]["type"] is SymType.Operator:
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and symbols[operator_stack[-1]]["precedence"] > symbols[token]["precedence"]:
                    # or (symbols[operator_stack[-1]]["precedence"] == symbols[token]["precedence"] and symbols[token]["left_asoc"])):
                if symbols[operator_stack[-1]]["type"] is SymType.Fun:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if symbols[operator_stack[-1]]["type"] is SymType.Fun:
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in symbols and symbols[operator_stack[-1]]["type"] is SymType.Fun:
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    return out_stack[-1]


class TreeDataset(Dataset):
    def __init__(self, symbols, train, test):
        self.symbols = {t["symbol"]: i for i, t in enumerate(symbols)}
        self.n_symbols = len(symbols)

        self.train = self.transform_trees(train)
        self.test = self.transform_trees(test)

    def __getitem__(self, idx):
        return self.train[idx]

    def __len__(self):
        return len(self.train)

    def transform_trees(self, tree_objects):
        trees = []
        for i, t in enumerate(tree_objects):
            t.create_target_vector(self.symbols, self.n_symbols)
            trees.append(t)
        return trees


class TreeSampler(Sampler):
    def __init__(self, batch_size, num_eq):
        self.batch_size = batch_size
        self.num_eq = num_eq

    def __iter__(self):
        for i in range(len(self)):
            batch = np.random.randint(low=0, high=self.num_eq, size=self.batch_size)
            yield batch

    def __len__(self):
        return self.num_eq // self.batch_size


class HVAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HVAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, output_size)
        self.decoder = Decoder(output_size, hidden_size, input_size)

    def forward(self, tree):
        mu, logvar = self.encoder(tree)
        z = self.sample(mu, logvar)
        out = self.decoder(z, tree)
        return mu, logvar, out

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def encode(self, tree):
        mu, logvar = self.encoder(tree)
        return mu, logvar

    def decode(self, z, symbol_dict):
        return self.decoder.decode(z, symbol_dict)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU221(input_size=input_size, hidden_size=hidden_size)
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, tree):
        tree_encoding = self.recursive_forward(tree)
        mu = self.mu(tree_encoding)
        logvar = self.logvar(tree_encoding)
        return mu, logvar

    def recursive_forward(self, tree):
        left = self.recursive_forward(tree.left) if tree.left is not None else torch.zeros(tree.target.size(0), 1, self.hidden_size)
        right = self.recursive_forward(tree.right) if tree.right is not None else torch.zeros(tree.target.size(0), 1, self.hidden_size)
        hidden = self.gru(tree.target, left, right)
        return hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.z2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru = GRU122(input_size=output_size, hidden_size=hidden_size)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)

    def forward(self, z, tree):
        hidden = self.z2h(z)
        self.recursive_forward(hidden, tree)
        return tree

    def recursive_forward(self, hidden, tree):
        prediction = self.h2o(hidden)
        a = F.softmax(prediction, dim=2)
        tree.prediction = prediction
        if tree.left is not None or tree.right is not None:
            left, right = self.gru(a, hidden)
            if tree.left is not None:
                self.recursive_forward(left, tree.left)
            if tree.right is not None:
                self.recursive_forward(right, tree.right)

    def decode(self, z, symbol_dict):
        hidden = self.z2h(z)
        tree = self.recursive_decode(hidden, symbol_dict)
        return tree

    def recursive_decode(self, hidden, symbol_dict):
        prediction = self.h2o(hidden)
        sampled, symbol, stype = self.sample_symbol(prediction, symbol_dict)
        if stype.value is SymType.Fun.value:
            left, right = self.gru(F.softmax(sampled, dim=2), hidden)
            l_tree = self.recursive_decode(left, symbol_dict)
            r_tree = None
        elif stype.value is SymType.Operator.value:
            left, right = self.gru(F.softmax(sampled, dim=2), hidden)
            l_tree = self.recursive_decode(left, symbol_dict)
            r_tree = self.recursive_decode(right, symbol_dict)
        else:
            l_tree = None
            r_tree = None
        return Node(symbol, right=r_tree, left=l_tree)

    def sample_symbol(self, prediction, symbol_dict):
        sampled = F.softmax(prediction, dim=2)
        sampled_symbol = torch.argmax(sampled).item()
        sd = symbol_dict[sampled_symbol]
        symbol = sd["symbol"]
        stype = sd["type"]
        # is_leaf = sd["type"] is SymType.Var or sd["type"] is SymType.Const
        # is_unary = sd["type"] is SymType.Fun
        return sampled, symbol, stype


class GRU221(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU221, self).__init__()
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h1, h2):
        h = torch.cat([h1, h2], dim=2)
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        return (1 - z) * n + (z / 2) * h1 + (z / 2) * h2


class GRU122(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU122, self).__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        dh = h.repeat(1, 1, 2)
        out = (1 - z) * n + z * dh
        return torch.split(out, self.hidden_size, dim=2)


if __name__ == '__main__':
    num_points = 1000
    x = np.random.random(10000)*20 - 10
    y = x**2/8 - x + 12
    mat = np.stack([x, x, x, y]).T

    eqs = []
    with open("../examples/data/eqs_test.txt", "r") as file:
        for l in file:
            eqs.append(l.strip().split(" "))

    generator = GeneratorHVAE.train_and_init(eqs, ["x1", "x2", "x3"], universal_symbols, epochs=20,
                                             hidden_size=32, representation_size=32, model_path="../examples/parameters/params_test.pt")
    # for i in range(5):
    #     print(generator.generate_one())

    # import json
    # with open("res.txt", "r") as file:
    #     results = json.load(file)
    #     code = json.loads(results[0]["code"])

    generator = GeneratorHVAE("/home/sebastian/IJS/ProGED/ProGED/examples/parameters/params_test.pt", ["x"], universal_symbols)
    # GeneratorHVAE.benchmark_reconstruction(eqs, generator=generator)

    ed = EqDisco(data=mat, variable_names=["x1", "x2", "x3", "y"], generator=generator, sample_size=100, constant_symbol="C")
    ed.generate_models()
    print(ed.models)
    ed.fit_models(estimation_settings={"max_constants": 5})
    print(ed.get_results())
    ed.write_results("res.txt")

