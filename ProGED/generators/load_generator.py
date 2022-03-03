from ProGED.generators.base_generator import BaseExpressionGenerator


class LoadGenerator(BaseExpressionGenerator):
    def __init__(self, filename):
        self.generator_type = "load"
        self.expr_counter = -1
        self.equations = []
        with open(filename, "r") as file:
            for l in file:
                self.equations.append(l.strip())

    def generate_one(self):
        self.expr_counter = (self.expr_counter + 1) % len(self.equations)
        return self.equations[self.expr_counter], 0, ""