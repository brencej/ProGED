import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

from ProGED.equation_discoverer import EqDisco
from ProGED.generators.load_generator import LoadGenerator


class BayesianSearch:
    """
    BayesianSearch class adds an interface for equation discovery using bayesian optimization strategy instead of the
    monte-carlo (random sampling) approach

    An instance of this class represents the equation discovery algorithm for a single problem. A typical workflow using
    BayesianSearch would look like:
            generator = GeneratorHVAE(parameters_file, variables, symbols)
            bs = BayesianSearch(generator=generator, initial_samples=300)
            x, y, best_x, best_y = bs.search(data)

    Arguments:
        generator (ProGED.generators.GeneratorHVAE): instance of an already trained generator
        bounds (torch.tensor): Tensor of dimension 2x(latent dimension) representing the bounds of the search space.
            Optional. None (default) defaults to (-2.576, 2.576)^(latent dimension)
        initial_samples (int): Number of randomly sampled expressions used for training the bayesian optimization model.
            Optional. Default: 512
        max_constants (int): Maximum number of constants allowed in an expression (more constants -> more time needed
            for parameter estimation). Optional. Default: 5
        default_error (float): Error for expressions with too many constants or those that do not converge. Default 10^4
        mc_samples (int): The number of samples to be used by the sampler. See
            https://github.com/pytorch/botorch/issues/364Optional for more details. Default: 1024,
            preferably a power of 2.
        restarts (int): The number of starting points for multistart acquisition function optimization. See
            https://github.com/pytorch/botorch/issues/366 for more details. Optional. Default: 10
        raw_samples (int):  The number of samples for initialization of the optimization of the acquisition function.
            See https://github.com/pytorch/botorch/issues/366 for more details. Optional. Default: 256
        verbose (bool): Prints out performance during search if True. Optional. Default: True

    Methods:
        search (data, iterations=32, eqs_per_iter=16): Searches for equations that fit the underlying data with the
            bayesian optimization loop.

            Arguments:
                data (np.array): Input data of shape N x M, where N is the number of samples and M is the number of
                    variables. Not required if 'task' is provided.
                iterations (int): number of iterations of the bayesian optimization loop. Optional. Default: 32.
                eqs_per_iter (int): number of equation tested in each iteration of the bayesian optimization loop.
                    Optional. Default: 16
            Returns:
                Tuple(torch.tensor, torch.tensor, List, List): First values are all the tested latent representations,
                    second value are the scores of the tested latent representations, third value is a list of best
                    latent representations in each iteration, fourth value is a list of best scores in each iteration.
    """
    def __init__(self, generator, bounds=None, initial_samples=512, max_constants=5, default_error=10000,
                 mc_samples=1024, restarts=10, raw_samples=256, verbose=True):
        self.generator = generator
        self.dimension = next(self.generator.model.decoder.parameters()).size(0)
        self.bounds = bounds if bounds is not None \
            else torch.tensor([[-2.576] * self.dimension, [2.576] * self.dimension], dtype=torch.double)
        self.initial_samples = initial_samples
        self.default_error = default_error
        self.max_constants = max_constants
        self.state_dict = None
        self.k = 1
        self.mc_samples = mc_samples
        self.restarts = restarts
        self.raw_samples = raw_samples
        self.verbose = verbose

    def search(self, data, iterations=10, eqs_per_iter=10):
        # Randomly sample and test equations that are going to be used to train the bayesian optimization model
        x, y = self._initialize(data)

        # Find the best values and k for the logistic function, such that the current best value transforms into 0.5
        best_ind = y.argmin()
        best_y = [y[0, best_ind, 0].item()]
        best_x = [x[0, best_ind, :]]
        self.k = BayesianSearch._find_logistic_fun_k(best_y[0])

        if self.verbose:
            print()
            print("------------------------")
            print("Iteration 0")
            print(f"Best value {best_y[-1]}")
            print(f"Best equation: {''.join(self.generator.decode_latent(best_x[-1][None, None, :]))}")
            print("------------------------")
            print()

        for i in range(iterations):
            # Train the gaussian process model. X is normalized and y transformed using the logistic function
            model = self._fit_gp_model(normalize(x, bounds=self.bounds), self._logistic_transform(y))

            # Initialize the MC sampler and the acquisition function
            qmc_sampler = SobolQMCNormalSampler(num_samples=self.mc_samples)
            qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=1)
            # qEI = qUpperConfidenceBound(model=model, sampler=qmc_sampler, beta=0.5)

            # Optimize the acquisition function, find new candidates and evaluate them
            new_x = self._optimize_and_find_candidates(qEI, eqs_per_iter)
            new_x, new_y = self._score_candidates(new_x, data)

            # Update training points
            x = torch.hstack((x, new_x))
            y = torch.hstack((y, new_y))

            # Update progress
            best_ind = y.argmin()
            best_y.append(y[0, best_ind, 0].item())
            best_x.append(x[0, best_ind, :])

            self.state_dict = model.state_dict()
            if self.verbose:
                print()
                print("------------------------")
                print(f"Iteration {i+1}/{iterations}")
                print(f"Best value {best_y[-1]}")
                print(f"Best equation: {''.join(self.generator.decode_latent(best_x[-1][None, None, :]))}")
                print("------------------------")
                print()
        return x, y, best_x, best_y

    def _initialize(self, data):
        initial_x = draw_sobol_normal_samples(self.dimension, self.initial_samples)
        # ed = EqDisco(data=data, variable_names=self.generator.variables + ["y"], generator=self.generator,
        #              sample_size=self.initial_samples, constant_symbol=self.generator.constant, verbosity=0)
        # ed.generate_models()
        # ed.fit_models(default_error=self.default_error, estimation_settings={"max_constants": self.max_constants,
        #                                                                      "verbosity": 1})
        # results = ed.write_results(dummy=self.default_error)
        # x = torch.tensor([json.loads(eq["code"])[0][0] for eq in results])[None, :, :]
        # y = torch.tensor([eq["error"] for eq in results])[None, :, None]
        x, y = self._score_candidates(initial_x, data)
        return x, y

    def _fit_gp_model(self, x, y):
        model = SingleTaskGP(train_X=x, train_Y=y)
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(x)
        fit_gpytorch_model(mll)
        return model

    def _optimize_and_find_candidates(self, acq_func, num_candidates):
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(self.dimension, dtype=torch.float),
                torch.ones(self.dimension, dtype=torch.float),
            ]),
            q=num_candidates,
            num_restarts=self.restarts,
            raw_samples=self.raw_samples,
        )

        return unnormalize(candidates.detach(), bounds=self.bounds)

    def _score_candidates(self, x, data):
        eqs = []
        for i in range(x.size(0)):
            latent = x[i][None, None, :].float()
            eq = self.generator.decode_latent(latent)
            eqs.append((eq, latent))
        lg = LoadGenerator("", eqs=eqs)
        ed = EqDisco(data=data, variable_names=self.generator.variables + ["y"], generator=lg, sample_size=x.size(0),
                     constant_symbol=self.generator.constant, verbosity=0)
        ed.generate_models()
        ed.fit_models(estimation_settings={"max_constants": self.max_constants, "verbosity": 1})
        results = ed.write_results(dummy=self.default_error)
        new_x = torch.stack([eq["code"][0][0] for eq in results])[None, :, :]
        new_y = torch.tensor([eq["error"] for eq in results])[None, :, None]
        return new_x, new_y

    @staticmethod
    def _find_logistic_fun_k(best_y):
        return -np.log(3)/best_y

    def _logistic_transform(self, x):
        return 2 / (1 + torch.exp(-self.k * x))
