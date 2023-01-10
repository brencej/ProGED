# **Pro**babilistic **G**enerative **E**quation **D**iscovery

ProGED discovers physical laws in data, expressed in the form of equations. 
A probabilistic context-free grammar (PCFG) is used to generate candidate equations. 
Their optimal values of their parameters are estimated and their perfomance evaluated.
The output of ProGED is a list of equations, ordered according to the likelihood that they represent the best model for the data.

## Scope
- algebraic equations
- systems of ordinary differential equations
- limited observability

## Features
- construct a grammar from a template or write a custom grammar
- intuitive and flexible parametrization of parsimony principle
- dimensionally-consistent expressions when measurement units are available

Details in https://www.sciencedirect.com/science/article/pii/S0950705121003403.

## Dependencies
CORE:
- numpy
- scipy
- sympy
- NLTK

OPTIONAL:
- diophantine (for dimensionally-consistent grammars)
- torch, tqdm (for HVAE generator)
- botorch (for Bayesian optimization with HVAE)
- scikim-tda (for persistent homology metric)
- hyperopt (for alternative optimization algorithm)

## Setup
You can install the package directly from the git repository:
```python3
pip install git+https://github.com/brencej/ProGED
```

## Automated testing
To check whether the installation works, run automated tests by calling
```
cd ProGED/tests/
py.test
```
or alternatively,
```
python3 -m pytest
```

## Usage example
First, generate data for a simple 1-dimensional problem:
```python3
import numpy as np

def f(x):
    return 2.0 * (x + 0.3)
	
X = np.linspace(-1, 1, 20).reshape(-1,1)
Y = f(X).reshape(-1,1)
data = np.hstack((X,Y))
```
ProGED provides an interface for common usage through the class EqDisco:
```python3
from ProGED import EqDisco

ED = EqDisco(data = data,
             sample_size = 5,
             verbosity = 1)
```
The algorithm has two main steps: generating the models and fiting the models:
```python3
print(ED.generate_models())
print(ED.fit_models())
```
Retrieve the best performing models by:
```python3
print(ED.get_results())
```
Some basic statistics on the sample of models can be computed by:
```python3
print(ED.get_stats())
```
See the paper referenced below for more details.

## Citation
```
@article{brence2021probabilistic,
  title={Probabilistic grammars for equation discovery},
  author={Brence, Jure and Todorovski, Ljup{\v{c}}o and D{\v{z}}eroski, Sa{\v{s}}o},
  journal={Knowledge-Based Systems},
  volume={224},
  pages={107077},
  year={2021},
  publisher={Elsevier}
}
```


# Usage details
Probabilistic grammar-based equation discovery can be decomposed to a number of separate steps and components, 
represented by different modules in ProGED. 

## Module descriptions 

### Equation discoverer (ProGED.equation_discoverer.py)
The highest level module, providing an interface to all the other components. Simplifies the processed
of setting up the equation discovery workflow and initalizing other modules. 

Usage: Set up the equation discovery task and the settings for the various modules and pass them when
creating the EqDisco instance. Then call EqDisco.generate_models and Eq.Disco.fit_models.
You can give the constructor the settings you wish to change as keywords, and let the constructor create
the intances for all other modules. Alternatively, you can ignore most keywords, create the module 
instances yourself and pass them to the constructor.

### Task (ProGED.task.py)
A simple class to represent the equation discovery task. 
Contains a reference to the data and information about it,
the task type (either algebraic or differential), metadata on the variables, etc.
Used by equation discoverer.

### Sampling strategy (ProGED.generate.py)
Algorithms for the generation of candidate equations. Requires a Generator instance and produces a ModelBox instance.
Currently, the only supported strategy is Monte-Carlo sampling.

Usage: Call ProGED.generate_models with the appropriate generator instance, strategy name and generator settings.

### ModelBox (ProGED.model_box.py)
ModelBox is a container object for candidate equations, represented by instances of Model.
Its core is a dictionary of Model instances, referenced by a string of respective canonical expression.
Also features a number of methods for the simplification and canonization of candidate equations.

Usage: add new models with ModelBox.add_model, access existing models as if you were using a dictionary.

### Model (ProGED.model.py)
A Model instance describes a collection of candidate equations that simplify to the same canonical expression.
To be used as an item, belonging to ModelBox.

Usage: 
Create a Model by giving the constructor at least the expression string, the probability of generation
and the code, used to reconstruct the expression from its generator. 
Use add_tree to record new parse trees 
that derive the same expression.
 Use set_estimated to record the results of parameter estimation. 
Use get_error to retrieve the error of the model. 
Use evaluate to use the model for evaluation. 
Use get_full_expr to obtain a string of the expression, with parameter values substituted.

### Grammar (ProGED.generators.grammar.py)
The only currently supported generator type are probabilistic context-free grammars.
The GeneratorGrammar class equips a NLTK.PCFG with additional methods.

Usage:
Create a generator grammar by passing it the PCFG in a string form. See NLTK.PCFG for details.
Call generate_one to obtain a single sample expression.
Use count_coverage and count_trees to perform a probabilistic analysis on the grammar.

### Grammar templates (ProGED.generators.grammar_construction.py)
ProGED provides a number of functions for the automatic construction of a GeneratorGrammar.
ProGED.GRAMMAR_LIBRARY provides a dictionary of the supported types of grammars.

Usage: Use grammar_from_template, passing it the template name 
and an appropriate generator_settings dict to construct a GeneratorGrammar of the chosen type.


## Equation discoverer arguments
| Argument | Description |
|---|---|
|task (ProGED.EDTask) | Instance of EDTask, containing specifications of the equation discovery problem. If not provided, created by EqDisco based on other arguments.|
|data (numpy.array) | Input data of shape N x M, where N is the number of samples and M is the number of variables. Not required if 'task' is provided. |
|target_variable_index (int) |  Index of column in data that belongs to the target variable.Not required if 'task' is provided.|
|time_index (int)| Index of column in data that belongs to measurement of time. Required for differential equations, None otherwise. Not required if 'task' is provided.|
|variable_names (list of strings)| Names of input variables. If not provided, names will be auto-generated. Not required if 'task' is provided.|
|task_type (string)| Specifies type of equation being solved. See ProGED.task.TASK_TYPES for supported equation types. Default: algebraic. Not required if 'task' is provided.|
|success_threshold (float)| Relative root mean squared error (RRMSE), below which a model is considered to be correct. Default: 1e-8.|
|generator (ProGED.generators.BaseExpressionGenerator or string)| Instance of generator, deriving from BaseExpressionGenerator or a string matching a geenrator type from GENERATOR_LIBRARY. Default: 'grammar'.If string, the instance will be created by EqDisco based on other arguments.|
|generate_template_name (string)| If constructing a grammar from the library, use this to specify the template name. Not required if a generator instance is provided. Default: 'universal'.|
|variable_probabilities (list of floats)| Prior probability distribution over variable symbols. If not provided, a uniform distribution is assumed. Not required if a generator instance is provided.|
|generator_settings (dict)| Arguments to be passed to the generator constructor. See documentation of the specific generator for possible settings. Has no effect if a generator instance is provided.|
|strategy (string)| Name of sampling strategy from STRATEGY_LIBRARY. Default: 'monte-carlo'.|
|strategy_settings (dict)| Arguments to be passed to the chosen sampling strategy function. See documentation for the specific strategy for available options.For Monte-Carlo sampling, the most important option is: N (int): total number of candidate equations to generate-|
|sample_size (int)| Total number of candidate equations to sample when using Monte-Carlo. Irrelevant when strategy_settings is provided. Default: 10.|
|estimation_settings (dict)| Arguments to be passed to the system for parameter estimation. See documentation for ProGED.fit_models for details and available options. Optional.|
|verbosity (int) | Level of printout desired. 0: none, 1:info, 2+: debug. |


