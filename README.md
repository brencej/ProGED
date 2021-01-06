**Pro**babilistic **G**rammar-based **E**quation **D**iscovery

ProGED discovers physical laws in data, expressed in the form of equations. 
A probabilistic context-free grammar (PCFG) is used to generate candidate equations. 
Their optimal values of their parameters are estimated and their perfomance evaluated.
The output of ProGED is a list of equations, ordered according to the likelihood that they represent the best model for the data.

#Features
- algebraic equations
- 1st order ordinary differential equations
- construct a grammar from a template or write a custom grammar
- intuitive and flexible paramterization of parsimony principle

Details in https://arxiv.org/abs/2012.00428.

#Dependencies
- numpy
- scipy
- sympy
- NLTK

#Setup
You can install the package directly from the git repository:
pip install git+https://github.com/brencej/PCFGproject




