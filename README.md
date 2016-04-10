DCCP
====

DCCP package provides an organized heuristic for difference of convex programming.
It tries to solve nonconvex problems that involve objective and constraint functions that are a sum of
a convex and a concave term. The solver method provided and the syntax for constructing problems are discussed in [our associated paper](https://stanford.edu/~boyd/papers/dccp.html).

DCCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install CVXPY. CVXPY install guid can be found [here](http://www.cvxpy.org/). If you already have CVXPY, make sure you have the latest version by running ``pip install —upgrade cvxpy``. 

Example
-------
The following code uses DCCP to approximately solve a problem of finding a smallest singular vector of a matrix.
```
# Problem data
m = 10; n = 10;
numpy.random.seed(1)
A = numpy.random.randn(m, n)

# DCCP
x = Variable(n)
prob = Problem(Minimize(norm(A*x)), [norm(x) == 1])
prob.solve(method = 'dccp')
print objective.value
print x.value
```

Functions and attributes
----------------
* ``expression.gradient`` returns a dictionary of the gradients of a DCP expression
w.r.t. its variables at the points specified by variable.value. (This attribute
is also in the core CVXPY package.)
* ``linearize(expression)`` returns the linearization of a DCP expression.
* ``expression.domain`` returns a list of constraints describing the domain of a
DCP expression. (This attribute is also in the core CVXPY package.)
* ``convexify(constraint)`` returns the transformed constraint (without slack
variables) satisfying DCP of a DCCP constraint.
 
Constructing and solving problems
---------------------------------
The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. Once the user has constructed a problem object, they can apply the following solve method:
* ``problem.solve(method = 'dccp')`` applies the CCP heuristic, and returns the value of the transformed cost function, the value of the weight of the slack variables, and the maximum value of slack variables at each iteration. Additional arguments can be used to specify the parameters.
