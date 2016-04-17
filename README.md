DCCP
====

DCCP package provides an organized heuristic for difference of convex programming.
It tries to solve nonconvex problems that involve objective and constraint functions that are a sum of
a convex and a concave term. The solver method provided and the syntax for constructing problems are discussed in [our associated paper](https://stanford.edu/~boyd/papers/dccp.html).

DCCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install the modified version of CVXPY hosted [here](https://github.com/xinyueshen/cvxpyhttps://github.com/xinyueshen/cvxpy).
Then install DCCP by running ``pip install dccp``.
To install from source, clone the repository and run ``python setup.py install`` inside.

Example
-------
The following code uses DCCP to approximately solve a simple difference of convex problem.
```
x = Variable(2)
y = Variable(2)
myprob = Problem(Maximize(norm(x-y,2)), [0<=x, x<=1, 0<=y, y<=1])
print "problem is DCP:", myprob.is_dcp()   # false
print "problem is DCCP:", myprob.is_dccp()  # true
result = myprob.solve(method = 'dccp')
print "x =", x.value
print "y =", y.value
print "cost value =", result[0]
```
The output of the above code is as follows.
```
problem is DCP: False
problem is DCCP: True
iteration= 1 cost value =  1.38578967145 tau =  0.005
iteration= 2 cost value =  1.41421356224 tau =  0.006
iteration= 3 cost value =  1.41421356224 tau =  0.0072
========================
x = [[  4.84999696e-11]
 [  4.84999696e-11]]
y = [[ 1.]
 [ 1.]]
cost value = 1.41421356224
```

Functions and attributes
----------------
* ``is_dccp(problem)`` returns a boolean indicating if an optimization problem satisfies dccp rules.
* ``expression.gradient`` returns a dictionary of the gradients of a DCP expression
w.r.t. its variables at the points specified by ``variable.value``. (This attribute
is also in the core CVXPY package.)
* ``expression.domain`` returns a list of constraints describing the domain of a
DCP expression. (This attribute is also in the core CVXPY package.)
* ``linearize(expression)`` returns the linearization of a DCP expression.
* ``linearize_para(expression)`` returns the linearization with CVXPY parameters of a DCP expression.
* ``convexify_obj(objective)`` returns the convexified objective (without slack
variables) of a DCCP objective.
* ``convexify_para_obj(objective)`` returns the convexified objective (without slack
variables) with CVXPY parameters of a DCCP objective.
* ``convexify_constr(constraint)`` returns the convexified constraint (without slack
variables) of a DCCP constraint.
* ``convexify_para_constr(constraint)`` returns the convexified constraint (without slack
variables) with CVXPY parameters of a DCCP constraint.
* ``dccp_transform(problem)`` returns the transformed problem with CVXPY parameters of a DCCP problem.

Constructing and solving problems
---------------------------------
The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. Once the user has constructed a problem object, they can apply the following solve method:
* ``problem.solve(method = 'dccp')`` applies the CCP heuristic, and returns the value of the cost function, the maximum value of the slack variables, and the value of each variable. Additional arguments can be used to specify the parameters.
