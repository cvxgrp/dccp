DCCP
====

DCCP package provides an organized heuristic for convex-concave programming.
It tries to solve nonconvex problems where every function in the obejctive and the constraints has any known curvature according to the rules of disciplined convex programming (DCP).
For instance, DCCP can be used to maximize a convex function. 
The full details of our approach are discussed in [the associated paper](https://stanford.edu/~boyd/papers/dccp.html). 
DCCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install [CVXPY 1.1](http://www.cvxpy.org/).

To install the most updated DCCP, please download the repository and run ``python setup.py install`` inside.

To install DCCP from pip, please run ``pip install dccp``.

DCCP rules
----------
A problem satisfies the rules of disciplined convex-concave programming (DCCP) if it has the form
```
minimize/maximize o(x)
subject to  l_i(x) ~ r_i(x),  i=1,...,m,
```
where ``o`` (the objective), ``l_i`` (left-hand sides), and ``r_i`` (right-hand sides) are expressions (functions
in the variable ``x``) with curvature known from the DCP composition rules, and ``∼`` denotes one of the
relational operators ``==``, ``<=``, or ``>=``.

In a disciplined convex program, the curvatures of ``o``, ``l_i``, and ``r_i`` are restricted to ensure that the problem is convex. For example, if the objective is ``maximize o(x)``, then ``o`` must be concave according to the DCP composition rules. In a disciplined convex-concave program, by contrast, the objective and right and left-hand sides of the constraints can have any curvature, so long as all expressions satisfy the DCP composition rules.

The variables, parameters, and constants in DCCP should be real numbers. Problems containing complex numbers may not be supported by DCCP.


Example
-------
The following code uses DCCP to approximately solve a simple nonconvex problem.
```python
import cvxpy as cvx
import dccp
x = cvx.Variable(2)
y = cvx.Variable(2)
myprob = cvx.Problem(cvx.Maximize(cvx.norm(x - y,2)), [0 <= x, x <= 1, 0 <= y, y <= 1])
print("problem is DCP:", myprob.is_dcp())   # false
print("problem is DCCP:", dccp.is_dccp(myprob))  # true
result = myprob.solve(method='dccp')
print("x =", x.value)
print("y =", y.value)
print("cost value =", result[0])
```
The output of the above code is as follows.
```
problem is DCP: False
problem is DCCP: True
x = [ 1. -0.]
y = [-0.  1.]
cost value = 1.4142135623730951
```

The solutions obtained by DCCP can depend on the initial point of the CCP algorithm.
The algorithm starts from the values of any variables that are already specified; for any that are not specified, random values are used. 
You can specify an initial value manually by setting the ``value`` field of the variable.
For example, the following code runs the CCP algorithm with the specified initial values for ``x`` and ``y``:
```python
x.value = numpy.array([1, 2])
y.value = numpy.array([-1, 1])
result = myprob.solve(method='dccp')
```
An option is to use random initialization for all variables by ``prob.solve(method=‘dccp’, random_start=TRUE)``, and by setting the parameter ``ccp_times`` you can specify the times that the CCP algorithm runs starting from random initial values for all variables each time.

Constructing and solving problems
---------------------------------
The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. 
Once a problem object has been constructed, the following solve method can be applied.
* ``problem.solve(method='dccp')`` applies the CCP heuristic, and returns the value of the cost function, the maximum value of the slack variables, and the value of each variable. Additional arguments can be used to specify the parameters.

Solve method parameters:
* The ``ccp_times`` parameter specifies how many random initial points to run the algorithm from. The default is 1.
* The ``max_iter`` parameter sets the maximum number of iterations in the CCP algorithm. The default is 100.
* The ``solver`` parameter specifies what solver to use to solve convex subproblems.
* The ``tau`` parameter trades off satisfying the constraints and minimizing the objective. Larger ``tau`` favors satisfying the constraints. The default is 0.005.
* The ``mu`` parameter sets the rate at which ``tau`` increases inside the CCP algorithm. The default is 1.2.
* The ``tau_max`` parameter upper bounds how large ``tau`` can get. The default is 1e8.

If the convex solver for subproblems accepts any additional keyword arguments, such as ``warm_start=True``, then you can set them in the ``problem.solve()`` function, and they will be passed to the convex solver.

Result status
----------------
After running the solve method, the result status is stored in ``problem.status``. 
The status ``Converged`` means that the algorithm has converged, i.e., the slack variables converge to 0, and changes in the objective value are small enough.
The obtained solution is at least a feasible point, but it is not guaranteed to be globally optimum.
The status ``Not converged`` indicates that the algorithm has not converged, and specifically, if the slack variables (printed in the log) are not close to 0, then it usually indicates that some nonconvex constraint has not been satisfied.

Other useful functions and attributes
----------------
* ``is_dccp(problem)`` returns a boolean indicating if an optimization problem satisfies DCCP rules.
* ``linearize(expression)`` returns the linearization of a DCP expression at the point specified by ``variable.value``.
* ``convexify_obj(objective)`` returns the convexified objective of a DCCP objective.
* ``convexify_constr(constraint)`` returns the convexified constraint (without slack
variables) of a DCCP constraint, and if any expression is linearized, its domain is also returned.
