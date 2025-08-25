# User Guide

## Introduction

DCCP package provides an organized heuristic for convex-concave programming. It tries to solve nonconvex problems where every function in the objective and the constraints has any known curvature according to the rules of disciplined convex programming (DCP). For instance, DCCP can be used to maximize a convex function. The full details of our approach are discussed in [the associated paper](https://stanford.edu/~boyd/papers/dccp.html). DCCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

## DCCP Rules

A problem satisfies the rules of disciplined convex-concave programming (DCCP) if it has the form

$$
\begin{align}
\text{minimize/maximize} \quad & o(x) \\
\text{subject to} \quad & l_i(x) \sim r_i(x), \quad i=1,\ldots,m,
\end{align}
$$

where $o$ (the objective), $l_i$ (left-hand sides), and $r_i$ (right-hand sides) are expressions (functions in the variable $x$) with curvature known from the DCP composition rules, and $\sim$ denotes one of the relational operators `==`, `<=`, or `>=`.

In a disciplined convex program, the curvatures of $o$, $l_i$, and $r_i$ are restricted to ensure that the problem is convex. For example, if the objective is `maximize o(x)`, then $o$ must be concave according to the DCP composition rules. In a disciplined convex-concave program, by contrast, the objective and right and left-hand sides of the constraints can have any curvature, so long as all expressions satisfy the DCP composition rules.

The variables, parameters, and constants in DCCP should be real numbers. Problems containing complex numbers may not be supported by DCCP.

## Basic Example

The following code uses DCCP to approximately solve a simple nonconvex problem.

```python
import cvxpy as cvx
import dccp

x = cvx.Variable(2)
y = cvx.Variable(2)
myprob = cvx.Problem(cvx.Maximize(cvx.norm(x - y, 2)), [0 <= x, x <= 1, 0 <= y, y <= 1])
print("problem is DCP:", myprob.is_dcp())   # False
print("problem is DCCP:", dccp.is_dccp(myprob))  # True
result = myprob.solve(method='dccp')
print("x =", x.value)
print("y =", y.value)
print("cost value =", result[0])
```

The output of the above code is as follows:

```text
problem is DCP: False
problem is DCCP: True
x = [ 1. -0.]
y = [-0.  1.]
cost value = 1.4142135623730951
```

## Initial Values and Randomization

The solutions obtained by DCCP can depend on the initial point of the CCP algorithm. The algorithm starts from the values of any variables that are already specified; for any that are not specified, random values are used. You can specify an initial value manually by setting the `value` field of the variable. For example, the following code runs the CCP algorithm with the specified initial values for `x` and `y`:

```python
import numpy

x.value = numpy.array([1, 2])
y.value = numpy.array([-1, 1])
result = myprob.solve(method='dccp')
```

By first clearing the variable values using `x.value = None` and `y.value = None`, the CCP algorithm will use random initial values.

Setting the parameter `k_ccp` specifies the number of times that the CCP algorithm runs, starting from random initial values for all variables. The best solution found is returned.

## Constructing and Solving Problems

The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. Once a problem object has been constructed, the following solve method can be applied:

* `problem.solve(method='dccp')` applies the CCP heuristic, and returns the value of the cost function, the maximum value of the slack variables, and the value of each variable. Additional arguments can be used to specify the parameters.

For detailed information about all available parameters, see the [DCCP Settings](settings.md) page.

## Result Status

After running the solve method, the result status is stored in `problem.status`. The status `"optimal"` means that the algorithm has converged, i.e., the slack variables converge to 0, and changes in the objective value are small enough. The obtained solution is at least a feasible point, but it is not guaranteed to be globally optimum. Any other status indicates that the algorithm has not converged.

## Utility Functions and Attributes

* `is_dccp(problem)` returns a boolean indicating if an optimization problem satisfies DCCP rules.
* `linearize(expression)` returns the linearization of a DCP expression at the point specified by `variable.value`.
* `convexify_obj(objective)` returns the convexified objective of a DCCP objective.
* `convexify_constr(constraint)` returns the convexified constraint (without slack variables) of a DCCP constraint, and if any expression is linearized, its domain is also returned.
