# DCCP

[![build](https://github.com/cvxgrp/dccp/actions/workflows/release.yaml/badge.svg)](https://github.com/cvxgrp/dccp/actions/workflows/release.yaml)
[![docs](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://www.cvxpy.org/dccp/)
[![codecov](https://codecov.io/gh/cvxgrp/dccp/graph/badge.svg)](https://codecov.io/gh/cvxgrp/dccp)
[![license](https://img.shields.io/github/license/cvxgrp/dccp)](https://github.com/cvxgrp/dccp/blob/main/LICENSE)
[![pypi](https://img.shields.io/pypi/v/dccp)](https://pypi.org/project/dccp/)

DCCP package provides an organized heuristic for convex-concave programming.
It tries to solve nonconvex problems where every function in the objective and the constraints has any known curvature according to the rules of disciplined convex programming (DCP).
For instance, DCCP can be used to maximize a convex function.
The full details of our approach are discussed in [the associated paper](https://stanford.edu/~boyd/papers/dccp.html).
DCCP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

## Installation

You should first install [CVXPY 1.5](http://www.cvxpy.org/) or greater.

You can install the latest DCCP package via pip:

```bash
pip install dccp
```

To install the development version, clone this repository and install in development mode:

```bash
git clone https://github.com/cvxgrp/dccp.git
cd dccp
pip install -e .
```

## DCCP Rules

A problem satisfies the rules of disciplined convex-concave programming (DCCP) if it has the form

$$
\begin{align}
\text{minimize/maximize} \quad & o(x) \\
\text{subject to} \quad & l_i(x) \sim r_i(x), \quad i=1,\ldots,m,
\end{align}
$$

where $o$ (the objective), $l_i$ (left-hand sides), and $r_i$ (right-hand sides) are expressions (functions
in the variable $x$) with curvature known from the DCP composition rules, and $\sim$ denotes one of the
relational operators $==$, $<=$, or $>=$.

In a disciplined convex program, the curvatures of $o$, $l_i$, and $r_i$ are restricted to ensure that the problem is convex. For example, if the objective is $\text{maximize} \, o(x)$, then $o$ must be concave according to the DCP composition rules. In a disciplined convex-concave program, by contrast, the objective and right and left-hand sides of the constraints can have any curvature, so long as all expressions satisfy the DCP composition rules.

The variables, parameters, and constants in DCCP should be real numbers. Problems containing complex numbers may not be supported by DCCP.

## Example

The following code uses DCCP to approximately solve a simple nonconvex problem.

```python
import cvxpy as cp
import dccp

x = cp.Variable(2)
y = cp.Variable(2)
myprob = cp.Problem(cp.Maximize(cp.norm(x - y, 2)), [0 <= x, x <= 1, 0 <= y, y <= 1])
print("problem is DCP:", myprob.is_dcp())   # False
print("problem is DCCP:", dccp.is_dccp(myprob))  # True
result = myprob.solve(method='dccp', seed=3)
print("x =", x.value.round(3))
print("y =", y.value.round(3))
print("cost value =", result)
```

The output of the above code is as follows.

```text
problem is DCP: False
problem is DCCP: True
x = [1. 0.]
y = [0.  1.]
cost value = 1.4142135623730951
```

The solutions obtained by DCCP can depend on the initial point of the CCP algorithm.
The algorithm starts from the values of any variables that are already specified; for any that are not specified, random values are used.
You can specify an initial value manually by setting the `value` field of the variable.
For example, the following code runs the CCP algorithm with the specified initial values for `x` and `y`:

```python
import numpy

x.value = numpy.array([1, 2])
y.value = numpy.array([-1, 1])
result = myprob.solve(method='dccp')
```

By first clearing the variable values using `x.value = None` and `y.value = None`, the CCP algorithm will use random initial values.

Setting the parameter `k_ccp` specifies the number of times that the CCP algorithm runs, starting from random initial values for all variables. The best solution found is returned.

For all available parameters, see the [documentation](https://www.cvxpy.org/dccp/).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you wish to cite DCCP, please cite the DCCP papers listed in our [citation guide](https://www.cvxgrp.org/dccp/citing) or copy the text below.

```bibtex
@article{shen2016disciplined,
    author       = {Xinyue Shen and Steven Diamond and Yuantao Gu and Stephen Boyd},
    title        = {Disciplined convex-concave programming},
    journal      = {2016 IEEE 55th Conference on Decision and Control (CDC)},
    pages        = {1009--1014},
    year         = {2016},
    url          = {https://stanford.edu/~boyd/papers/dccp.html},
}
```
