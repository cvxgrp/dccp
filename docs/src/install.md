# Installation

## Prerequisites

You should first install [CVXPY 1.5](http://www.cvxpy.org/) or greater.

## Quick Install

Install DCCP from [PyPI](https://pypi.org/project/dccp/) using pip:

```bash
pip install dccp
```

## Development Install

To install the development version, clone the repository and install in development mode:

```bash
git clone https://github.com/cvxgrp/dccp.git
cd dccp
pip install -e .
```

## Verify Installation

You can verify the installation by running a simple example:

```python
import cvxpy as cvx
import dccp

# create a simple problem
x = cvx.Variable(2)
y = cvx.Variable(2)
problem = cvx.Problem(cvx.Maximize(cvx.norm(x - y, 2)), [0 <= x, x <= 1, 0 <= y, y <= 1])

# check if it's DCCP
print("Problem is DCCP:", dccp.is_dccp(problem))
```
