"""DCCP package for solving Disciplined Convex-Concave Programming problems."""

import cvxpy as cp

from .constraint import convexify_constr
from .linearize import linearize
from .objective import convexify_obj
from .problem import dccp
from .utils import is_dccp

__all__ = ["convexify_constr", "convexify_obj", "is_dccp", "linearize"]


cp.Problem.register_solve("dccp", dccp)
