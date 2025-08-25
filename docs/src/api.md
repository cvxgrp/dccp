# API Reference

This page provides comprehensive API documentation for the DCCP package, automatically generated from numpy-style docstrings.

## Main Functions

The main DCCP interface provides these core functions:

```{eval-rst}
.. autofunction:: dccp.convexify_constr
.. autofunction:: dccp.convexify_obj
.. autofunction:: dccp.is_dccp
.. autofunction:: dccp.linearize
```

## Problem Solving

```{eval-rst}
.. autoclass:: dccp.problem.DCCP
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dccp.problem.DCCPIter
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: dccp.problem.dccp
```

## Utilities and Settings

```{eval-rst}
.. autoclass:: dccp.utils.DCCPSettings
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dccp.utils.NonDCCPError
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: dccp.utils.is_dccp
```

## Linearization

```{eval-rst}
.. autofunction:: dccp.linearize.linearize
```

## Objective Convexification

```{eval-rst}
.. autofunction:: dccp.objective.convexify_obj
```

## Constraint Convexification

```{eval-rst}
.. autoclass:: dccp.constraint.ConvexConstraint
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: dccp.constraint.convexify_constr
```

## Variable Initialization

```{eval-rst}
.. autofunction:: dccp.initialization.initialize
```
