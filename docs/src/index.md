---
description: DCCP - Disciplined Convex-Concave Programming for solving nonconvex optimization problems.
keywords: convex optimization, nonconvex optimization, DCCP, CVXPY, open source
---

# Disciplined Convex-Concave Programming

```{raw} html
<div style="text-align: center; margin: 20px 0;">
  <a href="https://discord.gg/4urRQeGBCr"
     target="_blank"
     style="display: inline-block;
            background: linear-gradient(135deg, #5865F2, #7289DA);
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(88, 101, 242, 0.3);
            transition: all 0.3s ease;
            border: none;
            margin-right: 10px;"
     onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(88, 101, 242, 0.4)';"
     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(88, 101, 242, 0.3)';">
    🚀 Join our CVXPY Discord Community
  </a>
  <p style="margin-top: 8px; font-style: italic; color: #666;">Discuss with fellow optimization enthusiasts.</p>
</div>
```

DCCP extends CVXPY to solve nonconvex optimization problems using an organized heuristic
for convex-concave programming.

## Documentation contents

- [📦 **Installation**](install.md) - Get started quickly
- [📖 **User Guide**](user-guide.md) - Get the most out of this package
- [⚙️ **Settings**](settings.md) - Algorithm parameters and configuration
- [🔗 **API Reference**](api.md) - Detailed function documentation
- [💡 **Examples**](examples/index.md) - Real-world applications

```{toctree}
:maxdepth: 1
:hidden:

install
user-guide
settings
api
examples/index
citing
```

## Quick Start

```python
import cvxpy as cvx
import dccp

# create a nonconvex problem
x = cvx.Variable(2)
y = cvx.Variable(2)
problem = cvx.Problem(
    cvx.Maximize(cvx.norm(x - y, 2)),
    [0 <= x, x <= 1, 0 <= y, y <= 1]
)

# solve with DCCP
result = problem.solve(method='dccp')
print(f"Optimal value: {result}")
```

## Key Features

- **Extends CVXPY**: Seamlessly integrates with existing CVXPY code
- **Handles Nonconvex Problems**: Solves problems where DCP rules are violated
- **Organized Heuristic**: Systematic approach using convex-concave decomposition
- **Multiple Restarts**: Built-in support for random initialization and restarts

## When to Use DCCP

DCCP is designed for optimization problems where:

- The objective or constraints are nonconvex
- All expressions have known curvature (not "UNKNOWN")
- You need approximate solutions to NP-hard problems

---

**Paper**: [Disciplined Convex-Concave Programming](https://stanford.edu/~boyd/papers/dccp.html)
