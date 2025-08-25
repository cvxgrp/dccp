# Settings

## Solve Method Parameters

The DCCP solver accepts various parameters to control the algorithm behavior:

| Name | Type | Description | Allowed values | Default value |
|------|------|-------------|----------------|---------------|
| `max_iter` | `int` | Maximum number of iterations in the CCP algorithm | $(1, \infty)$ | 100 |
| `max_iter_damp` | `int` | Maximum number of damping iterations when convergence fails | $(1, \infty)$ | 10 |
| `tau_ini` | `float` | Initial value for tau parameter (trades off constraints vs objective) | $(0, \infty)$ | 0.005 |
| `mu` | `float` | Rate at which tau increases during the algorithm | $(1, \infty)$ | 1.2 |
| `tau_max` | `float` | Upper bound for tau parameter | $(0, \infty)$ | 1e8 |
| `k_ini` | `int` | Number of random projections for variable initialization | $(1, \infty)$ | 1 |
| `k_ccp` | `int` | Number of random restarts for the CCP algorithm | $(1, \infty)$ | 1 |
| `max_slack` | `float` | Maximum slack variable value for convergence | $(0, \infty)$ | 1e-3 |
| `ep` | `float` | Convergence tolerance for objective value changes | $(0, \infty)$ | 1e-5 |
| `std` | `float` | Standard deviation for random variable initialization | $(0, \infty)$ | 10.0 |
| `seed` | `int \| None` | Random seed for reproducible results | $\mathbb{Z} \cup \{\text{None}\}$ | None |
| `verify_dccp` | `bool` | Whether to verify DCCP compliance before solving | $\{0, 1\}$ | 1 |

## Parameter Usage

You can pass these parameters to the solve method like so:

```python
import cvxpy as cvx
import dccp

# create your problem
x = cvx.Variable(2)
problem = cvx.Problem(cvx.Maximize(cvx.norm(x, 2)), [cvx.norm(x, 1) <= 1])

# solve with custom parameters
result = problem.solve(
    method='dccp',
    max_iter=200,
    tau_ini=0.01,
    k_ccp=5,
    seed=42
)
```
