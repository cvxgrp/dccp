import cvxpy as cp
import numpy as np

# Test what happens with tiny negative values
x = cp.Variable(1)
x.value = [-4.724727e-13]

print(f"x.value: {x.value}")
print(f"sqrt(x).value: {cp.sqrt(x).value}")
print(f"Is sqrt(x).value nan?: {np.isnan(cp.sqrt(x).value)}")

# Test with a small positive value
x.value = [4.724727e-13]
print(f"x.value (positive): {x.value}")
print(f"sqrt(x).value (positive): {cp.sqrt(x).value}")

# Test with zero
x.value = [0.0]
print(f"x.value (zero): {x.value}")
print(f"sqrt(x).value (zero): {cp.sqrt(x).value}")
