"""DCCP package."""

import matplotlib.pyplot as plt
import numpy as np
from cvxpy import *

n = 128
m = 3 * n
# data
x0r = np.random.rand(n, 1)
x0i = np.random.rand(n, 1)
Ar = np.random.rand(m, n)
Ai = np.random.rand(m, n)
yr = np.dot(Ar, x0r) + np.dot(Ai, x0i)
yi = np.dot(Ar, x0i) - np.dot(Ai, x0r)
y = np.power(yr, 2) + np.power(yi, 2)
y = np.power(y, 0.5)
# solve
xr = Variable((n, 1))
xi = Variable((n, 1))
x = Variable((2, n))
z = []
constr = []
c = np.matrix([[0, 1], [-1, 0]])
for k in range(m):
    z.append(Variable((2, 1)))
    z[-1].value = -np.random.rand(2, 1)
    constr.append(norm(z[-1]) == y[k])
    constr += [
        z[-1] == x @ np.reshape(Ar[k, :], (n, 1)) + c @ x @ np.reshape(Ai[k, :], (n, 1))
    ]
prob = Problem(Minimize(0), constr)
result = prob.solve(method="dccp")

# plot
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 8))
tan = np.array(x[0, :].value / x[1, :].value)
angle = np.arctan(tan)
tan0 = x0r / x0i
angle0 = np.arctan(tan0)
ax0.plot(angle0)
ax0.plot(angle, "r")
plt.xlim([0, 128])
ax0.legend(["phase of the original signal", "phase of the recovered signal"])
ax1.plot(np.array(np.power(x0r, 2) + np.power(x0i, 2)))
ax1.plot(np.array(np.power(x[0, :].value, 2) + np.power(x[1, :].value, 2)), "r--")
plt.xlim([0, 128])
ax1.legend(["amplitude of the original signal", "amplitude of the recovered signal"])
plt.show()
