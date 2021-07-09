__author__ = "Xinyue"
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import dccp

np.random.seed(0)
n = 10
r = np.linspace(1, 5, n)

c = cvx.Variable((n, 2))
constr = []
for i in range(n - 1):
    constr.append(cvx.norm(cvx.reshape(c[i, :], (1, 2)) - c[i + 1: n, :], 2, axis=1) >= r[i] + r[i + 1: n])
prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.max(cvx.abs(c), axis=1) + r)), constr)
prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)

l = cvx.max(cvx.max(cvx.abs(c), axis=1) + r).value * 2
pi = np.pi
ratio = pi * cvx.sum(cvx.square(r)).value / cvx.square(l).value
print("ratio =", ratio)
# plot
plt.figure(figsize=(5, 5))
circ = np.linspace(0, 2 * pi)
x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
for i in range(n):
    plt.plot(
        c[i, 0].value + r[i] * np.cos(circ), c[i, 1].value + r[i] * np.sin(circ), "b"
    )
plt.plot(x_border, y_border, "g")
plt.axes().set_aspect("equal")
plt.xlim([-l / 2, l / 2])
plt.ylim([-l / 2, l / 2])
plt.show()
