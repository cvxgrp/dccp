"""DCCP package."""

"""DCCP package."""
import matplotlib.pyplot as plt
import numpy as np
from cvxpy import *

np.random.seed(0)

T = 100
l = 6.0
m = 1
v_max = 0.15
d_min = 0.6
A = np.matrix([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 0.95, 0], [0, 0, 0, 0.95]])
B = np.matrix([[0, 0], [0, 0], [0.1 / float(m), 0], [0, 0.1 / float(m)]])
C = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
d = 2  # dim of space
n = 2  # number of systems
x_ini = np.array([[1, 4, 0.5, 0.5], [3, 5, -0.5, 0]])
x_end = np.array([[3, 3, 0, -0.5], [3.5, 3.5, 0.5, -0.5]])
f_max = 0.5

u = []
y = []
x = []
for i in range(n):
    u.append([])
    y.append([])
    x.append([])
cost = 0
constr = []

for i in range(n):
    for t in range(T):
        u[i] += [Variable(d)]
        constr += [norm(u[i][-1], "inf") <= f_max]
        cost += pnorm(u[i][-1], 1)
        y[i] += [Variable(d)]
        x[i] += [Variable(2 * d)]
        constr += [y[i][-1] == C @ x[i][-1]]
    constr += [x[i][0] == x_ini[i]]
    constr += [x[i][-1] == x_end[i]]
for i in range(n):
    for t in range(T - 1):
        constr += [x[i][t + 1] == A @ x[i][t] + B @ u[i][t]]

for t in range(T):
    for i in range(n - 1):
        for j in range(i + 1, n):
            constr += [norm(y[i][t] - y[j][t]) >= d_min]
prob = Problem(Minimize(cost), constr)
prob.solve(method="dccp", ep=1e-1)
###################################################
################ without avoidence ###############
u_c = []
y_c = []
x_c = []
for i in range(n):
    u_c.append([])
    y_c.append([])
    x_c.append([])
cost_c = 0
constr_c = []
for i in range(n):
    for t in range(T):
        u_c[i] += [Variable(d)]
        constr += [pnorm(u_c[i][-1], "inf") <= f_max]
        cost_c += pnorm(u_c[i][-1], 1)
        y_c[i] += [Variable(d)]
        x_c[i] += [Variable(2 * d)]
        constr_c += [y_c[i][-1] == C @ x_c[i][-1]]
    constr_c += [x_c[i][0] == x_ini[i]]
    constr_c += [x_c[i][-1] == x_end[i]]
for i in range(n):
    for t in range(T - 1):
        constr_c += [x_c[i][t + 1] == A @ x_c[i][t] + B @ u_c[i][t]]
prob_c = Problem(Minimize(cost_c), constr_c)
prob_c.solve()

# plot
plt.figure(figsize=(20, 5))
plt.subplot(132)
ax = [xx.value[0] for xx in y[0]]
ay = [xx.value[1] for xx in y[0]]
plt.plot(ax, ay, "b-")
plt.quiver(
    x_ini[0][0],
    x_ini[0][1],
    x_ini[0][2],
    x_ini[0][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
plt.quiver(
    x_ini[1][0],
    x_ini[1][1],
    x_ini[1][2],
    x_ini[1][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
plt.quiver(
    x_end[0][0],
    x_end[0][1],
    x_end[0][2],
    x_end[0][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
plt.quiver(
    x_end[1][0],
    x_end[1][1],
    x_end[1][2],
    x_end[1][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
bx = [xx.value[0] for xx in y[1]]
by = [xx.value[1] for xx in y[1]]
plt.plot(bx, by, "b--")
plt.axis("equal")
plt.xlim(0.5, 4.5)
plt.ylim(2, 6)

plt.subplot(131)
ax = [xx.value[0] for xx in y_c[0]]
ay = [xx.value[1] for xx in y_c[0]]
plt.plot(ax, ay, "r")
plt.quiver(
    x_ini[0][0],
    x_ini[0][1],
    x_ini[0][2],
    x_ini[0][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
plt.quiver(
    x_ini[1][0],
    x_ini[1][1],
    x_ini[1][2],
    x_ini[1][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
plt.quiver(
    x_end[0][0],
    x_end[0][1],
    x_end[0][2],
    x_end[0][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
plt.quiver(
    x_end[1][0],
    x_end[1][1],
    x_end[1][2],
    x_end[1][3],
    units="xy",
    scale=2,
    zorder=3,
    color="black",
    width=0.01,
    headwidth=4.0,
    headlength=5.0,
)
bx = [xx.value[0] for xx in y_c[1]]
by = [xx.value[1] for xx in y_c[1]]
plt.plot(bx, by, "r--")
plt.axis("equal")
plt.xlim(0.5, 4.5)
plt.ylim(2, 6)

distance = []
for t in range(T):
    distance.append(pnorm(y[0][t] - y[1][t], 2).value)
distance_c = []
for t in range(T):
    distance_c.append(pnorm(y_c[0][t] - y_c[1][t], 2).value)

plt.subplot(133)
plt.plot(range(0, T), distance)
plt.plot(range(0, T), distance_c, "r-.")
plt.plot(range(0, T), d_min * np.ones((T, 1)), "k--")
plt.legend(["with avoidance", "without avoidance"], loc=0)
plt.ylabel("$\|\|y^0_t - y^1_t\|\|_2$")
plt.xlabel("$t$")
plt.show()
