__author__ = "Xinyue"
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dccp

n = 50
l = 10
m = 5
a = np.matrix([0, 0]).T
b = np.matrix([l, l]).T
d = 2
p = np.matrix([[2, 4.5, 6, 7, 8.5], [2.2, 5, 8, 6, 9]])
r = [1, 0.8, 0.4, 1.4, 0.5]

x = []
for i in range(n + 1):
    x += [Variable((d, 1))]
L = Variable(1)
constr = [x[0] == a, x[n] == b]
cost = L
for i in range(n):
    constr += [norm(x[i] - x[i + 1]) <= L / n]
    for j in range(m):
        constr += [norm(x[i] - p[:, j]) >= r[j]]
prob = Problem(Minimize(cost), constr)
print("begin to solve")
result = prob.solve(method="dccp")
print("end")

# plot
fig, ax = plt.subplots(figsize=(5, 5))
for i in range(m):
    circle = mpatches.Circle(np.array(p[:, i]), r[i], ec="none")
    ax.add_patch(circle)
ax1 = [xx.value[0] for xx in x]
ax2 = [xx.value[1] for xx in x]
plt.plot(ax1, ax2, "r-+")
plt.ylim(0, 10)
plt.xlim(0, 10)
ax.grid()
plt.show()
