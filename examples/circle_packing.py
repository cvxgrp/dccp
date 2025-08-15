__author__ = "Xinyue"
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from dccp import is_dccp

np.random.seed(0)
n = 10
r = np.linspace(1, 5, n)

# c = cp.Variable((n, 2))
# constr = [
#     cp.norm(cp.reshape(c[i, :], (1, 2)) - c[i + 1 : n, :], 2, axis=1)
#     >= r[i] + r[i + 1 : n]
#     for i in range(n - 1)
# ]
# prob = cp.Problem(cp.Minimize(cp.max(cp.max(cp.abs(c), axis=1) + r)), constr)


c = cp.Variable((n, 2))
constr = []
for i in range(n - 1):
    for j in range(i + 1, n):
        constr += [cp.norm(c[i, :] - c[j, :]) >= r[i] + r[j]]
prob = cp.Problem(cp.Minimize(cp.max(cp.max(cp.abs(c), axis=1) + r)), constr)
prob.solve(method="dccp")

print("is_dccp:", is_dccp(prob))
prob.solve(method="dccp", solver="ECOS", ep=1e-3, max_slack=1e-3)
print("Optimal value:", prob.value)
print("problem status:", prob.status)
print("circles centers:", c.value)

l = cp.max(cp.max(cp.abs(c), axis=1) + r).value * 2
pi = np.pi
ratio = pi * cp.sum(cp.square(r)).value / cp.square(l).value
print("ratio =", ratio)

# --- plot with plt.Circle ---

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal", adjustable="box")

# draw circles
for i in range(n):
    circle = plt.Circle(
        (c[i, 0].value, c[i, 1].value),  # (x, y) center
        r[i],  # radius
        fill=False,  # outline only
        ec="b",
        lw=1.2,  # edge color/width
    )
    ax.add_patch(circle)

# draw square border
border = plt.Rectangle(
    (-l / 2, -l / 2),  # bottom-left
    l,
    l,  # width, height
    fill=False,
    ec="g",
    lw=1.5,
)
ax.add_patch(border)

# limits and cosmetics
ax.set_xlim(-l / 2, l / 2)
ax.set_ylim(-l / 2, l / 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Circle packing")

fig.savefig("circle_packing.png", dpi=300, bbox_inches="tight")
# fig.show()  # if you want to display interactively
