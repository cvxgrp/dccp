__author__ = "Xinyue"
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp.problem

np.random.seed(0)

n = 100
m = [50, 56, 62, 68, 74, 80]
k = [30, 34, 38, 42, 46, 50]
T = 1
proba = np.zeros((len(m), len(k)))
proba_l1 = np.zeros((len(m), len(k)))

for time in range(T):
    for kk in k:
        x0 = np.zeros((n, 1))
        ind = np.random.permutation(n)
        ind = ind[0:kk]
        x0[ind] = np.random.randn(kk, 1) * 10
        x0 = np.abs(x0)
        for mm in m:
            A = np.random.randn(mm, n)
            y = np.dot(A, x0)
            # sqrt of 0.5-norm minimization
            x_pos = Variable(shape=((n, 1)), nonneg=True)
            x_pos.value = np.ones((n, 1))
            cost = reshape(sum(sqrt(x_pos), axis=0), (1, 1))
            prob = Problem(Minimize(cost), [A @ x_pos == y])
            result = prob.solve(method="dccp", solver="SCS")

            if (
                x_pos.value is not None
                and pnorm(x_pos - x0, 2).value / pnorm(x0, 2).value <= 1e-2
            ):
                indm = m.index(mm)
                indk = k.index(kk)
                proba[indm, indk] += 1 / float(T)

            # l1 minimization
            xl1 = Variable((n, 1))
            cost = pnorm(xl1, 1)
            obj = Minimize(cost)
            constr = [A @ xl1 == y]
            prob = Problem(obj, constr)
            result = prob.solve()
            if pnorm(xl1 - x0, 2).value / pnorm(x0, 2).value <= 1e-2:
                indm = m.index(mm)
                indk = k.index(kk)
                proba_l1[indm, indk] += 1 / float(T)
            if x_pos.value is not None:
                print(
                    "time=",
                    time,
                    "k=",
                    kk,
                    "m=",
                    mm,
                    "relative error = ",
                    pnorm(x_pos - x0, 2).value / pnorm(x0, 2).value,
                )
            else:
                print("time=", time, "k=", kk, "m=", mm, "relative error = ", 1.0)
            print(
                "time=",
                time,
                "k=",
                kk,
                "m=",
                mm,
                "relative error = ",
                pnorm(xl1 - x0, 2).value / pnorm(x0, 2).value,
            )
print(proba)
print(proba_l1)
fig = plt.figure(figsize=[14, 5])
ax = plt.subplot(1, 2, 1)
plt.xticks(range(0, len(k)), k)
plt.xlabel("cardinality")
plt.yticks(range(0, len(m)), m)
plt.ylabel("number of measurements")
a = ax.imshow(proba, interpolation="none")
fig.colorbar(a)
ax.set_title("probability of recovery")
ax = plt.subplot(1, 2, 2)
b = ax.imshow(proba_l1, interpolation="none")
fig.colorbar(b)
plt.xticks(range(0, len(k)), k)
plt.xlabel("cardinality")
plt.yticks(range(0, len(m)), m)
plt.ylabel("number of measurements")
ax.set_title("probability of recovery")
plt.show()

# to run sparse_recovery_plot.py later, please save the following files
# np.save("sparse_rec/data100", proba)
# np.save("sparse_rec/data100_l1", proba_l1)
