__author__ = "Xinyue"
import numpy as np
import matplotlib.pyplot as plt

# please run sparse_recovery.py first and save the results

n = 100
m = [50, 56, 62, 68, 74, 80]
k = [30, 34, 38, 42, 46, 50]
T = 1

proba = np.zeros((6, 6))
proba_l1 = np.zeros((6, 6))

for t in range(T):
    fname = "sparse_rec/data" + str(t + 1) + ".npy"
    data = np.load(fname)
    proba += data / float(T)
    fname_l1 = "sparse_rec/data" + str(t + 1) + "_l1.npy"
    data_l1 = np.load(fname_l1)
    proba_l1 += data_l1 / float(T)

print proba
print proba_l1
fig = plt.figure(figsize=[14, 5])
ax = plt.subplot(1, 2, 2)
plt.xticks(range(0, len(k)), k)
plt.xlabel(r"cardinality of $x^\mathrm{true}$", fontsize=16)
plt.yticks(range(0, len(m)), m)
plt.ylabel(r"number of measurements $m$", fontsize=16)
a = ax.imshow(proba, interpolation="none")
fig.colorbar(a)
ax.set_title(r"recovery probability with $\ell_{1/2}$-norm", fontsize=18)
ax = plt.subplot(1, 2, 1)
b = ax.imshow(proba_l1, interpolation="none")
fig.colorbar(b)
plt.xticks(range(0, len(k)), k)
plt.xlabel(r"cardinality of $x^\mathrm{true}$", fontsize=16)
plt.yticks(range(0, len(m)), m)
plt.ylabel(r"number of measurements $m$", fontsize=16)
ax.set_title(r"recovery probability with $\ell_{1}$-norm", fontsize=18)
# plt.show()
plt.savefig("sparse_rec.png")
