__author__ = "Xinyue"
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

np.random.seed(0)
n = 20
N = 30
mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cov = np.eye(n)

for i in range(0, n - 3):
    cov[i, i + 3] = -0.2
    cov[i + 3, i] = -0.2
for i in range(0, n - 6):
    cov[i, i + 6] = 0.6
    cov[i + 6, i] = 0.6
pos = cov > 0
neg = cov < 0
zero = cov == 0

y = np.zeros((n, N))
Sigma = Variable((n, n))

t = Variable(1)
cost = log_det(Sigma) + t
t.value = [1]
Sigma.value = np.eye(n)

emp = np.zeros((n, n))
for k in range(N):
    y[:, k] = np.random.multivariate_normal(mean, cov)
    emp = emp + np.dot(np.matrix(y[:, k]).T, np.matrix(y[:, k])) / N
temp = sum([matrix_frac(y[:, i], Sigma) / N for i in range(N)])
trace_val = temp
constr = [
    trace_val <= t,
    multiply(pos, Sigma) >= 0,
    multiply(neg, Sigma) <= 0,
    multiply(zero, Sigma) == 0,
]
prob = Problem(Minimize(cost), constr)
prob.solve(method="dccp", max_iter=10, solver="SCS")

# plot
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(Sigma.value, interpolation="none")
plt.title("optimize over $\Sigma$ with signs")
plt.subplot(132)
plt.imshow(emp, interpolation="none")
plt.title("empirical covariance")
plt.subplot(133)
plt.imshow(cov, interpolation="none")
plt.title("true covariance")
plt.show()
