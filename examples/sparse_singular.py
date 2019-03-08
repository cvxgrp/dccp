__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp.problem

np.random.seed(3)
m = 10
n = 10
A0 = np.random.randn(m,n)
U, Sigma, V = np.linalg.svd(A0,0)
Sigma = Sigma/Sigma[-1]
A = np.dot(U, np.dot(np.diag(Sigma), V))

# smallest singular value
mu = Parameter(nonneg=True)
x = Variable(n)
cost = norm(A*x)
constr = [norm(x,2)==1, norm(x,1) <= mu]
obj = Minimize(cost)
prob = Problem(obj,constr)
singular_value = []
card = []
x_result = []
mu_vals = np.linspace(1,np.sqrt(n),50)
for val in mu_vals:
    mu.value = val
    prob.solve(method='dccp', max_iter=50)
    singular_value.append(norm(A*x).value)
    card.append(np.sum(np.abs(x.value)>=1e-2))
    x_result.append(x.value)

plt.figure(figsize = (5,5))
for ind in range(len(card)):
    plt.plot(card[ind],singular_value[ind],'r o')
plt.xlim([0,n])
plt.yscale('log')
plt.grid()
plt.ylabel(r'$\|\|Ax\|\|_2/\sigma_{\mathrm{min}}$', fontsize=16)
plt.xlabel('card($x$)', fontsize=16)
plt.show()