__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

n=100
m=40
k=[5,10,15,20,25]
np.random.seed(0)
T=1
proba = np.zeros((len(k),1))
proba_l1 = np.zeros((len(k),1))

for time in range(T):
    x_star = np.random.randn(n,1)*10
    ind_star = np.random.permutation(n)
    A = np.random.randn(m,n)
    for kk in k:
        ind = ind_star[0:kk]
        x0 = np.zeros((n,1))
        x0[ind] = x_star[ind]
        M = max(np.abs(x0))
        y = np.dot(A,x0)
        # recovery with boolean constraints
        x = Variable(n)
        t = Variable(n)
        t.value = np.ones((n,1))
        cost = sum_entries(t)
        constr = [abs(x)<=M*t, A*x==y, power(t,2)==t]
        obj = Minimize(cost)
        prob = Problem(obj, constr)
        prob.solve(method='dccp',solver='MOSEK')

        if pnorm(x - x0,2).value/pnorm(x0,2).value <=1e-2:
            indk = k.index(kk)
            proba[indk] += 1/float(T)

        #l1 minimization
        xl1 = Variable(n,1)
        cost = pnorm(xl1,1)
        obj = Minimize(cost)
        constr = [A*xl1==y]
        prob = Problem(obj, constr)
        result = prob.solve()
        if pnorm(xl1 - x0,2).value/pnorm(x0,2).value <=1e-2:
            indk = k.index(kk)
            proba_l1[indk] += 1/float(T)
        print "time =", time,"k =",kk,"relative error = ", pnorm(x - x0,2).value/pnorm(x0,2).value
        print "time =", time,"k =",kk,"relative error = ", pnorm(xl1 - x0,2).value/pnorm(x0,2).value

plt.figure(figsize=[5, 5])
plt.ylim([0,1])
plt.xlabel("cardinality")
plt.ylabel("recovery probability")
plt.plot(k,proba,'r-o')
plt.plot(k,proba_l1,'b--^')
plt.legend(["minimize cardinality","minimize $\ell_1$ norm"])
plt.show()

