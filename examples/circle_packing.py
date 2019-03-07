__author__ = 'Xinyue'
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import dccp
np.random.seed(0)
n = 4
r = np.linspace(1, 5, n)

c = cvx.Variable((n,2))
constr = []
for i in range(n-1):
    for j in range(i+1, n):
        constr.append(cvx.norm(cvx.vec(c[i,:]-c[j,:]), 2) >= r[i]+r[j])
prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.max(cvx.abs(c), axis=1) + r)), constr)
#prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.norm(c, "inf", axis=1) + r)), constr)
prob.solve(method = 'dccp', tau=0.000005, ccp_times = 1)

l = cvx.max(cvx.max(cvx.abs(c),axis=1)+r).value*2
pi = np.pi
ratio = pi*cvx.sum(cvx.square(r)).value/cvx.square(l).value
print "ratio =", ratio
print prob.status

# plot
plt.figure(figsize=(5,5))
circ = np.linspace(0,2*pi)
x_border = [-l/2, l/2, l/2, -l/2, -l/2]
y_border = [-l/2, -l/2, l/2, l/2, -l/2]
for i in xrange(n):
    plt.plot(c[i,0].value+r[i]*np.cos(circ),c[i,1].value+r[i]*np.sin(circ),'b')
plt.plot(x_border,y_border,'g')
plt.axes().set_aspect('equal')
plt.xlim([-l/2,l/2])
plt.ylim([-l/2,l/2])
plt.show()