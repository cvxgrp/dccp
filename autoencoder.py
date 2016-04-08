__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import examples.extensions.dccp

n = 100
x = np.ones((n+1,1))
x[0:-1] = np.random.randn(n,1)
m = 50
y = Variable(m)
A = Variable(m,n+1)
z = Variable(m+1)
B = Variable(m+1,n)
x_hat = Variable(n)
cost = norm(x[0:-1]-x_hat/4)
obj = Minimize(cost)
############# initial
constr_pre = [y==A*x,y==z[0:-1],z[-1]==1]
# x_hat == 4*B*z
for i in range(n):
    constr_pre += [x_hat[i] + sum_entries(square(B[:,i]-z))== sum_entries(square(B[:,i]+z))]

prob = Problem(obj, constr_pre)
prob.solve(method='dccp', solver = 'SCS')
#############
constr = [y==A*x,z[-1]==1]
# z[i] = sigmoid(y[i])
u = []
v = []
for i in range(m):
    u.append(Variable(2))
    v.append(Variable(2))
    constr += [2*z[i]+norm(v[-1]) == norm(u[-1])]
    constr += [u[-1][0] == 1]
    constr += [u[-1][1] == 1+y[i]*2]
    constr += [v[-1][0] == 1]
    constr += [v[-1][1] == 1-y[i]*2]

# x_hat == 4*B*z
for i in range(n):
    constr += [x_hat[i] + sum_entries(square(B[:,i]-z))== sum_entries(square(B[:,i]+z))]

prob = Problem(obj, constr)
prob.solve(method='dccp', max_iter = 100, solver = 'SCS')

print cost.value
print x[0:-1]
print y.value
print z.value
plt.plot(x[0:-1])
plt.plot(np.array(x_hat.value/4).flatten(),'r--')
plt.show()
