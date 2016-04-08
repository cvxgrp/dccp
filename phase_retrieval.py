__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import examples.extensions.dccp

n=128
m=3*n
# data
x0r = np.random.rand(n,1)
x0i = np.random.rand(n,1)
Ar = np.random.rand(m,n)
Ai = np.random.rand(m,n)
yr = np.dot(Ar,x0r)+np.dot(Ai,x0i)
yi = np.dot(Ar,x0i)-np.dot(Ai,x0r)
y = np.power(yr,2) + np.power(yi,2)
y = np.power(y,0.5)
# solve
xr = Variable(n)
#xr.value = np.ones((n,1))
xi = Variable(n)
x = Variable(2,n)
#xi.value = np.ones((n,1))
#z = Variable(2,m)
#z.value = np.ones((2,m))
z = []
constr = []
c = np.matrix([[0,1],[-1,0]])
for k in range(m):
    z.append(Variable(2))
    z[-1].value = -np.random.rand(2,1)
    constr.append(norm(z[-1]) == y[k])
    constr += [z[-1] == x*Ar[k,:] + c*x*Ai[k,:]]
    #constr += [norm(x*Ar[k,:].T + c*x*Ai[k,:].T) == y[k]]
prob = Problem(Minimize(0),constr)
result = prob.solve(method='dccp',solver = 'MOSEK')
#plot
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10,8))
tan = np.array(x[0,:].value/x[1,:].value)[0]
angle = np.arctan(tan)
tan0 = x0r/x0i
angle0 = np.arctan(tan0)
ax0.plot(angle0)
ax0.plot(angle,'r')
plt.xlim([0,128])
ax0.legend(["phase of the original signal", "phase of the recovered signal"])
ax1.plot(np.array(np.power(x0r,2)+np.power(x0i,2)))
ax1.plot(np.array(np.power(x[0,:].value,2)+np.power(x[1,:].value,2))[0],'r--')
plt.xlim([0,128])
ax1.legend(["amplitude of the original signal", "amplitude of the recovered signal"])
plt.show()

