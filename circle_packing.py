__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import examples.extensions.dccp
from test import iter_dccp
n = 10
r = np.linspace(1,5,n)

c = Variable(n,2)
constr = []
for i in range(n-1):
    for j in range(i+1,n):
        constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
prob = Problem(Minimize(max_entries(row_norm(c,'inf')+r)), constr)
#prob.solve(method = 'dccp')
iter_dccp(prob)

l = max_entries(row_norm(c,'inf')+r).value*2
pi = np.pi
ratio = pi*sum_entries(square(r)).value/square(l).value
print "ratio=", ratio
#### plot ####
plt.figure(figsize=(5,5))
circ = np.linspace(0,2*pi)
x_border = [-l/2, l/2, l/2, -l/2, -l/2]
y_border = [-l/2, -l/2, l/2, l/2, -l/2]
for i in xrange(n):
    #plt.plot(c[i][0].value+r[i]*np.cos(circ),c[i][1].value+r[i]*np.sin(circ),'b')
    plt.plot(c[i,0].value+r[i]*np.cos(circ),c[i,1].value+r[i]*np.sin(circ),'b')
plt.plot(x_border,y_border,'g')
plt.axes().set_aspect('equal')
plt.xlim([-l/2,l/2])
plt.ylim([-l/2,l/2])
plt.show()