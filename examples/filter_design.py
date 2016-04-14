__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

N = 100
n = 10
l_pass = N/2-15
l_stop = N/2
L_pass = 0.9
U_pass = 1.1
omega = np.linspace(0,np.pi,N)
expo = []
for omg_ind in range(N):
    cosine = np.cos(np.dot(omega[omg_ind],range(0,n,1)))
    sine = np.sin(np.dot(omega[omg_ind],range(0,n,1)))
    expo.append(np.matrix([cosine, sine]))

h = Variable(n)
U_stop = Variable()
constr = []
for l in range(N):
    if l < l_pass:
        constr += [norm(expo[l]*h,2) >= L_pass]
    if l < l_stop:
        constr += [norm(expo[l]*h,2) <= U_pass]
    else:
        constr += [norm(expo[l]*h,2) <= U_stop]
prob = Problem(Minimize(U_stop), constr)
result = prob.solve(method = 'dccp')

#plot
plt.figure(figsize=(5,5))
lowerbound = np.zeros((N,1))
lowerbound[0:l_pass] = L_pass*np.ones((l_pass,1))
upperbound = np.ones((N,1))*U_stop.value
upperbound[0:l_stop] = U_pass*np.ones((l_stop,1))
plt.plot(omega,upperbound,'--')
plt.plot(omega,lowerbound,'--')
H_amp = np.zeros((N,1))
for l in range(N):
    H_amp[l] = norm(expo[l]*h,2).value
plt.plot(omega,H_amp)
plt.xlabel("frequency")
plt.ylabel("amplitude")
#plt.arrow(omg[omg>=np.pi/2][0], 0.2, 0, delta_u.value*0.75, head_width=0.02, head_length=-delta_u.value*0.25, fc='k', ec='k')
#plt.arrow(omg[omg>=np.pi/2][0], 0.2+delta_u.value, 0, -delta_u.value*0.75, head_width=0.02, head_length=-delta_u.value*0.25, fc='k', ec='k')
#plt.text(1.7,delta_u.value*0.75+0.2,"$\Delta_u$")
plt.show()



