__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import examples.extensions.dccp
from test import iter_dccp

np.random.seed(0)
n = 5
N = 10
#mean = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#mean = [0,0,0,0,0,0,0,0,0,0]
mean = [0,0,0,0,0]
cov = np.eye(n)

for i in range(0,n-3):
    cov[i,i+3] = -0.2
    cov[i+3,i] = -0.2
for i in range(0,n-6):
    cov[i,i+6] = 0.6
    cov[i+6,i] = 0.6
pos = cov>0
neg = cov<0
zero = cov==0

y = np.zeros((n,N))
Sigma = Variable(n,n)
#Sigma.value = np.eye(n)
t = Variable(1)
cost = log_det(Sigma) + t

#left = 0
emp = np.zeros((n,n))
for k in range(N):
    y[:,k] = np.random.multivariate_normal(mean,cov)
#    left = left + matrix_frac(y[:,k],Sigma)
    emp = emp + np.dot(np.matrix(y[:,k]).T,np.matrix(y[:,k]))/N
trace_val = trace(sum([matrix_frac(y[:,i], Sigma)/N for i in range(N)]))
constr = [trace_val<=t, mul_elemwise(pos,Sigma) >= 0, mul_elemwise(neg,Sigma) <= 0, mul_elemwise(zero,Sigma) == 0]
prob = Problem(Minimize(cost), constr)
#prob.solve(method='dccp', solver = 'SCS', max_iter = 20)
iter_dccp(prob)
'''
S = Variable(n,n)
cost = log_det(S)-trace(S*emp)
obj = Maximize(cost)
prob_S = Problem(obj)
prob_S.solve()
'''
mask = cov==0
mask = -mask

plt.figure(figsize = (15,5))
plt.subplot(131)
plt.imshow(Sigma.value,interpolation='none')
plt.title('optimize over $\Sigma$ with signs')
plt.colorbar()
#plt.subplot(222)
#plt.imshow(np.linalg.inv(S.value),interpolation='none')
#plt.title('optimize over $\Sigma^{-1}$')
#plt.colorbar()
plt.subplot(132)
plt.imshow(emp,interpolation='none')
plt.title('empirical covariance')
plt.colorbar()
plt.subplot(133)
plt.imshow(cov,interpolation='none')
plt.title('true covariance')
plt.colorbar()
plt.show()
