__author__ = 'Xinyue'
from cvxpy import *
import dccp
import numpy as np
from dccp.problem import is_dccp

x = Variable(3)
y = Variable(3)
#myprob = Problem(Maximize(norm(x-y,2)), [0<=x, x<=1, 0<=y, y<=1])
myprob = Problem(Minimize(sum_entries(power(x-y,0.5))), [0<=x, x<=1, 0<=y, y<=1])
print "problem is DCP:", myprob.is_dcp()   # false
print "problem is DCCP:", is_dccp(myprob)  # true
result = myprob.solve(method = 'dccp',solver = 'MOSEK')
print "========================"
print "x =", x.value
print "y =", y.value
print "cost value =", result[0]

