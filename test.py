__author__ = 'Xinyue'
from cvxpy import *
import dccp_problem

x = Variable(2)
y = Variable(2)
myprob = Problem(Maximize(norm(x-y,2)), [0<=x, x<=1, 0<=y, y<=1])
print "problem is DCP:", myprob.is_dcp()   # false
print "problem is DCCP:", myprob.is_dccp()  # true
result = myprob.solve(method = 'dccp')
print "========================"
print "x = ", result[1][0]
print "y = ", result[1][1]
print "cost value= ", result[0]

