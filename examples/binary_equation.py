__author__ = 'Xinyue'
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import mosek
import sys
import dccp
# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

n= 100
T = 10
noise_sigma = np.sqrt(n/np.linspace(1,17,8))
error = np.zeros((len(noise_sigma),T))
er_bit_rate = np.zeros((len(noise_sigma),T))
error_M = np.zeros((len(noise_sigma),T))
er_M_bit_rate = np.zeros((len(noise_sigma),T))
dis = np.zeros((len(noise_sigma),T))

x = Variable(n)
constr = [square(x) == 1]

for t in range(T):
    A = np.random.randn(n,n)
    x0 = np.random.randint(0,2,size = (n,1))
    x0 = x0*2-1
    for noise_ind in range(len(noise_sigma)):
        v = np.random.randn(n,1)*noise_sigma[noise_ind]
        y = np.dot(A,x0) + v
        # solve by dccp
        prob = Problem(Minimize(norm(A*x-y)), constr)
        result = prob.solve(method='dccp')
        solution = [x_value.value for x_value in x]
        recover = np.matrix(solution)
        recover = np.transpose(recover)
        error[noise_ind,t] = np.linalg.norm(recover-x0,2)
        er_bit_rate[noise_ind,t] = sum(np.abs(recover-x0)>=1)
        print "error=", error[noise_ind,t] , "error bit rate = ", er_bit_rate[noise_ind,t]
        ################################################################################################################
        # solve by MOSEK
        # Make a MOSEK environment
        env = mosek.Env ()
        # Attach a printer to the environment
        env.set_Stream (mosek.streamtype.log, streamprinter)
        # Create a task
        task = env.Task(0,0)
        # Attach a printer to the task
        task.set_Stream (mosek.streamtype.log, streamprinter)
        # Objective coefficients
        c = [1]
        # Bound keys for variables
        bkx = [mosek.boundkey.fr]
        for i in range(n):
            bkx.append(mosek.boundkey.ra)
            c.append(0)
        for i in range(n):
            bkx.append(mosek.boundkey.fr)
            c.append(0)
        # Bound values for variables
        inf = 0
        blx = [-inf]
        bux = [+inf]
        for i in range(n):
            blx.append(0)
            bux.append(1)
        for i in range(n):
            blx.append(-inf)
            bux.append(+inf)
        # Bound keys for constraints
        bkc = []
        for j in range(n):
            bkc.append(mosek.boundkey.fx)
        # Bound values for constraints
        blc = np.array(y+np.dot(A,np.ones((n,1)))).flatten()
        buc = np.array(y+np.dot(A,np.ones((n,1)))).flatten()
        asub = []
        for i in range(2*n+1):
            asub.append(np.transpose([ ii for ii in range(0,n)]))
        aval = np.zeros((n,1))
        aval = np.append(aval,2*A,axis=1)
        aval = np.append(aval,-np.eye(n),axis=1)
        numvar = len(bkx)
        numcon = len(bkc)
        task.appendcons(numcon)
        task.appendvars(numvar)
        for j in range(numvar):
        # Set the linear term c_j in the objective.
            task.putcj(j,c[j])
            # Set the bounds on variable j
            task.putvarbound(j,bkx[j],blx[j],bux[j])
            # Input column j of A
            task.putacol(j,              # Variable (column) index.
                     asub[j],            # Row index of non-zeros in column j.
                     aval[:,j])            # Non-zero Values of column j.
        # Set the bounds on constraints.
        for i in range(numcon):
            task.putconbound(i,bkc[i],blc[i],buc[i])
        task.putconboundslice(0,numcon,bkc,blc,buc);
        # add cone
        cone = [0]
        for ii in range(n+1,2*n+1):
            cone.append(ii)
        task.appendcone(mosek.conetype.quad,
                  0.0,
                  cone)
        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)
         # Define variables to be integers
        type = []
        for ii in range(1,n):
            type.append(mosek.variabletype.type_int)
        task.putvartypelist([ ii for ii in range(1,n) ],
                      type)
        # Solve the problem
        task.optimize()
        # Print a summary containing information
        # about the solution for debugging purposes
        #task.solutionsummary(mosek.streamtype.msg)
        # Output a solution
        xx = np.zeros(numvar, float)
        task.getxx(mosek.soltype.itg,xx)
        error_M[noise_ind,t] = np.linalg.norm(2*xx[1:n+1]-1-np.transpose(x0),2)
        er_M_bit_rate[noise_ind,t] = sum(sum(np.abs(2*xx[1:n+1]-1-np.transpose(x0))>=1))
        print "error=", error_M[noise_ind,t] , "error bit rate = ", er_M_bit_rate[noise_ind,t]
        dis[noise_ind,t] = np.linalg.norm(2*xx[1:n+1]-1-solution,2)
        print "difference = ", dis[noise_ind,t]


plt.figure(figsize = (5,5))
#plt.subplot(121)
#for noise_ind in range(len(noise_sigma)):
#    proba = np.zeros((100,1))
#    max_dis = max(dis[noise_ind,:])
#    x_dis = max_dis*np.linspace(0,1,len(proba))
#    for p in range(len(proba)):
#        proba[p] = sum(dis[noise_ind,:]<=x_dis[p])/float(T)
#    plt.plot(x_dis,proba)
#plt.xlabel('distance')
#plt.ylabel('CDF')
#plt.subplot(122)
plt.plot(n/np.square(noise_sigma),np.sum(er_bit_rate,axis=1)/T,'b-o')
plt.plot(n/np.square(noise_sigma),np.sum(er_M_bit_rate,axis=1)/T,'g-^')
plt.xlabel('$n/\sigma^2$')
plt.ylabel('bit error rate')
plt.legend(["dccp", "global optimal"],loc = 0)
plt.show()




