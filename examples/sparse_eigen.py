__author__ = "Xinyue"
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

np.random.seed(2)
m = 10
n = 10
A0 = np.random.randn(m, n) / 3
A0[:, -1] = A0[:, -2] + A0[:, -3]  # spark(A) = 3
for j in range(6, 8):
    A0[:, j] = A0[:, j - 1] + A0[:, j - 3]
U, Sigma, V = np.linalg.svd(A0, 0)
Sigma += 1
A = np.dot(U, np.dot(np.diag(Sigma), V))

######################smallest singular value
mu_min = Parameter(nonneg=True)
x_min = Variable(n)
# cost_min = norm(A*x_min)+lambd_min*norm(x_min,1)
cost_min = norm(A @ x_min)
constr_min = [norm(x_min, 2) == 1, norm(x_min, 1) <= mu_min]
obj_min = Minimize(cost_min)
prob_min = Problem(obj_min, constr_min)
singular_min_value = []
card_min = []
x_min_result = []
# lambd_min_vals = gamma_vals = np.logspace(-2,2,40)
mu_vals = np.linspace(1, np.sqrt(n), 50)
for val in mu_vals:
    mu_min.value = val
    prob_min.solve(method="dccp")
    singular_min_value.append(norm(A @ x_min).value)
    card_min.append(np.sum(np.abs(x_min.value) >= 1e-2))
    x_min_result.append(x_min.value)

plt.figure(figsize=(5, 5))
# plt.subplot(121)
# for i in range(n):
#    plt.plot(mu_vals, [np.abs(xi[i,0]) for xi in x_min_result])
##for ind in range(len(card)-1):
##    plt.axvspan(lambd_vals[ind], lambd_vals[ind+1], facecolor=str(card[ind]/float(10)), edgecolor = 'none', alpha=0.3)
# plt.xlabel(r'$\mu$', fontsize=16)
# plt.ylabel(r'$\|x_{i}\|$', fontsize=16)
# plt.ylim([0,1])

plt.subplot(111)
card_plot = []
s_value_plot = []
count = []
for ind in range(len(card_min)):
    if card_min[ind] not in card_plot:
        card_plot.append(card_min[ind])
        s_value_plot.append(singular_min_value[ind])
        count.append(1)
    else:
        temp_ind = card_plot.index(card_min[ind])
        s_value_plot[temp_ind] = np.min(
            [s_value_plot[temp_ind], singular_min_value[ind]]
        )
        count[temp_ind] += 1
plt.plot(s_value_plot, card_plot, "r o")
plt.ylim([0, 6])
plt.grid()
plt.xlabel(r"$\|\|Ax\|\|_2/\sigma_{\mathrm{min}}$", fontsize=16)
plt.ylabel("card($x$)", fontsize=16)
print("singular values = ", Sigma)
plt.show()
