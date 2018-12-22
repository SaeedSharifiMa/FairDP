import import_data as imp
import dp_postproc as dpalgo1

import numpy as np
import pandas as pd
import cvxopt as cvx
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeRegressor as DT
import matplotlib.pyplot as plt

# Use imp.clean_communities(num_sens), imp.clean_lawschool(num_sens), imp.clean_adult(num_sens)
# to import a data set.
# communities: 1, 2, 6, 8, 13
# lawschool: 1 gender, others look bad.
# adult: 1
X, A, Y = imp.clean_lawschool(1)
base_clf = LR
epsilon = 5
base_gamma_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
            0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.1, 0.2, 0.4, 0.8, 1]
gamma_list = sorted(set(base_gamma_list + list(np.linspace(0.01, 0.2, 51))))
beta = 0.05
num_rounds = 1000

fitted_clf = base_clf(random_state=123).fit(X, Y)
Yhat = fitted_clf.predict(X)
qhat = dpalgo1.dp_postproc.qhat(Yhat, A, Y)
errhat = dpalgo1.dp_postproc.error(Yhat, Y)
unfhat = dpalgo1.dp_postproc.unfairness(Yhat, A, Y)

errtilde = np.empty([len(gamma_list), num_rounds])
unftilde = np.empty([len(gamma_list), num_rounds])
err = []
unf = []

for i in range(len(gamma_list)):
    for j in range(num_rounds):
        errtilde[i,j], unftilde[i,j] = dpalgo1.dp_postproc.eq_odds(Yhat, A, Y, qhat, epsilon, gamma_list[i], beta)
    err.append(np.mean(errtilde[i,errtilde[i,:] != -1]))
    unf.append(np.mean(unftilde[i,unftilde[i,:] != -1]))

plt.plot(err, unf, 'ro')
plt.xlabel('error')
plt.ylabel('fairness violation')
plt.title('dp_postproc, lawschool, base_clf = LR, epsilon = %i' %epsilon)
plt.savefig('output.png')
plt.show()