"""
Created on Fri Nov 13 01:13:58 2020

@author: CARLOS
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from scipy import stats
from ANN_functions import *
from concNoZero_config import *
from regression import x_add_features

        
xIn,yIn = x_add_features(X_stand, y_fromStand)

#%%
#-------- REGRESSION ANN -------------------------

# Range of hidden units
hidden_units = np.arange(5,11,5)
opt_n_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units = annr_validate(xIn, yIn, hidden_units, 10, n_replicates=1, max_iter=10000)

print(opt_n_hidden_units)

plt.figure(1, figsize=(8,8))
plt.title('Optimal number of hidden units: {}'.format(opt_n_hidden_units))
plt.plot(hidden_units,train_err_vs_hidden_units.T,'b.-',hidden_units,test_err_vs_hidden_units.T,'r.-')
plt.xlabel('Number of hidden units')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()

#%%
#-------MULTI-CLASS ANN CLASSIFICATION ---------------------------

# Range of hidden units
hidden_units = np.arange(5,12,5)
opt_n_hidden_units = ann_multiclass_classification(xIn, y_class_encoding, C, hidden_units, 10, n_replicates=1, max_iter=10000)

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# #############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()