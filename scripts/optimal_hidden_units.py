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
hidden_units = np.arange(5,11,5)
opt_n_hidden_units = ann_multiclass_validate(xIn, y_class, C, hidden_units, 10, n_replicates=1, max_iter=10000)
print(f)
# torch.CrossEntropy: combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

"""
def regularized_multinominal_regression():
    '''
    # Fit multinomial logistic regression model
    regularization_strength = 1e-3
    #Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-4, random_state=1, 
                                   penalty='l2', C=1/regularization_strength)
    mdl.fit(X_train,y_train)
    y_test_est = mdl.predict(X_test)
    
    test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)
    
    predict = lambda x: np.argmax(mdl.predict_proba(x),1)
    plt.figure(2,figsize=(9,9))
    visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
    plt.title('LogReg decision boundaries')
    plt.show()
    '''
    return 

"""