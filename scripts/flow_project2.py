"""
Model choice flow script 

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 07.11.2020
"""

# Import configuration file that determines the dataset to plot
from concNoZero_config import *
#from concRaw_config import *

import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.linear_model as lm
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid

from CrossValidation import twoLevelCV_compare, twoLevelCV_compare_PCA, twoLevelCV_single, twoLevelCV_single_PCA

from regression import x_add_features, x_tilda_poly, x_tilda_transform, x_tilda_downSample
from regularization import rlr_validate

# -------------------------------------------------------
# Define basic parameters
# Maybe change this convention?
X_stand = X_stand
y_fromStand = y_fromStand

# -------------------------------------------------------
# Initialize comparison relevant parameters
model = lm.LinearRegression()
K1 = 10
K2 = 10

K3 = 10 # Number of total comparison loops

modelsToCompare = ["Regular Linear Regression", "6 PCA", "Added Features", "Polynomial regression", "Transformed features", "Chosen features", "Transformed + PCA"]
modelErrors = np.zeros((K3, len(modelsToCompare)))

for i in range(K3):
    
    # -------------------------------------------------------
    # Compute error for the regular model
    xIn, yIn =  X_stand, y_fromStand
    modelErrors[i, 0] = twoLevelCV_single(xIn, yIn, model, K1, K2)
    
    # -------------------------------------------------------
    # Compute error for the 6 PCA model
    xIn, yIn =  X_stand, y_fromStand
    modelErrors[i, 1] = twoLevelCV_single_PCA(xIn, yIn, model, K1, K2)
    
    # -------------------------------------------------------
    # Compute error for the added features model
    xIn, yIn =  x_add_features(X_stand, y_fromStand)
    modelErrors[i, 2] = twoLevelCV_single(xIn, yIn, model, K1, K2)
    
    # -------------------------------------------------------
    # Compute error for the polynomial regression model
    xIn, yIn =  x_tilda_poly(X_stand, y_fromStand)
    modelErrors[i, 3] = twoLevelCV_single(xIn, yIn, model, K1, K2)
    
    # -------------------------------------------------------
    # Compute error for the transformed features model
    xIn, yIn =  x_tilda_transform(X_stand, y_fromStand)
    modelErrors[i, 4] = twoLevelCV_single(xIn, yIn, model, K1, K2)
    
    # -------------------------------------------------------
    # Compute error for the chosen features model
    features = np.array([0, 4, 7])
    xIn, yIn = x_tilda_downSample(xIn, yIn, features)
    modelErrors[i, 5] = twoLevelCV_single(xIn, yIn, model, K1, K2)
    
    # -------------------------------------------------------
    # Compute error for the transfomrmed + PCA features model
    xIn, yIn =  x_tilda_transform(X_stand, y_fromStand)
    modelErrors[i, 6] = twoLevelCV_single_PCA(xIn, yIn, model, K1, K2)
    
modelErrorsAvg = np.mean(modelErrors, axis = 0)    

plt.figure(figsize=(15,7))
plt.bar(modelsToCompare, modelErrorsAvg)
plt.ylabel('Estimated generalization error - MSE', fontsize = 16)
plt.xlabel('Model', fontsize = 16)
plt.title('Regression model choice', fontsize = 16)
plt.show()
    





"""
# -----------------------------------------------------------------------------------
# REGRESSION, PART A. 2nd point-------------------------------------------------------

# Add offset attribute
xIn = np.concatenate((np.ones((xIn.shape[0],1)), xIn),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

# Values of lambda
lambdas = np.power(10.,range(-4,9))
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(xIn, yIn, lambdas, 10)

# Display the results for the last cross-validation fold
figure(1, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
# ------------------------------------------------------------------------------------
"""


