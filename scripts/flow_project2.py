"""
General flow script

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 30.10.2020
"""

# Import configuration file that determines the dataset to plot
from concNoZero_config import *

import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.linear_model as lm
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid

from CrossValidation import twoLevelCV
from regularization import rlr_validate

# -------------------------------------------------------
# Define input and output matrices that are to be used 
xIn = X_stand
yIn = y_stand

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


# REGRESSION, PART B. 1st point-------------------------------------------------------
# Define the models
model1 = lm.LinearRegression(fit_intercept=False) # fir_intercept = False because centered data
model2 = lm.LinearRegression(fit_intercept=False) # for testing the function

# Put the models together in a list for comparation
modelsToCompare = [model1, model2]
estimatedGenError, best_model_idx = twoLevelCV(xIn, yIn, modelsToCompare, K1=10, K2=10)
# -----------------------------------------------------------------------------------
