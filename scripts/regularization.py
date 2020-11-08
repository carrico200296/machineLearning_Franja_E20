"""
Regularization script

Created: 01.11.2020
"""
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from regression import x_add_features


def rlr_validate(xIn, yIn, lambdas, cvf):
    
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        xIn       training data set MUST BE STANDARDIZED!
        yIn       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = xIn.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    yIn = yIn.squeeze()
    
    for train_index, test_index in CV.split(xIn,yIn):
        
        X_train = xIn[train_index]
        y_train = yIn[train_index]
        X_test = xIn[test_index]
        y_test = yIn[test_index]
        
        # Precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        for l in range(0,len(lambdas)):
            
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power( y_train-X_train @ w[:,f,l].T, 2).mean(axis=0)
            test_error[f,l] = np.power( y_test-X_test @ w[:,f,l].T, 2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda


# Import configuration file that determines the dataset to be used
from concNoZero_config import *
#from concRaw_config import *

xIn, yIn = x_add_features(X_stand, y_fromStand)

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
# ---------


