"""
Two-Level Cross-Validation

Usage: Two-level cross-validation for model selection and performace evaluation
Input: list models, int K1(fods in outer loop), int K2(folds in inner loop)
Output: estimated generalization error, table

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 30.10.2020
"""

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
from concNoZero_config import *

def twoLevelCrossValidation(X, y, models, K1, K2):
    
    CV_outer = model_selection.KFold(n_splits=K1,shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2,shuffle=True)
    
    # Initialize variables
    error_train = np.empty((K1,1))
    error_test = np.empty((K1,1))
    error_val = np.empty((K2,len(models)))
    gen_error_models = np.empty((len(models),1))
    
    k1 = 0
    # Outer cross-validation loop. Performance Evaluation
    for par_index, test_index in CV_outer.split(X):
        print("---------OUTER ITERATION {}------------".format(k1+1))
        # extract par and test set for current CV fold
        X_par = X[par_index,:]
        y_par = y[par_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        k2 = 0
        # Inner cross-validation loop. Model Selection
        for train_index, val_index in CV_inner.split(X_par):
            print("-INNER ITERATION {}-".format(k2+1))
            # extract train and test set for current CV fold
            X_train = X[train_index,:]
            y_train = y[train_index]       
            X_val = X[par_index,:]
            y_val = y[par_index]           
            
            for s,model in enumerate(models):
                m = model.fit(X_train, y_train)
                error_train[k2][s] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
                error_val[k2][s] = np.square(y_val-m.predict(X_val)).sum()/y_val.shape[0]
            k2+=1
            
        gen_error_models = (X_val.shape[0]/X_par.shape[0])*error_test.sum(axis=0) #sum in columns
        best_model_idx = gen_error_models.argmin()
        best_model = models[best_model_idx].fit(X_par, y_par)
        
        error_test[k1] = np.square(y_test-best_model.predict(X_test)).sum()/y_test.shape[0]
        k1+=1
        
    estimated_gen_error = (X_test.shape[0]/X.shape[0])*error_test.sum()
    print("Estimated Generalization Error: {}".format(estimated_gen_error))
    
    return estimated_gen_error


# -------------------------------------------------------
# Define input and output matrices that are to be used for plots
xIn = X_stand
yIn = y_stand

# Define the models
model_regression = lm.LinearRegression(fit_intercept=False) # fir_intercept = False because centered data
# model2
# model3

# Put the models together in a list for comparation
modelsToCompare = [model_regression]

gen_error_to_test = twoLevelCrossValidation(X_stand, y_stand, modelsToCompare, K1=10, K2=10)
