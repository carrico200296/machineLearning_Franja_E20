"""
Cross-Validation functions

Description: This script has different functions for cross-validation:
                - K-fold cross-validation for model and regularization parameter (lambda) selection (algorithm 5, lecture notes)
                - Two-level cross-validation for model selection and performace evaluation (algorithm 6, lecture notes)
Input: see each function
Output: see each function

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 30.10.2020
"""

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
#from concNoZero_config import *

# ALGORITHM 6 (LECTURE NOTES)
def twoLevelCV(X, y, models, K1, K2):
    '''
    Input: (numpy array) X matrix, (numpy array) y matrix, (list) models, 
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
            (list) lamda_range: range of values of lambda
    Output: (numpy array) estimatedGenError, (int) best_model_idx:return the index of the best model
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1,shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2,shuffle=True)
    
    # Initialize variables
    
    error_test = np.empty((K1,1))
    error_train = np.empty((K2,len(models)))
    error_val = np.empty((K2,len(models)))
    gen_error_models = np.empty((len(models),1))
    
    k1 = 0
    # Outer cross-validation loop. Performance Evaluation
    for par_index, test_index in CV_outer.split(X):
        print("\nOuter Iteration {}/{} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = X[par_index,:]
        y_par = y[par_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        k2 = 0
        # Inner cross-validation loop. Model Selection
        for train_index, val_index in CV_inner.split(X_par):
            print("\n   Inner Iteration {}/{}".format(k2+1,K2))
            # extract train and test set for current CV fold
            X_train = X[train_index,:]
            y_train = y[train_index]       
            X_val = X[par_index,:]
            y_val = y[par_index]           
            
            for s,model in enumerate(models):
                m = model.fit(X_train, y_train)
                error_train[k2][s] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
                error_val[k2][s] = np.square(y_val-m.predict(X_val)).sum()/y_val.shape[0]
                print("   Val_error-Model{0}: {1}".format(s+1,error_val[k2][s]))
            k2+=1
            
        gen_error_models = (X_val.shape[0]/X_par.shape[0])*error_test.sum(axis=0) #sum in columns
        best_model_idx = gen_error_models.argmin()
        best_model = models[best_model_idx].fit(X_par, y_par)
        
        error_test[k1] = np.square(y_test-best_model.predict(X_test)).sum()/y_test.shape[0]
        k1+=1
        
    estimatedGenError = (X_test.shape[0]/X.shape[0])*error_test.sum()    
    print("\nBest model: no {}".format(best_model_idx))
    print("Estimated Generalization Error: {}".format(estimatedGenError))

    return estimatedGenError, best_model_idx



# ALGORITHM 5: K-fold cross-validation for model selection (LECTURE NOTES)
def Kfold_crossValidation(X, y, models, K):
    '''
    Input:  (numpy array) X, (numpy array) y, (list) models, 
            (int) K:folds in the cross-validation loop, 
            (list) lamda_range: range of values of lambda
    Output: (numpy array) estimatedGenError, (int) best_lambda_idx:return the index of the best lambda
    '''
    CV = model_selection.KFold(n_splits=K,shuffle=True)

    # Initialize variables
    estimatedGenError_models = np.empty((K,1))
    error_train = np.empty((K,len(lambda_range)))
    error_test = np.empty((K,len(lambda_range)))
    
    k = 0
    for train_index, test_index in CV.split(X):
        print("\nIteration {}/{} -----------".format(k+1, K))
        # extract train and test set for current CV fold
        X_test = X[test_index,:]
        y_test = y[test_index]
        X_train = X[train_index,:]
        y_train = y[train_index]   
        
        for s,model in enumerate(models):
            m = model.fit(X_train, y_train)
            error_train[k][s] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
            error_test[k][s] = np.square(y_val-m.predict(X_val)).sum()/y_val.shape[0]
            print("   Test_error-lambda{}: {}".format(lambda_value,error_test[k][s]))
        k+=1
        
    estimatedGenError_models = (X_test.shape[0]/X.shape[0])*error_test.sum(axis=0) #sum in columns
    best_lambda_idx = estimatedGenError_models.argmin()
    
    print("\nEstimated Generalization Error: {}".format(estimatedGenError_models))
    
    return estimatedGenError_models