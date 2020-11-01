"""
Cross-Validation functions

Description: This scripts performs:
                        -Two level cross-validation for model selection and 
                        performace evaluation (algorithm 6, lecture notes)
                        
Input: see specific function
Output: see specific function

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 30.10.202
"""

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np

# ALGORITHM 6 (LECTURE NOTES)
def twoLevelCV(xIn, yIn, models, K1, K2):
    
    '''
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix, (list) models, 
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
            (list) lamda_range: range of values of lambda
    Output: (numpy array) estimatedGenError, (int) best_model_idx:return the index of the best model
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_test = np.empty((K1, len(models)))
    error_train = np.empty((K2, len(models)))
    error_val = np.empty((K2, len(models)))
    gen_error_models = np.empty((len(models), 1))
    best_models_idx = np.empty((1, len(models)))
    estimatedGenError = np.empty((1, len(models)))
    
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        print("\nOuter Iteration {0}/{1} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        
        # Inner cross-validation loop. Model Selection
        k2 = 0
        trainSetsX = []
        trainSetsY = []     
        
        for train_index, val_index in CV_inner.split(X_par):
            
            print("\nInner Iteration {0}/{1}".format(k2+1, K2))
            # extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
    
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)           
            
            for s, model in enumerate(models):
                
                m = model.fit(X_train, y_train)
                 
                # Compute average MSE for every model
                error_train[k2, s] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
                error_val[k2, s] = np.square( y_val - m.predict(X_val)).sum() / y_val.shape[0]
                
                print("Validation error - Model {0}: {1}".format(s+1, np.round(error_val[k2, s], 4) ))
                
            k2 += 1
            
        for s, model in enumerate(models): 
            
            # Find the CV index of optimal model
            best_models_idx[0, s] = error_val[:, s].argmin()
            # Trace back the model according to its CV fold index
            m = model.fit(trainSetsX[int(best_models_idx[0, s])], trainSetsY[int(best_models_idx[0, s])])
            
            # Train the models on D_par and calculate the test errors
            m = model.fit(X_par, y_par)
            error_test[k1, s] = np.square( y_test - m.predict(X_test)).sum()/y_test.shape[0]
        
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    for s in range(len(models)):
        print("Estimated Generalization Error for Model {0}: {1}".format(s+1, estimatedGenError[s]))

    return estimatedGenError
