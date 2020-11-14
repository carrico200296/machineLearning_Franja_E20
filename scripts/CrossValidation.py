"""
Cross-Validation script performs Two level cross-validation for model selection and 
performace evaluation (based on algorithm 6, lecture notes)


So far, only twoLevelCV_compare has been developed.

Usage:                  
Input: see specific function
Output: see specific function

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 07.11.2020
"""

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np

from pca_analysis import pca_compute

from regularization import rlr_validate

from ANN_functions import annr_validate

# Define PCA parameters
threshold = 0.95 
pcUsed = 6


# -------------------------------------------------------
# Compute E_gen for a single model

def twoLevelCV_single(xIn, yIn, model, K1, K2):
    
    '''
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix, (string) model,
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
    Output: (numpy array) estimatedGenError
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_test = np.empty(K1)
    error_train = np.empty(K2)
    error_val = np.empty(K2)
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        print("\nOuter Iteration {0}/{1} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        trainSetsX = []
        trainSetsY = [] 
                
        # Inner cross-validation loop. Model Selection
        k2 = 0
        
        for train_index, val_index in CV_inner.split(X_par):
            
            print("\nInner Iteration {0}/{1}".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
    
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)
            
            m = model.fit(X_train, y_train)
             
            # Compute MSEs
            error_train[k2] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
            error_val[k2] = np.square( y_val - m.predict(X_val) ).sum() / y_val.shape[0]
             
            print("Validation error {0}:".format(np.round(error_val[k2], 4) ))
                
            k2 += 1
        
        # Trace back the model according to its CV fold index               
        print("Inner CV fold of the best model for the last loop: {0}".format(error_val.argmin()+1))
        m = model.fit(trainSetsX[error_val.argmin()], trainSetsY[error_val.argmin()])
        
        # ---------------------------------
        #print(model.coef_)
        # Still TBD
        # m = model.fit(X_par, y_par)
        # ---------------------------------
        
        # Compute MSE
        error_test[k1] = np.square( y_test - m.predict(X_test) ).sum()/y_test.shape[0]
        
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    print("Estimated Generalization Error: {0}".format(estimatedGenError))

    return estimatedGenError


# -------------------------------------------------------
# Compute E_gen for a single model that is using PCA acquired features
    

def twoLevelCV_single_PCA(xIn, yIn, model, K1, K2):
    
    '''
    NO NEED TO INPUT PCA TRANSFORMED DATA!
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix,(string) model,
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
    Output: (numpy array) estimatedGenError
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_test = np.empty(K1)
    error_train = np.empty(K2)
    error_val = np.empty(K2)
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        print("\nOuter Iteration {0}/{1} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        trainSetsX = []
        trainSetsY = []
                
        # Inner cross-validation loop. Model Selection
        k2 = 0
        
        for train_index, val_index in CV_inner.split(X_par):
            
            print("\nInner Iteration {0}/{1}".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
            
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)
            
            # Extract projected data set and PCA space vector, fit the model
            V_D_temp = pca_compute(X_train, X_train, threshold, pcUsed)[6]
            X_train_PCA = pca_compute(X_train, X_train, threshold, pcUsed)[2]
            
            m = model.fit(X_train_PCA, y_train)
                 
            # Project validation data into the training PCA space, compute MSE
            X_val_temp = X_val @ V_D_temp
            X_val_temp = X_val_temp[:, :pcUsed]
             
            # Compute MSEs
            error_train[k2] = np.square( y_train - m.predict(X_train_PCA) ).sum() / y_train.shape[0]
            error_val[k2] = np.square( y_val - m.predict(X_val_temp) ).sum() / y_val.shape[0]
             
            print("Validation error {0}:".format(np.round(error_val[k2], 4) ))
                
            k2 += 1
        
        # Trace back the model according to its CV fold index               
        print("Inner CV fold of the best model for the last loop: {0}".format(error_val.argmin()+1))
                   # Extract projected data set and PCA space vector, fit the model
        X_temp = trainSetsX[error_val.argmin()]
        y_temp = trainSetsY[error_val.argmin()]
        
        V_D_temp = pca_compute(X_temp, y_temp, threshold, pcUsed)[6]
        X_temp = pca_compute(X_temp, X_temp, threshold, pcUsed)[2]
        
        m = model.fit(X_temp, y_temp)
        
        # ---------------------------------
        #print(model.coef_)
        # Still TBD
        # m = model.fit(X_par, y_par)
        # ---------------------------------
        
        X_test_PCA = X_test @ V_D_temp
        X_test_PCA = X_test_PCA[:, :pcUsed]
        
        # Compute MSE        
        error_test[k1] = np.square( y_test - m.predict(X_test_PCA) ).sum()/y_test.shape[0]
        
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    print("Estimated Generalization Error: {0}".format(estimatedGenError))

    return estimatedGenError

# -------------------------------------------------------
# Basic 2 model comparison

def twoLevelCV_compare(xIn, yIn, models, K1, K2, lambdas, hidden_units, baseline):
    
    '''
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix, (list) models, 
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
            
            Function assumes that the input will contain lin reg model and ANN with their
            respected lambda and h ranges!
            
            
    Output: (numpy array) estimatedGenError
    '''
    
    CV_outer = model_selection.KFold(n_splits=K1, shuffle=True)
    CV_inner = model_selection.KFold(n_splits=K2, shuffle=True)
    
    # Initialize variables
    error_test = np.empty((K1, len(models)))
    error_baseline = np.empty(K1,)
    
    error_train = np.empty((K2, len(models)))
    error_val = np.empty((K2, len(models)))
    inner_lambdas = np.zeros(K2) # Inner loop values for optimal lambda
    outer_lambdas = np.zeros(K1) # Outer loop values for optimal lambda - lambds of optimal models

    inner_hidden_units = np.zeros(K2) # Inner loop values for optimal number of hidden units
    outer_hidden_units = np.zeros(K1) # Outer loop values for optimal number of hidden units -num hidden units of optimal models

    
    gen_error_models = np.empty((len(models), 1))
    best_models_idx = np.empty((1, len(models)))
    estimatedGenError = np.empty((1, len(models)))
    
    # r parameter for the correlated t test initialization
    r = []
    
    # Outer cross-validation loop. Performance Evaluation
    k1 = 0 
    for par_index, test_index in CV_outer.split(xIn):
        
        print("\nOuter Iteration {0}/{1} -----------".format(k1+1, K1))
        # extract par and test set for current CV fold
        X_par = xIn[par_index, :]
        y_par = yIn[par_index]
        X_test = xIn[test_index, :]
        y_test = yIn[test_index]
        
        # Inner cross-validation loop. Model selection and parameter optimization
        k2 = 0
        trainSetsX = []
        trainSetsY = []     
        
        for train_index, val_index in CV_inner.split(X_par):
            
            print("\nInner Iteration {0}/{1}".format(k2+1, K2))
            
            # Extract train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
    
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)           
            
            for s, model in enumerate(models):
                
                if s == 0: # Linear regression
                    
                    m = model.fit(X_train, y_train)
                    
                    # Add offset attribute - to accomodate for the specificity of the rlr_validate code
                    xInReg = np.concatenate((np.ones((X_train.shape[0],1)), X_train),1)
                    opt_lambda = rlr_validate(xInReg, y_train, lambdas, 10)[1]
                    # Save the values of the optimal regularization strength
                    inner_lambdas[k2] = opt_lambda
 
                    # Compute MSEs
                    error_train[k2, s] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
                    error_val[k2, s] = np.square( y_val - m.predict(X_val) ).sum() / y_val.shape[0]
                    
                if s==1: # Insert code for ANN here....

                    opt_n_hidden_units = annr_validate(X_train, y_train, hidden_units, 10, n_replicates=1, max_iter=10000)[0]
                    inner_hidden_units[k2] = opt_n_hidden_units
                
                    # Compute MSEs
                    error_train[k2, s] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
                    error_val[k2, s] = np.square( y_val - m.predict(X_val) ).sum() / y_val.shape[0]
                    
                else:
                    pass
                    # baseline
                    
                
                print("Validation error - Model {0}: {1}".format(s+1, np.round(error_val[k2, s], 4) ))
                
            k2 += 1
            
        for s, model in enumerate(models): 
            
            # Find the CV index of optimal models
            best_models_idx[0, s] = error_val[:, s].argmin()
            print("Inner CV fold of the best model {0} (last loop): {1}".format(s, best_models_idx[0, s]+1))
            
            # Save the optimal lambda of the optimal model
            if s == 0:
                
                # Trace back the model according to its CV fold index
                m = model.fit(trainSetsX[int(best_models_idx[0, s])], trainSetsY[int(best_models_idx[0, s])])
                
                # ---------------------------------
                #print(model.coef_)
                # Still TBD
                # m = model.fit(X_par, y_par)
                # ---------------------------------
                
                # Compute MSE for the optimal model
                error_test[k1, s] = np.square( y_test - m.predict(X_test) ).sum()/y_test.shape[0]    
                outer_lambdas[k1] = inner_lambdas[int(best_models_idx[0, s])]
                
            else: # Insert code for optimal h (for ANN here)...
                pass
            
            
            # Compute MSE for the baseline
            error_baseline[k1] = np.square( y_test - baseline ).sum()/y_test.shape[0]
                      
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    
    # Make different r vectors for different combinations of models...
    
    # Append the list of the differences in the GENERATLIZATION errors of the two models 
    r.append( np.mean(error_test[:, 0] - error_test[:, 1]) )
    # Can also be
    r.append ( estimatedGenError  )
    
    print("\n")
    for s in range(len(models)):
        print("Estimated Generalization Error for Model {0}: {1}".format(s+1, estimatedGenError[s]))

    return error_test, outer_lambdas, error_baseline, r, estimatedGenError 

# 11.11.2020 Deprecated


# -------------------------------------------------------
# Compare a regular model with the model that is using PCA acquired features

def twoLevelCV_compare_PCA(xIn, yIn, models, K1, K2):
    
    '''
    Input: (numpy array) xIn matrix, (numpy array) yIn matrix, (list) models, 
            (int) K1:folds in outer loop, (int) K2:folds in inner loop, 
            (list) lamda_range: range of values of lambda
    Output: (numpy array) estimatedGenError
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
            
            # Extract DEFAULT train and test set for current CV fold
            X_train = X_par[train_index, :]
            y_train = y_par[train_index]       
            X_val = X_par[val_index, :]
            y_val = y_par[val_index]
            
            trainSetsX.append(X_train) # To trace back optimal models
            trainSetsY.append(y_train)
                
            for s, model in enumerate(models):
                
                # Determine specific process flow for the PCA model
                if s == 1:
                    
                    # Extract projected data set and PCA space vector, fit the model
                    V_D_temp = pca_compute(X_train, X_train, threshold, pcUsed)[6]
                    X_train_PCA = pca_compute(X_train, X_train, threshold, pcUsed)[2]
                    
                    m = model.fit(X_train_PCA, y_train)
                         
                    # Project validation data into the training PCA space, compute MSE
                    X_val_temp = X_val @ V_D_temp
                    X_val_temp = X_val_temp[:, :pcUsed]
                    
                    # Compute MSEs
                    error_train[k2, s] = np.square( y_train - m.predict(X_train_PCA) ).sum() / y_train.shape[0]
                    error_val[k2, s] = np.square( y_val - m.predict(X_val_temp) ).sum() / y_val.shape[0]
                    print("Validation error - Model {0}: {1}".format(s+1, np.round(error_val[k2, s], 4) ))
                    
                else:
                    
                    m = model.fit(X_train, y_train)
                                         
                    # Compute MSE
                    error_train[k2, s] = np.square( y_train - m.predict(X_train) ).sum() / y_train.shape[0]
                    error_val[k2, s] = np.square( y_val - m.predict(X_val) ).sum() / y_val.shape[0]
                    
                    print("Validation error - Model {0}: {1}".format(s+1, np.round(error_val[k2, s], 4) ))
                
            k2 += 1
            
        # STILL ToDo
        # ---------------------------------
        # Train the model on D_par & compute the error
        # ---------------------------------

        for s, model in enumerate(models): 
            
            # Find the CV index of optimal model
            best_models_idx[0, s] = error_val[:, s].argmin()
            print("Inner CV fold of the best model {0} (last loop): {1}".format(s, best_models_idx[0, s]+1))
                                                
            # Determine specific process flow for the PCA model
            if s == 1:
                
                # Extract projected data set and PCA space vector, fit the model
                X_temp = trainSetsX[int(best_models_idx[0, s])]
                y_temp = trainSetsY[int(best_models_idx[0, s])]
                
                V_D_temp = pca_compute(X_temp, y_temp, threshold, pcUsed)[6]
                X_temp = pca_compute(X_temp, X_temp, threshold, pcUsed)[2]
                
                m = model.fit(X_temp, y_temp)
                
                X_test_PCA = X_test @ V_D_temp
                X_test_PCA = X_test_PCA[:, :pcUsed]
                
                error_test[k1, s] = np.square( y_test - m.predict(X_test_PCA) ).sum()/y_test.shape[0]
                                    
            else:
                
                # Trace back the model according to its CV fold index
                m = model.fit(trainSetsX[int(best_models_idx[0, s])], trainSetsY[int(best_models_idx[0, s])])
                
                # Compute MSE
                error_test[k1, s] = np.square( y_test - m.predict(X_test) ).sum()/y_test.shape[0]
        
        k1 += 1
        
    estimatedGenError = np.round(np.mean(error_test, axis = 0), 4)
    
    print("\n")
    for s in range(len(models)):
        print("Estimated Generalization Error for Model {0}: {1}".format(s+1, estimatedGenError[s]))

    return estimatedGenError
