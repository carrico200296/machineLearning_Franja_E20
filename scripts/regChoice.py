"""
Regression model choice script 

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340
Created: 07.11.2020
"""

# Import configuration file that determines the dataset to be used
from concNoZero_config import *
#from concRaw_config import *

import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.linear_model as lm
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid

from CrossValidation import twoLevelCV_compare, twoLevelCV_compare_PCA, twoLevelCV_single, twoLevelCV_single_PCA

from regression import x_add_features, x_tilda_poly, x_tilda_transform, x_tilda_downSample

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
    
print("Model with the smallest E_gen of {0} is the {1} model.".format(round(modelErrorsAvg.min(), 2), modelsToCompare[modelErrorsAvg.argmin()]))



