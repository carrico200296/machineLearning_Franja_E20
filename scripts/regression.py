"""
Regression analysis
Created: 30.10.2020
"""

import sklearn.linear_model as lm
from sklearn import model_selection
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
from matplotlib.pylab import subplot, hist
import math
import numpy as np


# Import configuration file that determines the dataset to be used
#from concRaw_config import *
#from concNoZero_config import *

# -------------------------------------------------------
# Feature transformations to accomodate various linear regression methods

def x_add_features(xIn, yIn):
    
    # -------------------------------------------------------
    # Additional nonlinear attributes features
    Xf1 = np.power(xIn[:, 0], 2).reshape(-1,1)
    #Xf1 = np.sqrt(xIn[:, 0]).reshape(-1,1)
    Xf2 = np.power(xIn[:, 4], 2).reshape(-1,1)
    Xf3 = np.power(xIn[:, 7], 2).reshape(-1,1)
    
    # Add the transformed features into the dataset
    xAddFeat = np.asarray(np.bmat('xIn, Xf1, Xf2, Xf3'))
    yOut = yIn # For traceability
    
    return xAddFeat, yOut 

def x_tilda_poly(xIn, yIn):
    
    # -------------------------------------------------------
    # Do a feature transformation - polynomial regression
    xTildaPoly = np.zeros((xIn.shape[0], xIn.shape[1]))
    
    for i in range(xIn.shape[0]):   
        for j in range(xIn.shape[1]):
            xTildaPoly[i, j] = xIn[i, j]**(j+1)
        
    yOut = yIn # For traceability

    return xTildaPoly, yOut

# --------------------------------
# Vice, 01.11.2020
# Do we want this in radians?

def x_tilda_transform(xIn, yIn):
    
    # -------------------------------------------------------
    # Do a feature transformation - trigonometry based
    xTildaTrans = np.zeros((xIn.shape[0], xIn.shape[1]))
    
    for i in range(xIn.shape[0]):   
        for j in range(xIn.shape[1]):
            if j == 0:    
                xTildaTrans[i, j] = xIn[i, j]
            elif (j % 2) == 1:
                xTildaTrans[i, j] = math.sin(xIn[i, j]*j)
            else:
                xTildaTrans[i, j] = math.cos(xIn[i, j]*j)
        
    yOut = yIn # For traceability

    return xTildaTrans, yOut


def x_tilda_downSample(xIn, yIn, features):
    
    # -------------------------------------------------------
    # Consider only some features in the dataset
    xTildaDown = np.zeros((xIn.shape[0], 1))
                  
    for i in range(np.size(features)):
        x_temp = np.reshape(xIn[:, features[i]], (xIn[:, features[i]].shape[0], 1))
        xTildaDown = np.append(xTildaDown, x_temp, axis = 1 )
   
    xTildaDown = np.delete(xTildaDown, 0, 1)
    yOut = yIn # For traceability

    return xTildaDown, yOut

"""
Could be a useful block of code....


# -------------------------------------------------------
# Define input and output 
xIn = Z_D2_out 
yIn = y_noOut

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(xIn, yIn, test_size = test_proportion)

# Choose a number of features taken in by the regression model
# If the number is larger than then number attributes = collumns, numpy will take all the collumns!
print("Number of available attributes: {0}".format(xIn.shape[1]))

Km = 8
print("Number of attributes used in the lin.reg.model: {0}".format(Km))
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X_train[:, :Km], y_train)

y_est_train = np.asarray(model.predict(X_train[:, :Km]), dtype=float) 
y_est_test = np.asarray(model.predict(X_test[:, :Km]),dtype=float)

residuals = y_test - y_est_test

# Plot original data and the model output
observations = np.arange(1,int(N/2)+1).reshape(-1,1)
f = figure()
plot(observations, y_est_test,'-') # Plot the predictions
plot(observations, y_test,'.') # Plot the nominal values

xlabel('N - Observations'); ylabel('y - Value')
legend(['Regression fit (model) Km={0}'.format(Km), 'Test data',])
show()

# Plot the residuals
f = figure()
hist(residuals, 40)
xlabel("Residual value"); ylabel("Occurence")
show()

print("Mean residual value: {0}".format(np.round(np.mean(residuals), 4))) 
"""
