"""
Statistical evaluation

Usage: This script is using the two layer CV output to perform statistical performance evaluation 
of various methods. It also introduces baseline models for comparrison. The comparison is to be made pairwise.              
Input: 
Output: 

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 10.11.2020
"""

from CrossValidation import twoLevelCV_compare, twoLevelCV_compare_PCA, twoLevelCV_single, twoLevelCV_single_PCA
from regression import x_add_features, x_tilda_poly, x_tilda_transform, x_tilda_downSample
import sklearn.linear_model as lm

import sklearn.tree


from toolbox_02450 import *

# Setup 1

# See box 11.3.4 and modify Jeffry interval to accomodate two regression models

# Setup 2

# Check is the correlated t-test enough,  dataset is random

# Import configuration file that determines the dataset to be used
from concNoZero_config import *
#from concRaw_config import *

# Create a baseline model

# Very similar values for both...
# y_baseline = np.mean(y)
y_baseline = np.mean(y_fromStand)

# Create the dataset for the optimal regression model
xIn, yIn = x_add_features(X_stand, y_fromStand)

# Initialize 2 layer CV parameters
K1 = 10
K2 = 10

# Include sklearn.linear_model.Ridge  ALPHA parameters
# Put baseline on the training data

# Values of lambda
lambdas = np.power(10.,np.arange(-4,9,0.5))
# Range of hidden units
hidden_units = np.arange(1,12,5)

# Next block of code is to be used for comparing linear regression and ANN
models = [lm.LinearRegression(), ]
error_test, outer_lambdas, error_baseline, r, estimatedGenError = twoLevelCV_compare(xIn, yIn, models, K1, K2, lambdas, hidden_units, y_baseline)


# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K1
p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
print(" P value for setup II: {0}".format(round(p_setupII, 4)))
print(" CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))


