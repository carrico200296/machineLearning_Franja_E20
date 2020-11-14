"""
Statistical evaluation

Usage: This script is using the two layer CV output to perform statistical performance evaluation 
of various methods. It also introduces baseline models for comparrison. The comparison is to be made pairwise.              
Input: 
Output: 

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 10.11.2020
"""

from CrossValidation import twoLevelCV_regression, twoLevelCV_classification
from regression import x_add_features, x_tilda_poly, x_tilda_transform, x_tilda_downSample
import sklearn.linear_model as lm
from concNoZero_config import *
import scipy.stats as st

def correlated_ttest(r, rho, alpha=0.05):
    """
    made by .....
    """
    rhat = np.mean(r)
    shat = np.std(r)
    J = len(r)
    sigmatilde = shat * np.sqrt(1 / J + rho / (1 - rho))

    CI = st.t.interval(1 - alpha, df=J - 1, loc=rhat, scale=sigmatilde)  # Confidence interval
    p = 2*st.t.cdf(-np.abs(rhat) / sigmatilde, df=J - 1)  # p-value
    return p, CI

# Setup 1

# See box 11.3.4 and modify Jeffry interval to accomodate two regression models

# Setup 2

# Check is the correlated t-test enough,  dataset is random

#%% REGRESSION

#_______CREATE DATASET WITH ADDED FEATURES_______
xIn, yIn = x_add_features(X_stand, y_fromStand)

# Initialize 2 layer CV parameters
K1 = 10
K2 = 10

# Values of lambda
lambdas = np.power(10.,np.arange(-4,9,0.5))
# Range of hidden units
hidden_units = np.array((1,5,10,15))

# Comparing with two layer Cross-Validation: linear regression ,ANN and baseline
models = ['REGULARIZED_LINEAR_REGRESSION', 'ANN_REGRESSION', 'BASELINE_REGRESSION']
error_test, outer_lambdas, outer_hidden_units, r, estimatedGenError = twoLevelCV_regression(xIn, yIn, models, K1, K2, lambdas, hidden_units)

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K1

print('Statistical Evaluation for Regression')
print('ANN vs. RLR')
p_setupII, CI_setupII = correlated_ttest(r[:,0], rho, alpha=alpha)
print("\nP value for setup II: {0}".format(round(p_setupII, 4)))
print("CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))

print('ANN vs. Baseline')
p_setupII, CI_setupII = correlated_ttest(r[:,1], rho, alpha=alpha)
print("\nP value for setup II: {0}".format(round(p_setupII, 4)))
print("CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))

print('RLR vs. Baseline')
p_setupII, CI_setupII = correlated_ttest(r[:,2], rho, alpha=alpha)
print("\nP value for setup II: {0}".format(round(p_setupII, 4)))
print("CI setup II: from {0} to {1}:".format(round(CI_setupII[0], 4), round(CI_setupII[1], 4) ))

#%% CLASSIFICATION
"""
#_______CREATE DATASET WITH ADDED FEATURES_______
xIn, yIn = x_add_features(X_stand, y_fromStand)

# Initialize 2 layer CV parameters
K1 = 3
K2 = 3

# Values of lambda
lambdas = np.power(10.,np.arange(-4,9,0.5))
# Range of hidden units
hidden_units = np.arange(5,11,5)

# Comparing with two layer Cross-Validation: linear regression ,ANN and baseline
models = ['REGULARIZED_MULTINOMINAL_REGRESSION', 'ANN_MULTICLASS', 'BASELINE_CLASSIFICATION']
error_test, outer_lambdas, outer_hidden_units, r, estimatedGenError = twoLevelCV_classification(xIn, yIn, models, K1, K2, lambdas, hidden_units)
"""




