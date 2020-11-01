"""
Regression analysis
Created: 30.10.2020
"""

import sklearn.linear_model as lm
from sklearn import model_selection
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
from matplotlib.pylab import subplot, hist
import math


# Import configuration file that determines the dataset to be used
#from concRaw_config import *
from concNoZero_config import *
from pca_analysis import Z_D1_out, Z_D2_out

# -------------------------------------------------------
# Define input and output 
xIn = Z_D2_out
yIn = y_stand 

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(xIn,yIn,test_size=test_proportion)

Km = 6 # no of features taken in by the regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X_train[:, :Km], y_train)

y_est_train = np.asarray(model.predict(X_train[:, :Km]), dtype=float) 
y_est_test = np.asarray(model.predict(X_test[:, :Km]),dtype=float)

residuals = y_test - y_est_test

# Plot original data and the model output
observations = np.arange(1,int(N/2)+1).reshape(-1,1)
f = figure()
plot(observations, y_est_test,'-') # Plot the prediction
plot(observations, y_test,'.') # Plot the nominal values

#plot(X,y_est,'-') # Plot the prediction
xlabel('N - Observations'); ylabel('y - Value')
legend(['"Training data', 'Regression fit (model) Km={0}'.format(Km)])
show()

# Plot the residuals
f = figure()
hist(residuals, 40)
xlabel("Residual value"); ylabel("Occurence")
show()
print(np.mean(residuals)) 


# Define input and output 
xIn = X_stand
yIn = y_stand 

# -------------------------------------------------------
# Additional nonlinear attributes features
def add_features(xIn, yIn):
    
    # First two attributes squared, for example
    Xf1 = np.power(xIn[:, 1],2).reshape(-1,1)
    Xf2 = np.power(xIn[:, 2],2).reshape(-1,1)
    
    # Add the transformed features into the dataset
    xAddFeat = np.asarray(np.bmat('xIn, Xf1, Xf2'))
    yOut = yIn # For traceability
    
return xAddFeat, yOut

def x_tilda(xIn)
# -------------------------------------------------------
# Create vector of basic functions
fi = math.sin(20)




# Add vector of ones before the regularization
# Could be useful: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge