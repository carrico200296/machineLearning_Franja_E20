"""
Regression analysis
Created: 30.10.2020
"""
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show

# Import configuration file that determines the dataset to be used
#from concRaw_config import *
from concNoZero_config import *

# -------------------------------------------------------
# Define input and output 
xIn = X_stand
yIn = y_stand 

model = lm.LinearRegression(fit_intercept=True)
model = model.fit(xIn,yIn)

y_est = model.predict(xIn)

# Plot original data and the model output
f = figure()

plot(yIn,y_est,'.')
xlabel('yIn'); ylabel('y predicted')
legend(['Training data', 'Regression fit (model)'])

show()









# Could be useful: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge