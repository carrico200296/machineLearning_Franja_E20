"""
Regression analysis
Created: 30.10.2020
"""
import sklearn.linear_model as lm
from sklearn import model_selection
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show

# Import configuration file that determines the dataset to be used
#from concRaw_config import *
from concNoZero_config import *

# -------------------------------------------------------
# Define input and output 
xIn = X_stand
yIn = y_stand 

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(xIn,yIn,test_size=test_proportion)

model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X_train,y_train)

y_est_train = np.asarray(model.predict(X_train), dtype=float) 
y_est_test = np.asarray(model.predict(X_test),dtype=float)

"""
# Plot original data and the model output
f = figure()

plot(yIn,y_est,'.')
xlabel('yIn'); ylabel('y predicted')
legend(['Training data', 'Regression fit (model)'])

show()
"""







# Add vector of ones before the regularization









# Could be useful: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge