"""
Regression analysis of the two datasets

Created: 19.10.2020
"""

from sklearn import model_selection, tree

# Import configuration file that determines the dataset to be used
#from concRaw_config import *
from concNoZero_config import *

# -------------------------------------------------------
# Define input and output 
xIn = X_stand
yIn = y_class   

def tree_class(xIn, yIn):
    
    # Fit regression tree classifier, Gini split criterion, no pruning
    criterion='gini'
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
    dtc = dtc.fit(xIn, y_class)
    
    fname='tree_' + criterion
    # Export tree graph .gvz file to parse to graphviz
    out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)
    
    # Use the manual procedure to produce the tree plot - see Exercise_content_keywords file
    
    return dtc

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(xIn,yIn,test_size=test_proportion)

dtc = tree_class(xIn, yIn)

y_est_test = np.asarray(dtc.predict(X_test),dtype=int)
y_est_train = np.asarray(dtc.predict(X_train), dtype=int) 

y_est_test = np.reshape(y_est_test, (y_est_test.shape[0], 1))
y_est_train = np.reshape(y_est_train, (y_est_train.shape[0], 1))

misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
#Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train