"""
Process concRaw data set config file.

Usage: Can be used only if the number of attributes remain constant or changed inside the flow.
Must start from RAW data set!
Input: loadDataSet.py and preProcessing.py 
Output: Processed data and statistical tables. Final output depends on the process flow.

Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 03.10.2020
"""

from preProcessing import *
from loadDataSet import *

# Import Raw data
xRaw = X
yRaw = y

X_cent, y_cent, X_stand, y_stand, classNames, y_class, y_class_encoding, X_noZeros, y_noZeros, X_noOut, y_noOut, M, N, C = pre_process(xRaw, yRaw, attributeNames, outputAttribute)
basic_statistics_x, basic_statistics_y = sum_stat(xRaw, yRaw, attributeNames, outputAttribute)

# Remove outliers
xIn = X_noOut
yIn = y_noOut

X_cent, y_cent, X_stand, y_stand, classNames, y_class, y_class_encoding, X_noZeros, y_noZeros, X_noOut, y_noOut, M, N, C = pre_process(xIn, yIn, attributeNames, outputAttribute)
basic_statistics_x, basic_statistics_y = sum_stat(xIn, yIn, attributeNames, outputAttribute)


