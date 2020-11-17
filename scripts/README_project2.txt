Readme file - Project 2
Authors: Vice Roncevic - s190075, Carlos Ribera - s192340
Created: 16.11.2020

IMPORTANT: All scripts, text files and data set files are to be kept in the same working directory.
           Each script has its own description, explaining briefly its usage.
           The scripts can be divided into two groups depending on the assignment:

PROJECT 1:
- loadDataSet.py
- basicStatistics.py
- preprocessing.py
- concNoZero_config.py
- concRaw_config.py
- datasetQuality.py
- pca_analysis.py

PROJECT 2:
- featureTransform.py (only functions)
- Cross-Validation.py (only functions)
- regularization.py (only functions)
- ANN_functions.py (only functions)
- regChoice.py 
- choice_lambdas_hidden_units.py
- statEval_regression.py  (important file)
- statEval_classification.py  (important file)

What shpuld I run for project 2? Easy!
 
1. All scripts, text files and data set files are to be kept in the same working directory.

2. Run "statEval_regression.py" to compare Regularized Linear Regression, ANN for Regression and Baseline models using two level Cross-Validation
   It also include the estimation of the optimal lambda and number of hidden units as a complexity controlling parameter.
   It also performs statistical performance evaluation.
   
3. Run "statEval_classification.py" to compare Regularized Multinominal Regression, ANN for MultiClassification and Baseline models using two level Cross-Validation
   It also include the estimation of the optimal lambda and number of hidden units as a complexity controlling parameter.
   It also performs statistical performance evaluation.
   
4. To see the comparation between different feature transformation run: "regChoice.py"

5. To see how the range of lambdas and number of hidden units have been chosen run: "choice_lambdas_hidden_units.py" 