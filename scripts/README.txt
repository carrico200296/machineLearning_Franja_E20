Readme file - Project 1
Authors: Vice Roncevic - s190075, Carlos Ribera - S192340, Mu Zhou - s202718
Created: 04.10.2020

Scipts are utilized in a way that the order of data pre-processing steps is determined
in config scripts. All scripts, text files and data set files are to be kept in the same working directory. 
loadDataSet.py, basicStatistics.py and preProcessing.py are stand-alone scripts and do not require 
editting. Config scripts utilize preProcessing.py script to subject the data to all computed 
transformations and calculations: centralization, standardization, outlier removal, thresholding 
and removal of zero values (Last values outputed by the config file are subjective to the sequence 
in the config file! User awerness is assumed). Config files also contain statistical summary output. 
Config scripts are to be imported into datasetQuality.py and pca_analysis.py scripts to produce relevant 
output. concRaw_config.py and concNoZero_config.py contain pre-processing steps for the data sets elaborated in the report.

  