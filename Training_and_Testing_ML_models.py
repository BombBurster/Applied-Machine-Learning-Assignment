import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from Functions_file import *
from Random_Forest import *
from Logistic_Regressor import *

path = "./Absenteeism/"

# load dataset from csv file
dataset = load_data(path+'Absenteeism_at_work_editted_continous_features_target_not_normalised.csv')
# perform Sklearn logistic regression
log_accuracy, log_precision, log_recall, log_f1_score = cross_validation(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000000), 5, dataset, 'Absenteeism time in hours')
print('Logistic Regressor: accuracy=', log_accuracy, ', precision=', log_precision, ', recall=', log_recall, ', f1_score=', log_f1_score)
# perform logistic regression implemented from 1st principle
log_1_accuracy, log_1_precision, log_1_recall, log_1_f1_score = cross_validation_log_1(LogisticRegressorOneVsAll(max_iter=1000), 5, dataset, 'Absenteeism time in hours')
print('Logistic Regressor 1 st principles: accuracy=', log_1_accuracy, ', precision=', log_1_precision, ', recall=', log_1_recall, ', f1_score=', log_1_f1_score)
# perform Sklearn Random Forest Classifier
RF_accuracy, RF_precision, RF_recall, RF_f1_score = cross_validation(RandomForestClassifier(n_estimators=1000, random_state=4), 5, dataset, 'Absenteeism time in hours')
print('Random Forest: accuracy=', RF_accuracy, ', precision=', RF_precision, ', recall=', RF_recall, ', f1_score=', RF_f1_score)
# perform Random Forest implemented from 1st principle
RF_1_accuracy, RF_1_precision, RF_1_recall, RF_1_f1_score = cross_validation_RF_1(RandomForestRegressor(n_estimators=10), 5, dataset, 'Absenteeism time in hours')
print('Random Forest 1 st principles: accuracy=', RF_1_accuracy, ', precision=', RF_1_precision, ', recall=', RF_1_recall, ', f1_score=', RF_1_f1_score)
