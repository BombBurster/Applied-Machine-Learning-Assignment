import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from Functions_file import *
from Random_Forest import *

path = "./Absenteeism/"

dataset = load_data(path+'Absenteeism_at_work_editted_continous_features_target_not_normalised.csv')
# test_set, train_set = split_data(0.8, dataset)
# RF = RandomForestClassifier(n_estimators=1000, random_state=42)
# log_reg = LogisticRegression()
RF_1_accuracy, RF_1_precision, RF_1_recall, RF_1_f1_score = cross_validation_RF_1(RandomForestRegressor(n_estimators=30), 5, dataset, 'Absenteeism time in hours')
print('RF 1 st principles: ', RF_1_accuracy, ', ', RF_1_precision, ', ', RF_1_recall, ', ', RF_1_f1_score)
# RF_accuracy, RF_precision, RF_recall, RF_f1_score = cross_validation(RandomForestClassifier(n_estimators=1000, random_state=4), 5, dataset, 'Absenteeism time in hours')
# print('RF: ', RF_accuracy, ', ', RF_precision, ', ', RF_recall, ', ', RF_f1_score)
# log_accuracy, log_precision, log_recall, log_f1_score = cross_validation(LogisticRegression(solver='lbfgs', multi_class='auto'), 5, dataset, 'Absenteeism time in hours')
# print('Log_reg: ', log_accuracy, ', ', log_precision, ', ', log_recall, ', ', log_f1_score)
