import os
import numpy as np
import pandas as pd
from Functions_file import *

path = "./Absenteeism/"


try:
    dataset = load_data(path+'Absenteeism_at_work.csv')
    print(dataset)
    dataset = column_to_categorical(dataset, 'Absenteeism time in hours', [0, 5, 10])
    dataset = feature_selection(dataset, 0.6)
    dataset = scale(dataset, dataset.keys())
    print(dataset)
    # dataset = load_data(path+'Absenteeism_at_work.csv')
    # dataset = column_to_categorical(dataset, 'Absenteeism time in hours', [0, 5, 10])
    # dataset = normalize(dataset, dataset.keys())
    # print(dataset)
    # dataset = load_data(path+'Absenteeism_at_work.csv')
    # dataset = column_to_categorical(dataset, 'Absenteeism time in hours', [0, 5, 10])
    # dataset = normalize_z_score(dataset, dataset.keys())
    # print(dataset)
    # y_test, x_test, y_train, x_train = split_data(0.8, dataset, 'Absenteeism time in hours')
    # print(y_test, '\n\n', x_test)
    # output_data(dataset, path, 'Absenteeism_at_work_editted')
except Exception as inst:
    print(type(inst))
    print('Error has occured: ' + inst.args)