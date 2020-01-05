import os
import numpy as np
import pandas as pd
from Functions_file import *

path = "./Absenteeism/"


try:
    # feature selection, pca, normalization and scaling
    dataset = load_data(path+'Absenteeism_at_work.csv')
    print(dataset)
    dataset = column_to_categorical(dataset, 'Absenteeism time in hours', [0, 10])
    # dataset = column_to_categorical(dataset, 'Transportation expense', [200, 300])
    # dataset = column_to_categorical(dataset, 'Distance from Residence to Work', [20, 40])
    # dataset = column_to_categorical(dataset, 'Age', [35, 45])
    # dataset = column_to_categorical(dataset, 'Body mass index', [18.5, 25, 30])
    dataset = feature_selection(dataset, 0.6)
    print(dataset)
    dataset = principle_component_analysis(dataset, 0.9)
    print(dataset)
    columns = dataset.keys()
    columns = columns.delete(len(columns)-1)
    dataset = normalize(dataset, columns)
    dataset = scale(dataset, columns)
    cols = list(dataset.columns.values)
    cols.pop(cols.index('Absenteeism time in hours'))
    dataset = dataset[cols + ['Absenteeism time in hours']]

    # output to a csv file
    output_data(dataset, path, 'Absenteeism_at_work_editted_continous_features_target_not_normalised')
    print(dataset)
except Exception as inst:
    print(type(inst))
    print('Error has occured: ' + inst.args)