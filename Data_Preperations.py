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
    dataset = column_to_categorical(dataset, 'Transportation expense', [200, 300])
    dataset = column_to_categorical(dataset, 'Distance from Residence to Work', [20, 40])
    dataset = column_to_categorical(dataset, 'Age', [35, 45])
    dataset = column_to_categorical(dataset, 'Body mass index', [18.5, 25, 30])
    dataset = column_to_categorical(dataset, 'Age', [35, 45])
    dataset = feature_selection(dataset, 0.6)
    print(dataset)
    dataset = principle_component_analysis(dataset, 0.9)
    print(dataset)
    dataset = normalize(dataset, dataset.keys())
    dataset = scale(dataset, dataset.keys())

    # output to a csv file
    output_data(dataset, path, 'Absenteeism_at_work_editted')
    print(dataset)
except Exception as inst:
    print(type(inst))
    print('Error has occured: ' + inst.args)