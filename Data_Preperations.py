import os
import numpy as np
import pandas as pd
from Functions_file import *

path = "./Absenteeism/"


try:
    # load dataset
    dataset = load_data(path+'Absenteeism_at_work.csv')
    # make the output column categorical
    dataset = column_to_categorical(dataset, 'Absenteeism time in hours', [0, 10])
    # perform feature selection
    dataset = feature_selection(dataset, 0.6)
    # perform PCA
    dataset = principle_component_analysis(dataset, 0.9)
    # get all column names and remove the target column name
    columns = dataset.keys()
    columns = columns.delete(len(columns)-1)
    # normalize all features except for the column
    dataset = normalize_z_score(dataset, columns)
    # rescale all featuers except for the column
    dataset = scale(dataset, columns)
    # add the target back into the now normalized, scaled, and reduced dimensionality dataset
    cols = list(dataset.columns.values)
    cols.pop(cols.index('Absenteeism time in hours'))
    dataset = dataset[cols + ['Absenteeism time in hours']]
    # output the dataset to a csv file
    output_data(dataset, path, 'Absenteeism_at_work_editted_continous_features_target_not_normalised')

except Exception as inst:
    print(type(inst))
    print('Error has occured: ' + inst.args)