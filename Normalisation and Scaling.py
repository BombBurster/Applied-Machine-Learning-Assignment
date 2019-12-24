import pickle
import os
import numpy as np
import pandas as pd

path = "./Absenteeism/"


def load_data(file):
    data = pd.read_csv(path+file)

    return data


def column_to_categorical(data, column):
    minimum = data[column].min()
    maximum = data[column].max()
    minimum = float(minimum)
    maximum = float(maximum)
    # print(minimum)
    # print(maximum)
    for index, row in data.iterrows():
        s = data.xs(index)
        column_i = s[column]
        print('old', s[column])
        if float(column_i) <= (maximum / 3):
            data.at[index, column] = 0  # low
        elif float(column_i) <= (2 * (maximum / 3)):
            data.at[index, column] = 0.5  # medium
        elif float(column_i) > (2 * (maximum / 3)):
            data.at[index, column] = 1  # high
        else:
            Exception('Out of range')
        print('new', data.at[index, column])

    return data


# z-score normalization
def normalize_z_score(data, features):
    for feature in features:
        pd.to_numeric(data[feature])
        s = data[feature]
        new_array = s.to_numpy()
        new_array = new_array.astype(float)
        s = s.to_numpy()
        s = s.astype(float)
        counter = 0
        x_mean = s.mean()
        x_std = s.std()
        for i in s:
            new_val = (i-x_mean)/x_std
            new_array[counter] = new_val
            counter += 1
        data = data.drop(feature, axis=1)
        data[feature] = new_array
    return data


# normalization
def normalize(data, features):
    for feature in features:
        pd.to_numeric(data[feature])
        s = data[feature]
        new_array = s.to_numpy()
        new_array = new_array.astype(float)
        s = s.to_numpy()
        s = s.astype(float)
        counter = 0
        x_mean = s.mean()
        x_min = s.min()
        x_max = s.max()
        for i in s:
            new_val = (i-x_mean)/(x_max-x_min)
            new_array[counter] = new_val
            counter += 1
        data = data.drop(feature, axis=1)
        data[feature] = new_array
    return data


# min_max scaling
def scale(data, features):
    for feature in features:
        pd.to_numeric(data[feature])
        s = data[feature]
        new_array = s.to_numpy()
        new_array = new_array.astype(float)
        s = s.to_numpy()
        s = s.astype(float)
        x_min = s.min()
        x_max = s.max()
        counter = 0
        for i in s:
            new_val = (i-x_min)/(x_max-x_min)
            new_array[counter] = new_val
            counter += 1
        data = data.drop(feature, axis=1)
        data[feature] = new_array
    return data


try:
    dataset = load_data('Absenteeism_at_work.csv')
    print(dataset)
    # dataset = column_to_categorical(dataset, 'Absenteeism time in hours')
    dataset = scale(dataset, dataset.keys())
    print(dataset)
    dataset = load_data('Absenteeism_at_work.csv')
    dataset = normalize(dataset, dataset.keys())
    print(dataset)
    dataset = load_data('Absenteeism_at_work.csv')
    dataset = normalize_z_score(dataset, dataset.keys())
    print(dataset)
except Exception as inst:
    print(type(inst))
    print('Error has occured: ' + inst.args)
