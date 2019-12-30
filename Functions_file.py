import os
import numpy as np
import pandas as pd
import string
import random


def load_data(file):
    data = pd.read_csv(file)

    return data


def output_data(data, path, filename):
    if filename != '':
        if filename.find('.csv') is True:
            data.to_csv(path+filename, index=None, header=True)
        else:
            data.to_csv(path+filename+'.csv', index=None, header=True)
    else:
        print('No filename entered')


# function to split the data into test set and training set
def split_data(split, dataset, Y_column_name=0):
    training_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for index, row in dataset.iterrows():
        s = dataset.xs(index)
        if random.random() < split:
            training_set = training_set.append(s, ignore_index=True)
        else:
            test_set = test_set.append(s, ignore_index=True)

    if Y_column_name is 0:
        return test_set, training_set
    else:
        Y_test_set = test_set[Y_column_name]
        Y_test_set = Y_test_set.rename(Y_column_name, axis='columns')
        X_test_set = test_set.drop(Y_column_name, axis=1)
        Y_training_set = training_set[Y_column_name]
        Y_training_set = Y_training_set.rename(Y_column_name, axis='columns')
        X_training_set = training_set.drop(Y_column_name, axis=1)
        return Y_test_set, X_test_set, Y_training_set, X_training_set


# groups are considered as follows:
# example [0,5,10]: x<0 = 0, 0<x>5 = 1, 5<x>10 = 2, x>10 = 3
def column_to_categorical(data, column, groups):
    length = len(groups)

    for index, row in data.iterrows():
        s = data.xs(index)
        column_i = s[column]
        # print('old', s[column])
        j = 0
        for val in groups:
            if j is 0:
                if float(column_i) <= float(val):
                    data.at[index, column] = j
            elif (j > 0) and (j < (length-1)):
                if (float(column_i) <= float(val)) and (float(column_i) > float(groups[j - 1])):
                    data.at[index, column] = j
            elif j is (length-1):
                if (float(column_i) <= float(val)) and (float(column_i) > float(groups[j - 1])):
                    data.at[index, column] = j
                elif float(column_i) > float(val):
                    data.at[index, column] = j + 1
            else:
                print("Categorization Error")
            j += 1
        # print('new', data.at[index, column])

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
