import os
import numpy as np
import pandas as pd
import string
import random
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr


def load_data(file):
    data = pd.read_csv(file)

    return data


# Define a function for Pearson Correlation from first principles
def my_pearsonr(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator=np.sum((x-mean_x)*(y-mean_y))
    denominator=(np.sum((x-mean_x)**2) * np.sum((y-mean_y)**2))**0.5
    return numerator / denominator, None


# Function to identify which features to drop
def get_correlated_features(corr, treshold):
    feature_list = []
    cols = corr.columns.size
    # print('Correlated Features:')
    for i in range(cols):
        for j in range(i+1, cols):
            if corr.iloc[i, j] > treshold:
                feature_list.append((i,j))
                # print(corr.columns[i] + ' <--> ' + corr.columns[j])
    return np.array(feature_list)


def feature_selection(data, correlation_threshold=0.6):
    # Load the dataset and seperate the target column
    original_dataset = data
    target = data['Absenteeism time in hours']
    data = data.drop('Absenteeism time in hours', axis=1)

    # Calculate the feature correlation
    cols = data.columns.size
    relations = np.zeros((cols, cols))

    for i in range(cols):
        for j in range(i, cols):
            a, _ = my_pearsonr(data.iloc[:, i], data.iloc[:, j])
            b, _ = my_pearsonr(data.iloc[:, j], data.iloc[:, i])
            relations[i, j] = a
            relations[j, i] = b
    corr = pd.DataFrame(data=relations, index=data.columns, columns=data.columns)

    to_drop = get_correlated_features(corr, correlation_threshold)
    for x in to_drop[:, 0]:
        # print('Dropping feature: ' + corr.columns[x])
        original_dataset = original_dataset.drop(corr.columns[x], axis=1)

    return original_dataset


# Calcuate the covariance matrix
def my_cov(df, X):
    return np.dot(X, X.T)/df.shape[1]


def principle_component_analysis(data, threshold=0.9):
    # Load the dataset and seperate the target column
    original_dataset = data
    target = data['Absenteeism time in hours']
    data = data.drop('Absenteeism time in hours', axis=1)

    # normalize the data
    normalized_data = normalize(data, data.keys())

    # transpose the normalized data and comput the covariance matrix
    X_t = normalized_data.T
    covariance_matrix = my_cov(data, X_t)

    # Compute the Eigen values and vectors
    U, S, V = np.linalg.svd(covariance_matrix)

    K = 1
    while True:
        preserve = np.sum(S[0:K]) / np.sum(S)
        if (preserve >= threshold):
            break
        K += 1

    # Compute the Principle Components
    pca_x = np.dot(data, V)
    # Drop the unwanted components
    pca_x = pca_x[:, 0:K]

    # Change to pandas
    pca_result = pd.DataFrame(data=pca_x, columns=data.keys()[0:K])
    pca_result['Absenteeism time in hours'] = target
    return pca_result


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
