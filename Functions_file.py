import os
import numpy as np
import pandas as pd
import string
import random
import statistics


# Load data from csv file
def load_data(file):

    # use pandas function for loading data from csv
    data = pd.read_csv(file)
    return data


# Define a function for Pearson Correlation from first principles (for Feature Selection)
def my_pearsonr(x,y):
    # calculate the means for X and Y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    # calculate the numerator for pearson correlation
    numerator = np.sum((x-mean_x)*(y-mean_y))
    # calculate the denominator for pearson correlation
    denominator = (np.sum((x-mean_x)**2) * np.sum((y-mean_y)**2))**0.5
    return numerator / denominator, None


# Function to identify which features are correlated over a threshold (for Feature Selection)
def get_correlated_features(corr, treshold):
    feature_list = []
    cols = corr.columns.size
    # looping over the columns
    for i in range(cols):
        # check the correlation between column i and all the other columns and if the correlation is greater than
        # threshold add column names that are correlated
        for j in range(i+1, cols):
            if corr.iloc[i, j] > treshold:
                feature_list.append((i, j))
    return np.array(feature_list)


# Perform feature selection
def feature_selection(data, correlation_threshold=0.6):
    # Seperate the target column from the rest of the data
    original_dataset = data
    target = data['Absenteeism time in hours']
    data = data.drop('Absenteeism time in hours', axis=1)

    # initialize cols and relations to be used below
    cols = data.columns.size
    relations = np.zeros((cols, cols))

    # Calculate the feature correlation
    # populate a list with the columns of the dataset as both the x and y indexes with the correlation between the columns
    for i in range(cols):
        for j in range(i, cols):
            a, _ = my_pearsonr(data.iloc[:, i], data.iloc[:, j])
            b, _ = my_pearsonr(data.iloc[:, j], data.iloc[:, i])
            relations[i, j] = a
            relations[j, i] = b
    corr = pd.DataFrame(data=relations, index=data.columns, columns=data.columns)

    # get a list of the features that are correlated above the threshold
    to_drop = get_correlated_features(corr, correlation_threshold)
    # remove the correlated features
    for x in to_drop[:, 0]:
        original_dataset = original_dataset.drop(corr.columns[x], axis=1)

    return original_dataset


# Calcuate the covariance matrix (for PCA)
def my_cov(df, X):
    return np.dot(X, X.T)/df.shape[1]


# Perform principle component analysis
def principle_component_analysis(data, threshold=0.9):
    # Seperate the target column from the rest of the data
    original_dataset = data
    target = data['Absenteeism time in hours']
    data = data.drop('Absenteeism time in hours', axis=1)

    # normalize the data
    normalized_data = normalize(data, data.keys())

    # transpose the normalized data and compute the covariance matrix
    X_t = normalized_data.T
    covariance_matrix = my_cov(data, X_t)

    # Compute the Eigen values and vectors
    U, S, V = np.linalg.svd(covariance_matrix)

    # Decide on the number of features to preserve according to the threshold
    K = 1
    while True:
        preserve = np.sum(S[0:K]) / np.sum(S)
        if preserve >= threshold:
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


# Output data to csv file
def output_data(data, path, filename):
    # check if the filename is left empty
    if filename != '':
        # check if the extension is written and output data to csv
        if filename.find('.csv') is True:
            data.to_csv(path+filename, index=None, header=True)
        else:
            data.to_csv(path+filename+'.csv', index=None, header=True)
    else:
        print('No filename entered')


# Split the data into test set and training set
def split_data(split, dataset, Y_column_name=0):
    # initialize empty Dataframes for training and testing data
    training_set = pd.DataFrame()
    test_set = pd.DataFrame()
    # loop over the dataset and randomly put values either in the training or test set
    for index, row in dataset.iterrows():
        s = dataset.xs(index)
        if random.random() < split:
            training_set = training_set.append(s, ignore_index=True)
        else:
            test_set = test_set.append(s, ignore_index=True)

    # if a target column name is not inputted output the data split between training and test set
    # else of a target column name is inputted split the target and features for both the training and test set
    # and output seperately the X_train, Y_train, X_test, Y_test
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


# Change a column from continous to categorical
# groups are considered as follows:
# example [0,5,10]: x<0 = 0, 0<x>5 = 1, 5<x>10 = 2, x>10 = 3
def column_to_categorical(data, column, groups):
    # get the length of the groups list
    length = len(groups)

    # loop over the data
    for index, row in data.iterrows():
        # get the value of the column to categorize for the particular row
        s = data.xs(index)
        column_i = s[column]
        # initialize the iterator over the groups
        j = 0
        # loop of the groups list
        for val in groups:
            # if the iterator is 0 check if the value is smaller than the first element of the groups and if smaller
            # change the value to category 0
            if j is 0:
                if float(column_i) <= float(val):
                    data.at[index, column] = j
            # else if the iterator is greater than 0 but less than the length of the group list
            # check if the value is smaller than the current group value, and bigger than the previous group value
            # if this is the case change the value to category j
            elif (j > 0) and (j < (length-1)):
                if (float(column_i) <= float(val)) and (float(column_i) > float(groups[j - 1])):
                    data.at[index, column] = j
            # else if the iterator is at the length of the group list, check if the value is less than the value of the
            # last group and bigger than the previous group value, and if this is the case change the value to category
            # j, otherwise if the value is greater than the value at the last of the group list change the category to
            # value j+1
            elif j is (length-1):
                if (float(column_i) <= float(val)) and (float(column_i) > float(groups[j - 1])):
                    data.at[index, column] = j
                elif float(column_i) > float(val):
                    data.at[index, column] = j + 1
            else:
                print("Categorization Error")
            j += 1

    return data


# perform z-score normalization
def normalize_z_score(data, features):
    # loop over the features
    for feature in features:
        # change the column to numeric
        pd.to_numeric(data[feature])
        s = data[feature]
        # convert the column to numpy array and as type float and create a new array with these contents
        new_array = s.to_numpy()
        new_array = new_array.astype(float)
        s = s.to_numpy()
        s = s.astype(float)
        counter = 0
        # calculate the mean and standard deviation of the column
        x_mean = s.mean()
        x_std = s.std()
        # for every element in the column calculate the normalized value using the z-score equation and put the
        # normalized value in the new array
        for i in s:
            new_val = (i-x_mean)/x_std
            new_array[counter] = new_val
            counter += 1
        # drop the previous column and replace with the new normalized column
        data = data.drop(feature, axis=1)
        data[feature] = new_array
    return data


# perform standard normalization
def normalize(data, features):
    # loop over the features
    for feature in features:
        # change the column to numeric
        pd.to_numeric(data[feature])
        s = data[feature]
        # convert the column to numpy array and as type float and create a new array with these contents
        new_array = s.to_numpy()
        new_array = new_array.astype(float)
        s = s.to_numpy()
        s = s.astype(float)
        counter = 0
        # calculate the mean and minimum and maximum of the column
        x_mean = s.mean()
        x_min = s.min()
        x_max = s.max()
        # for every element in the column calculate the normalized value and put the normalized value in the new array
        for i in s:
            new_val = (i-x_mean)/(x_max-x_min)
            new_array[counter] = new_val
            counter += 1
        # drop the previous column and replace with the new normalized column
        data = data.drop(feature, axis=1)
        data[feature] = new_array
    return data


# perform min_max scaling
def scale(data, features):
    # loop over the features
    for feature in features:
        # change the column to numeric
        pd.to_numeric(data[feature])
        s = data[feature]
        # convert the column to numpy array and as type float and create a new array with these contents
        new_array = s.to_numpy()
        new_array = new_array.astype(float)
        s = s.to_numpy()
        s = s.astype(float)
        # calculate the mean and minimum and maximum of the column
        x_min = s.min()
        x_max = s.max()
        counter = 0
        # for every element in the column calculate the new scaled value and put the scaled value in the new array
        for i in s:
            new_val = (i-x_min)/(x_max-x_min)
            new_array[counter] = new_val
            counter += 1
        # drop the previous column and replace with the new normalized column
        data = data.drop(feature, axis=1)
        data[feature] = new_array
    return data


# Calculate the confusion matrix
# Indexes for vertical are the actual, indexes for horizontal are the predicted
def confusion_matrix(y_actual, y_predict, full_target_column):
    # Get the classes values and initialize an empty confusion matrix
    classes = np.unique(full_target_column)
    conf_matrix = pd.DataFrame(columns=classes, index=classes)
    conf_matrix.fillna(0, inplace=True)
    # loop over the confusion matrix
    for index, row in conf_matrix.iterrows():
        i = 0
        # loop over the actual target class values and if the target class value is the same as the row index of the
        # confusion matrix increment the location at which y_predict resulted
        for y in y_actual:
            if y == index:
                conf_matrix[y_predict[i]][index] = conf_matrix[y_predict[i]][index] + 1
            i += 1
    return conf_matrix


# Calculate the accuracy
def accuracy(y_actual, y_predict, full_target_column):
    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y_actual, y_predict, full_target_column)
    total = 0
    good = 0
    # loop over the confusion matrix and calculate the correctly predicted, as well as the total
    for index, row in conf_matrix.iterrows():
        conf_matrix_row = conf_matrix.xs(index)
        i = 0
        for column in conf_matrix_row:
            total += column
            if index == i:
                good += column
            i += 1
    # return the correctly predicted divided by the total
    return good/total


# Calcualte the precision
def precision(y_actual, y_predict, full_target_column):
    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y_actual, y_predict, full_target_column)
    # initialize an empty numerator array
    numerator = [0.0]*len(conf_matrix)
    # loop over the confusion matrix columns (predicted values)
    for column in range(0, len(conf_matrix)):
        true_pos = 0
        denominator = 0
        # loop over the confusion matrix rows (actual values)
        for row in range(0, len(conf_matrix)):
            # increment the total predictions over the class
            denominator += conf_matrix[column][row]
            # if the row and column are the same increment the true positive
            if row == column:
                true_pos += conf_matrix[column][row]

        # check for division by 0 if division by 0 write 'nan'
        if denominator != 0:
            # calculate precision over a particular class
            numerator[column] = (true_pos/denominator)
        else:
            numerator[column] = 'nan'

    # clean the numerator array for values of 'nan' and output the mean
    cleaned_numer = [numer for numer in numerator if str(numer) != 'nan']
    return sum(cleaned_numer)/len(cleaned_numer)


# Calculate the recall
def recall(y_actual, y_predict, full_target_column):
    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y_actual, y_predict, full_target_column)
    # initialize an empty numerator array
    numerator = [0.0]*len(conf_matrix)
    # loop over the confusion matrix rows (actual values)
    for row in range(0, len(conf_matrix)):
        true_pos = 0
        denominator = 0
        # loop over the confusion matrix columns (predicted values)
        for column in range(0, len(conf_matrix)):
            # increment the total actual frequency over the class
            denominator += conf_matrix[column][row]
            # if the row and column are the same increment the true positive
            if column == row:
                true_pos += conf_matrix[column][row]

        # check for division by 0 if division by 0 write 'nan'
        if denominator != 0:
            # calculate recall over a particular class
            numerator[row] = (true_pos/denominator)
        else:
            numerator[row] = 'nan'

    # clean the numerator array for values of 'nan' and output the mean
    cleaned_numer = [numer for numer in numerator if str(numer) != 'nan']
    return sum(cleaned_numer) / len(cleaned_numer)


# calculate the f1_score
def f1_score(y_actual, y_predict, full_target_column):
    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y_actual, y_predict, full_target_column)
    # initialize an empty precision array as well as recall array
    prec = [0.0]*len(conf_matrix)
    rec = [0.0]*len(conf_matrix)
    # initialize an empty numerator array
    numerator = [0.0]*len(conf_matrix)
    # loop over the confusion matrix columns (predicted values)
    for column in range(0, len(conf_matrix)):
        true_pos = 0
        denominator = 0
        # loop over the confusion matrix rows (actual values)
        for row in range(0, len(conf_matrix)):
            # increment the total predictions over the class
            denominator += conf_matrix[column][row]
            # if the row and column are the same increment the true positive
            if row == column:
                true_pos += conf_matrix[column][row]

        # check for division by 0 if division by 0 write 'nan'
        if denominator != 0:
            # calculate precision over a particular class
            prec[column] = (true_pos/denominator)
        else:
            prec[column] = 'nan'

    # loop over the confusion matrix rows (actual values)
    for row in range(0, len(conf_matrix)):
        true_pos = 0
        denominator = 0
        # loop over the confusion matrix columns (predicted values)
        for column in range(0, len(conf_matrix)):
            # increment the total actual frequency over the class
            denominator += conf_matrix[column][row]
            # if the row and column are the same increment the true positive
            if column == row:
                true_pos += conf_matrix[column][row]

        # check for division by 0 if division by 0 write 'nan'
        if denominator != 0:
            # calculate recall over a particular class
            rec[row] = (true_pos/denominator)
        else:
            rec[row] = 'nan'
    # check for division by 0 or for nan values
    for i in range(0, len(prec)):
        if (prec[i] == 'nan') or (rec[i] == 'nan') or (prec[i] == 0) or (rec[i] == 0):
            numerator[i] = 'nan'
        else:
            # calculate the f1_score
            numerator[i] = (2*((prec[i]*rec[i])/(prec[i]+rec[i])))

    # clean the numerator array for values of 'nan' and output the mean
    cleaned_numer = [numer for numer in numerator if str(numer) != 'nan']
    return sum(cleaned_numer) / len(cleaned_numer)


# Perform cross validation for sklearn functions
def cross_validation(Classifier, n_folds, data, target_name):
    # get the target column
    Y = data[target_name]
    # create an array of classifiers
    models = [Classifier]*n_folds
    # create empty accuracy, precision, recall, and f1_score lists
    accuracies = [0.0]*n_folds
    precisions = [0.0]*n_folds
    recalls = [0.0]*n_folds
    f1_scores = [0.0]*n_folds
    # loop for the number of folds
    for i in range(0, n_folds):
        # split the data into features and target, for training and testing
        Y_test, X_test, Y_train, X_train = split_data(0.8, data, target_name)
        # fit model
        models[i] = models[i].fit(X_train, Y_train)
        # predict values
        Y_pred = models[i].predict(X_test)
        # calculate accuracy, precision, recall and f1_score and insert the result in their respective lists
        accuracies[i] = accuracy(Y_test, Y_pred, Y)
        precisions[i] = precision(Y_test, Y_pred, Y)
        recalls[i] = recall(Y_test, Y_pred, Y)
        f1_scores[i] = f1_score(Y_test, Y_pred, Y)
        print(confusion_matrix(Y_test, Y_pred, Y))

    # clean precision, recall and f1_score lists from 'nan' values
    cleaned_prec = [prec for prec in precisions if str(prec) != 'nan']
    cleaned_rec = [rec for rec in recalls if str(rec) != 'nan']
    cleaned_f1_score = [f1 for f1 in f1_scores if str(f1) != 'nan']

    # return the means for the evaluation results
    return statistics.mean(accuracies), statistics.mean(cleaned_prec), statistics.mean(cleaned_rec), statistics.mean(cleaned_f1_score)


# Perform cross validation for the Random Forest from first principle
def cross_validation_RF_1(Classifier, n_folds, data, target_name):
    # get the target column
    Y = data[target_name]
    # create an array of classifiers
    models = [Classifier]*n_folds
    # create empty accuracy, precision, recall, and f1_score lists
    accuracies = [0.0]*n_folds
    precisions = [0.0]*n_folds
    recalls = [0.0]*n_folds
    f1_scores = [0.0]*n_folds
    # loop for the number of folds
    for i in range(0, n_folds):
        # split the data into training and testing
        test, train = split_data(0.8, data)
        # fit model
        models[i].fit(train)
        Y_test = test[target_name]
        # predict values
        Y_pred = models[i].predict(test)
        # calculate accuracy, precision, recall and f1_score and insert the result in their respective lists
        accuracies[i] = accuracy(Y_test, Y_pred, Y)
        precisions[i] = precision(Y_test, Y_pred, Y)
        recalls[i] = recall(Y_test, Y_pred, Y)
        f1_scores[i] = f1_score(Y_test, Y_pred, Y)
        print(confusion_matrix(Y_test, Y_pred, Y))

    # clean precision, recall and f1_score lists from 'nan' values
    cleaned_prec = [prec for prec in precisions if str(prec) != 'nan']
    cleaned_rec = [rec for rec in recalls if str(rec) != 'nan']
    cleaned_f1_score = [f1 for f1 in f1_scores if str(f1) != 'nan']

    # return the means for the evaluation results
    return statistics.mean(accuracies), statistics.mean(cleaned_prec), statistics.mean(cleaned_rec), statistics.mean(cleaned_f1_score)


# Perform cross validation for the logistic regressor from first principle
def cross_validation_log_1(Classifier, n_folds, data, target_name):
    # get the target column
    Y = data[target_name]
    # create an array of classifiers
    models = [Classifier]*n_folds
    # create empty accuracy, precision, recall, and f1_score lists
    accuracies = [0.0]*n_folds
    precisions = [0.0]*n_folds
    recalls = [0.0]*n_folds
    f1_scores = [0.0]*n_folds
    # loop for the number of folds
    for i in range(0, n_folds):
        # split the data into features and target, for training and testing
        Y_test, X_test, Y_train, X_train = split_data(0.8, data, target_name)

        # for multiclass to perform 1 versus all prepare the Y_train
        y_1vsall = []
        nb_classes = 3
        for c in range(nb_classes):
            y_one = np.where(Y_train == c, 1, 0)
            y_1vsall.append(y_one)

        # fit the model
        models[i].fit(X_train, y_1vsall)
        # predict values
        Y_pred = models[i].predict(X_test)
        # calculate accuracy, precision, recall and f1_score and insert the result in their respective lists
        accuracies[i] = accuracy(Y_test, Y_pred, Y)
        precisions[i] = precision(Y_test, Y_pred, Y)
        recalls[i] = recall(Y_test, Y_pred, Y)
        f1_scores[i] = f1_score(Y_test, Y_pred, Y)
        print(confusion_matrix(Y_test, Y_pred, Y))

    # clean precision, recall and f1_score lists from 'nan' values
    cleaned_prec = [prec for prec in precisions if str(prec) != 'nan']
    cleaned_rec = [rec for rec in recalls if str(rec) != 'nan']
    cleaned_f1_score = [f1 for f1 in f1_scores if str(f1) != 'nan']

    # return the means for the evaluation results
    return statistics.mean(accuracies), statistics.mean(cleaned_prec), statistics.mean(cleaned_rec), statistics.mean(cleaned_f1_score)
