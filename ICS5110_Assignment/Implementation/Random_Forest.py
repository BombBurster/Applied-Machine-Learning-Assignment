import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import r2_score


# to get the output for the predict function
def get_output_result(array, len_internal_array):
    answer = []
    for i in range(0, len_internal_array):
        predicted_values = []
        for row in array:
            predicted_values.append(row[i])

        values, freq = np.unique(predicted_values, return_counts=True)
        index = np.where(freq == np.amax(freq))
        answer.extend(values[index])

    return answer


class DecisionTreeRegressor:
    def __init__(self, min_rows=5):
        self.min_rows = min_rows

    # Calculate the residual sum of squares for vector v
    def _calc_rss(self, v):
        v_sorted = np.sort(v)
        len_v = len(v_sorted)
        totals = []
        for i in range(len_v - 1):
            mean = np.mean(v_sorted[i:i + 1])
            mean_left = np.mean(v_sorted[0:i + 1])
            mean_right = np.mean(v_sorted[i + 1:len_v])
            rss_left = np.sum((v_sorted[0:i + 1] - mean_left) ** 2)
            rss_right = np.sum((v_sorted[i + 1:len_v] - mean_right) ** 2)
            totals.append(rss_left + rss_right)
        return v_sorted, np.array(totals)

    # Get the value for the minimum rss
    def _min_rss(self, X):
        vals, rss = self._calc_rss(X)
        minrss = np.min(rss)
        idx = vals[np.where(rss == minrss)][0]
        count = len(X[X == idx])
        return idx, count, minrss

    # Get the feature that produce the best split
    def _best_feature_to_split(self, X):
        feature = None
        cutoff = None
        rss = None
        # Calculate the mean of  the last column (target)
        value = np.mean(X[:, -1])
        # Calculate the min rss for all features
        # Exclude the target column
        for i in range(X.shape[1] - 1):
            idx, count, minrss = self._min_rss(X[:, i])
            # Only split the feature if there are different values for the feature
            if (X.shape[0] > count):
                if feature == None:
                    feature = i
                    cutoff = idx
                    rss = minrss
                elif (minrss < rss):
                    feature = i
                    cutoff = idx
                    rss = minrss
        return feature, cutoff, value

    def _split(self, frame):
        left_branch = right_branch = name = None
        feature, cutoff, value = self._best_feature_to_split(frame.to_numpy())
        if feature is not None:
            name = frame.columns[feature]
            left_branch = frame.loc[frame.iloc[:, feature] <= cutoff]
            right_branch = frame.loc[frame.iloc[:, feature] > cutoff]
        return feature, name, cutoff, value, left_branch, right_branch

    def fit(self, frame):
        if (frame.shape[0] < self.min_rows):
            return {None}
        feature, name, cutoff, value, left, right = self._split(frame)
        if (feature is None):
            return {None}
        else:
            node = {'feature': feature, 'name': name, 'cutoff': cutoff, 'value': value}
            node['left_node'] = self.fit(left)
            node['right_node'] = self.fit(right)
        self.nodes = node
        return node

    def predict(self, frame):
        y_pred = []
        for i in range(frame.shape[0]):
            # Leave the target (last column) out
            y_pred.append(self._run_tree(self.nodes, frame.iloc[i, :-1]))

        # print(y_pred)
        y_pred = np.round(y_pred, decimals=0, out=None)
        return np.array(y_pred, dtype=np.int64)

    def _run_tree(self, node, frame):
        feature = node['feature']
        name = node['name']
        cutoff = node['cutoff']
        value = node['value']
        left_node = node['left_node']
        right_node = node['right_node']
        if left_node == {None} and right_node == {None}:
            return value
        if frame[name] > cutoff:
            if right_node == {None}:
                return value
            else:
                return self._run_tree(right_node, frame)
        else:
            if left_node == {None}:
                return value
            else:
                return self._run_tree(left_node, frame)

    def get_nodes(self):
        return self.nodes


class RandomForestRegressor():
    def __init__(self, n_estimators=1000, random_state=0, bag_features=4):
        self.n_estimators = n_estimators
        np.random.RandomState = random_state
        self.bag_features = bag_features

    def _bootstrapX(self, X):
        bx = []
        bX = []
        for i in range(X.shape[0]):
            b = X.iloc[np.random.randint(X.shape[0]), :]
            bX.append(b)
        return pd.DataFrame(bX)

    def _bagX(self, X):
        # Do not select target column
        num_cols = X.shape[1]
        columns = random.sample(range(num_cols - 1), self.bag_features)
        # re-append target column
        columns.append(num_cols - 1)
        return X.iloc[:, columns]

    def fit(self, frame):
        self.trees = []
        for i in range(self.n_estimators):
            # print(i)
            bootstrapX = self._bootstrapX(frame)
            bagX = self._bagX(bootstrapX)
            # print(bagX)
            try:
                tree = DecisionTreeRegressor()
                tree.fit(bagX)
                self.trees.append(tree)
            except:
                pass
        print ('Trees generated: ' + str(len(self.trees)))
        return

    def predict(self, frame):
        y_preds = []
        len_int_array = 0
        models = [dt for dt in self.trees]
        nodes = [model.get_nodes() for model in models]
        for model in models:
            y_pred = model.predict(frame)  # was df
            len_int_array = len(y_pred)
            y_preds.append(y_pred)
        # print(y_preds)

        # mean_y_preds = np.mean(y_preds, axis=0) #, dtype=np.int64
        # print(mean_y_preds)
        # mean_y_preds = np.round(mean_y_preds, decimals=0, out=None)
        # mean_y_preds = mean_y_preds.astype(dtype=np.int64)

        # values, freq = np.unique(y_preds, return_index=True, axis=0)
        # print(values)
        # print(freq)
        # # Get the indices of maximum element in numpy array
        # result = np.where(freq == np.amax(freq))
        # y_actual_preds.append(values[result])
        # print(y_actual_preds)

        y_actual_preds = get_output_result(y_preds, len_int_array)
        # print(y_actual_preds)
        return y_actual_preds  # mean_y_preds

    def get_trees(self):
        return self.trees
