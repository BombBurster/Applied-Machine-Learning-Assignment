import numpy as np
import matplotlib.pyplot as plt
#Prevent scientific notation
np.set_printoptions(suppress=True)
import pandas as pd


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

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self._run_tree(self.nodes, X.iloc[i, :]))  # changed was df.iloc[i, :]
        return np.array(y_pred, dtype=np.int64)

    def _run_tree(self, node, x):
        feature = node['feature']
        name = node['name']
        cutoff = node['cutoff']
        value = node['value']
        left_node = node['left_node']
        right_node = node['right_node']
        if left_node == {None} and right_node == {None}:
            return value
        if x[name] > cutoff:
            if right_node == {None}:
                return value
            else:
                return self._run_tree(right_node, x)
        else:
            if left_node == {None}:
                return value
            else:
                return self._run_tree(left_node, x)

    def get_nodes(self):
        return self.nodes