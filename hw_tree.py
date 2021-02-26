import math
import random
import pandas as pd
import numpy as np
import operator


class Node:
    def __init__(self, X, Y, min_samples):
        self.X = X
        self.Y = Y
        self.end = True
        if len(Y) >= min_samples:
            self.end = False
            min_e = find_best_split(self.X, self.Y)
            self.split_i = min_e['Index']
            self.split_val = min_e['Value']
            self.l_child = Node(X=X[np.where(X[:, self.split_i] < self.split_val)],
                                Y=Y[np.where(X[:, self.split_i] < self.split_val)],
                                min_samples=min_samples)
            self.r_child = Node(X=X[np.where(X[:, self.split_i] > self.split_val)],
                                Y=Y[np.where(X[:, self.split_i] > self.split_val)],
                                min_samples=min_samples)
        else:
            counts = {}
            for v in np.unique(Y):
                counts[v] = len(np.where(Y == v))
            self.majority = max(counts.items(), key=operator.itemgetter(1))[0]

    def predict(self, X):
        if self.end:
            return self.majority
        else:
            if X[self.split_i] > self.split_val:
                return self.l_child.predict(X)
            else:
                return self.r_child.predict(X)


class Tree:
    def __init__(self, rand, min_samples, get_candidate_columns):
        self.min_samples = min_samples
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns

    def build(self, X, Y):
        return Node(X, Y, self.min_samples)


class Bagging:
    def __init__(self, n):
        self.rand = random.Random()
        self.tree_builder = Tree()
        self.n = n

    def build(self):
        """
        returns the model as an object, whose predict method returns the predicted target class of given input samples
        :return:
        """


class RandomForest:
    def __init__(self, n):
        self.rand = random.Random()
        self.n = n
        self.min_samples = 0

    def build(self):
        """
        returns the model as an object, whose predict method returns the predicted target class of given input samples
        :return:
        """


"""
HOMEWORK FUNCTIONS
"""


def hw_tree_full():
    """
    Build a tree with min_samples=2, then return misclassification rates on training and testing data.
    :return:
    """


def hw_cv_min_samples():
    """
    Find the best value of min_samples with 5-fold CV on training data. Return misclassification rates on
    training and testing data for a tree with the best value of min_samples. Also, return the best min_samples
    :return:
    """


def hw_bagging():
    """
    Use bagging with n=50 trees with min_samples=2. Return misclassification rates on training and testing data
    :return:
    """


def hw_randomforests():
    """
    Use random forests with n=50 trees with min_samples=2. Return misclassification rates on training and testing data.
    :return:
    """


def gini(Y):
    """
    1/Nm * sum(Index(yi = k))
    """

    Nm = len(Y)
    to_sum = []
    for c in np.unique(Y):
        count = len(np.where(Y == c)[0])
        pi_c = count/Nm
        to_sum.append(pi_c * (1-pi_c))
    return sum(to_sum)


def find_best_split(X, Y, head=None):
    min_error = {'Index': -1, 'Value': -1, 'Error': math.inf}
    if head != None:
        min_error.update({'Attribute': None})
    for i in range(len(X[0])):
        col_vals = sorted(np.unique(X[:, i]))
        val_candidates = [(a + b) / 2 for (a, b) in list(zip(col_vals, col_vals[1:]))]
        for val in val_candidates:
            X_less, Y_less = X[np.where(X[:, i] < val)], Y[np.where(X[:, i] < val)]
            X_more, Y_more = X[np.where(X[:, i] > val)], Y[np.where(X[:, i] > val)]
            w_less = len(X_less) / len(X)
            w_more = len(X_more) / len(X)
            gini_less = gini(Y_less)
            gini_more = gini(Y_more)
            error = w_less * gini_less + w_more * gini_more
            if error < min_error['Error']:
                min_error['Index'] = i
                min_error['Value'] = val
                min_error['Error'] = error
                if head != None:
                    min_error['Attribute'] = head[i]

    return min_error


if __name__ == "__main__":
    md = {}
    df = pd.read_csv('data/housing3.csv')
    train, test = df[0:12], df[5:500]
    Y_train, X_train = np.array(train.Class), train.drop('Class', axis=1).to_numpy()
    Y_test, X_test = np.array(train.Class), train.drop('Class', axis=1).to_numpy()
    print(find_best_split(X_train, Y_train))
    print(train)
