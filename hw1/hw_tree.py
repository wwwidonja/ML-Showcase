import math
import random
import pandas as pd
import numpy as np
import itertools
import operator
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('seaborn-deep')


def most_common(L):
    SL = sorted((x, i) for i, x in enumerate(L))
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index

    return max(groups, key=_auxfun)[0]


class Node:
    def __init__(self, X, Y, min_samples, rand=random.Random(), isRF=False, gcc=None):
        self.gcc = gcc
        self.isRF = isRF
        if not isRF:
            self.X = X
            self.Y = Y
        else:
            indices = gcc()
            self.X = X[:, indices]
            self.Y = Y
        self.rand = rand
        self.end = True
        if (len(Y) >= min_samples) and (len(np.unique(Y)) > 1) and (len(np.unique(X)) > 1):
            self.end = False
            min_e = find_best_split(self.X, self.Y)
            self.split_i = min_e['Index']
            self.split_val = min_e['Value']
            self.l_child = Node(X=X[np.where(X[:, self.split_i] < self.split_val)],
                                Y=Y[np.where(X[:, self.split_i] < self.split_val)],
                                min_samples=min_samples,
                                rand=self.rand,
                                isRF=self.isRF,
                                gcc=self.gcc)
            self.r_child = Node(X=X[np.where(X[:, self.split_i] > self.split_val)],
                                Y=Y[np.where(X[:, self.split_i] > self.split_val)],
                                min_samples=min_samples,
                                rand=self.rand,
                                isRF = self.isRF,
                                gcc = self.gcc
                                )
        else:
            counts = {}
            for v in np.unique(Y):
                counts[v] = len(np.where(Y == v))
            self.majority = max(counts.items(), key=operator.itemgetter(1))[0]

    def predict_el(self, X):
        if self.end:
            return self.majority
        else:
            if X[self.split_i] > self.split_val:
                return self.r_child.predict_el(X)
            else:
                return self.l_child.predict_el(X)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_el(x))
        return np.array(predictions)


class MultiNode:
    def __init__(self, node_list):
        self.node_list = node_list

    def predict(self, X):
        predictions = []
        for i in range(len(self.node_list)):
            prediction = self.node_list[i].predict(X)
            predictions.append(prediction)

        predictions = np.array(predictions)
        final_predictions = []

        for i in range(len(predictions[0])):
            final_predictions.append(most_common(predictions[:, i]))
        return np.array(final_predictions)


class Tree:
    def __init__(self, rand, min_samples, get_candidate_columns):
        self.min_samples = min_samples
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns

    def build(self, X, Y):
        return Node(X, Y, self.min_samples, rand=self.rand)

    def rf_build(self, X, Y):
        return Node(X, Y, self.min_samples, rand=self.rand, isRF=True, gcc=self.get_candidate_columns)


class Bagging:
    def __init__(self, rand, tree_builder, n):
        self.rand = rand
        self.tree_builder = tree_builder
        self.n = n


    def build(self, X, Y):
        nodes = []
        for i in range(self.n):
            print(f'building tree number {i}/{self.n}')
            rand_indices = self.rand.choices(range(len(X)), k=len(X))
            boot_X = np.array([X[i] for i in rand_indices])
            boot_Y = np.array([Y[i] for i in rand_indices])
            nodes.append(self.tree_builder.build(boot_X, boot_Y))

        return MultiNode(nodes)


class RandomForest:
    def __init__(self, rand, n, min_samples):
        self.rand = rand
        self.n = n
        self.min_samples = min_samples

    def build(self, X, Y):
        nodes = []
        for i in range(self.n):
            def gcc():
                return self.rand.sample(range(len(self.X)), int(math.sqrt(len(self.X))))

            rand_indices = self.rand.choices(range(len(X)), k=len(X))
            boot_X = np.array([X[i] for i in rand_indices])
            boot_Y = np.array([Y[i] for i in rand_indices])
            t = Tree(self.rand, self.min_samples, gcc)
            node = t.build(boot_X, boot_Y)
            nodes.append(node)

        return MultiNode(nodes)


"""
HOMEWORK FUNCTIONS
"""


def hw_tree_full(tr, ts):
    X_tr, Y_tr = np.array(tr[0]), np.array(tr[1])
    X_ts, Y_ts = np.array(ts[0]), np.array(ts[1])

    def gcc():
        return [1, 2, 3]

    print('building tree')
    t = Tree(random.Random(), 2, gcc).build(X_tr, Y_tr)
    print('Making test predictions')
    test_predictions = t.predict(X_ts)
    print('Making train predictions')
    train_predictions = t.predict(X_tr)
    ts_matches = [i == j for i, j in zip(test_predictions, Y_ts)]
    tr_matches = [i == j for i, j in zip(train_predictions, Y_tr)]
    bars = [1- (sum(tr_matches) / len(tr_matches)), 1- (sum(ts_matches) / len(ts_matches))]
    fig,ax = plt.subplots()
    return bars[0], bars[1]


def hw_cv_min_samples(tr, ts):
    X_tr, Y_tr = np.array(tr[0]), np.array(tr[1])
    X_ts, Y_ts = np.array(ts[0]), np.array(ts[1])
    np.random.seed(1)
    np.random.shuffle(X_tr)
    np.random.seed(1)
    np.random.shuffle(Y_tr)
    split_Xtr = np.array_split(X_tr, 5)
    split_Ytr = np.array_split(Y_tr, 5)

    mc_rates = {'test': [], 'train': [], 'sample_sizes': []}
    best_cv_acc = {'sample_size': -100, 'mcr': 1}
    for sample_size in range(1, 30, 1):
        print(f'Now computing for sample size {sample_size}')
        mc_rates['sample_sizes'].append(sample_size)
        cv_acc_ts = []
        cv_acc_tr = []
        t = Tree(random.Random(), sample_size, gcc)
        for _pass in range(5):
            print(f'pass {_pass + 1}/5')
            CV_Xtr, CV_Xts = np.concatenate([y for i, y in enumerate(split_Xtr) if i != _pass]), split_Xtr[_pass]
            CV_Ytr, CV_Yts = np.concatenate([y for i, y in enumerate(split_Ytr) if i != _pass]), split_Ytr[_pass]
            dt = t.build(CV_Xtr, CV_Ytr)
            predictions_ts = dt.predict(CV_Xts)
            predictions_tr = dt.predict(CV_Xtr)
            matches_ts = [i == j for i, j in zip(predictions_ts, CV_Yts)]
            matches_tr = [i==j for i,j in zip(predictions_tr, CV_Ytr)]
            cv_acc_ts.append(sum(matches_ts) / len(matches_ts))
            cv_acc_tr.append(sum(matches_tr) / len(matches_tr))
        cv_accuracy_ts, cv_accuracy_tr = sum(cv_acc_ts) / len(cv_acc_ts), sum(cv_acc_tr) / len(cv_acc_tr)
        cv_mc_rate_ts, cv_mc_rate_tr = 1 - cv_accuracy_ts, 1 - cv_accuracy_tr
        if cv_mc_rate_ts <= best_cv_acc['mcr']:
            best_cv_acc['mcr'] = cv_mc_rate_ts
            best_cv_acc['sample_size'] = sample_size
        mc_rates['test'].append(cv_mc_rate_ts)
        mc_rates['train'].append(cv_mc_rate_tr)


    adf = pd.DataFrame(mc_rates)
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(data=adf, x='sample_sizes', y='test', label='test', ax=ax[0])
    sns.lineplot(data=adf, x='sample_sizes', y='train', label='train', ax=ax[0])
    ax[0].vlines(x=best_cv_acc['sample_size'], ymin = 0, ymax = 0.2, ls='--', label='best_threshold')
    ax[0].set_title('Internal Cross-Validation Misclassification rates')
    ax[0].set_ylabel('Misclassification rate')
    ax[0].set_xlabel('Node split threshold (min_samples)')
    #plt.show()

    t = Tree(random.Random(), best_cv_acc['sample_size'], gcc).build(X_tr, Y_tr)
    test_predictions = t.predict(X_ts)
    test_matches = [i==j for i, j in zip(test_predictions, Y_ts)]
    train_predictions = t.predict(X_tr)
    train_matches = [i==j for i, j in zip(train_predictions, Y_tr)]
    fin_misclas = {'test' : 1 - (sum(test_matches)/len(test_matches)), 'train' : 1-(sum(train_matches)/len(train_matches))}
    sns.barplot(x=['test', 'train'], y=[fin_misclas['test'], fin_misclas['train']], ax=ax[1])
    ax[1].set_title('Full data')
    ax[1].set_ylabel('Misclassification rate')
    ax[1].set_xlabel('Node split threshold (min_samples)')
    plt.show()

    return fin_misclas['train'], fin_misclas['test'], best_cv_acc['sample_size']





def hw_bagging(tr, ts):
    X_tr, Y_tr = np.array(tr[0]), np.array(tr[1])
    X_ts, Y_ts = np.array(ts[0]), np.array(ts[1])

    ntrees = {'num_of_trees' : [], 'test' : [], 'train' : []}
    for num_of_trees in range(5, 80, 5):
        print(f'num of trees = {num_of_trees}/80')
        for i in range(5):
            bg = Bagging(random.Random(), Tree(random.Random().seed(i+1), 2, gcc), num_of_trees).build(X_tr, Y_tr)
            predictions_ts = bg.predict(X_ts)
            predictions_tr = bg.predict(X_tr)
            matches_ts = [i==j for i,j in zip(predictions_ts, Y_ts)]
            matches_tr = [i == j for i, j in zip(predictions_tr, Y_tr)]
            ntrees['num_of_trees'].append(num_of_trees)
            ntrees['test'].append(1- (sum(matches_ts)/len(matches_ts)))
            ntrees['train'].append(1 - (sum(matches_tr) / len(matches_tr)))

    df = pd.DataFrame(ntrees)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='num_of_trees', y='test', label='test')
    sns.lineplot(data=df, x='num_of_trees', y='train', label='train')
    ax.set_title('Effect of Number of trees on performance')
    ax.set_ylabel('Misclassification rate (CI=95%)')
    ax.set_xlabel('Number of trees')
    plt.suptitle('bagging')
    plt.show()


    return ntrees['train'][9], ntrees['test'][9]

    """
    Use bagging with n=50 trees with min_samples=2. Return misclassification rates on training and testing data
    :return:
    """



def gini(Y):
    Nm = len(Y)
    to_sum = []
    for c in np.unique(Y):
        count = len(np.where(Y == c)[0])
        pi_c = count / Nm
        to_sum.append(pi_c * (1 - pi_c))
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
    train, test = df[:400], df[400:]
    Y_train, X_train = np.array(train.Class), train.drop('Class', axis=1).to_numpy()
    Y_test, X_test = np.array(test.Class), test.drop('Class', axis=1).to_numpy()
    train = X_train, Y_train
    test = X_test, Y_test


    def gcc():
        return


    #hw_tree_full(train, test)
    #hw_cv_min_samples(train, test)
    #hw_bagging(train, test)
