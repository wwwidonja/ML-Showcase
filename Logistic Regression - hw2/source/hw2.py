import numpy as np
from scipy import optimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('seaborn-pastel')

translate = {'very poor': 0, 'poor': 1, 'average': 2, 'good': 3, 'very good': 4}
inv_translate = dict((y, x) for x, y in translate.items())


def scale_data(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)


def categoric(i, theta):
    prob = theta[i]
    if theta[i]< 0.0000000001:
        return 0.0000000001
    else: return prob


def ll_multinomial(beta, X, y, n_classes, optimizing=True):
    if optimizing:
        beta = np.append(beta, (len(X[0])+1) * [0])
    beta = beta.reshape((n_classes, len(X[0])+1))
    #print(beta)
    likelihood = 0
    for i in range(len(X)):
        u = np.dot(beta[:, :-1], X[i]) + beta[:, -1]
        probabilities = softmax(u)
        l = np.log(categoric(y[i], probabilities))
        likelihood += l
    return -likelihood


def softmax(U):
    denominator = sum([np.exp(u) for u in U])
    return [np.exp(u) / denominator for u in U]


def logistic(x):
    return 1 / (1 + np.exp(-x))


def ll_ordinal(beta, X, y, n_classes):
    ts = [0]
    deltas = beta[:n_classes-2]
    betaTrue = beta[n_classes-2:]
    betaTrue = betaTrue[:-1].reshape(1, len(X[0]))
    likelihood = 0
    for i in range(len(deltas)):
        ts.append(ts[i] + deltas[i])
    ts.insert(0, -np.inf)
    ts.append(np.inf)
    for i in range(len(X)):
        Xi = np.array(X[i])
        probabilities = []
        u = np.dot(betaTrue, Xi) + beta[-1]
        for j in range(1, n_classes + 1):
            p = logistic(ts[j] - u) - logistic(ts[j - 1] - u)
            probabilities.append(p)
        likelihood += np.log(categoric(y[i], probabilities))
    return -likelihood

def ll_baseline(probabilties, y):
    likelihood = 0
    for i in range(len(y)):
        likelihood += np.log(categoric(y[i], probabilties))
    return - likelihood


class Obj:
    def __init__(self, beta, shape, header=None):
        beta = np.append(beta, [0] * shape[1])
        beta = beta.reshape(shape)
        self.beta = beta
        self.header = header
        pass

    def predict(self, x):
        probabilities = softmax(np.dot(self.beta[:, :-1], x) + self.beta[:, -1])
        return probabilities.index(max(probabilities))


class Obj2:
    def __init__(self, beta, num_thresh):
        self.oldbeta = beta
        self.beta = beta[num_thresh:]
        deltas = beta[:num_thresh]
        ts = [0]
        for i in range(len(deltas)):
            ts.append(ts[i] + deltas[i])
        ts.insert(0, -np.inf)
        ts.append(np.inf)
        self.ts = ts

    def predict(self, x):
        probabilities = []
        bx = np.dot(self.beta[:-1], x) + self.beta[-1]
        for j in range(1, len(self.ts)):
            probabilities.append(logistic(self.ts[j] - bx) - logistic(self.ts[j - 1] - bx))
        return probabilities.index(max(probabilities))


class MultinomialLogReg:
    def __init__(self, X, Y, n_classes):
        self.num_c = n_classes
        self.num_f = len(X[0])
        self.beta = np.ones((self.num_f+1, self.num_c - 1)) / 2
        self.X = X
        self.Y = Y

        pass

    def build(self):
        self.beta = optimize.minimize(ll_multinomial, self.beta, args=(self.X, self.Y, self.num_c)).x
        o = Obj(self.beta, (self.num_c, self.num_f+1))
        return o
        pass

class BaselinePredictor:
    def __init__(self, ps):
        self.ps = ps

    def predict(self, x):
        return self.ps.index(max(self.ps))

class OrdinalLogReg:
    def __init__(self, X, Y, n_classes):
        self.X = X
        self.Y = Y
        self.levels = n_classes
        self.num_thresh = self.levels - 2
        self.beta = np.ones(len(self.X[0]) + 1) / 2
        self.deltas = np.ones(self.num_thresh) / 2
        self.bounds = ([(0.000000001, np.inf)] * self.num_thresh) + ([(-np.inf, np.inf)] * len(self.beta))
        self.bb = np.append(self.deltas, self.beta)

    def build(self):
        beta = optimize.fmin_l_bfgs_b(ll_ordinal, self.bb, args=(self.X, self.Y, self.levels), bounds=self.bounds, approx_grad=True)[
            0]

        o = Obj2(beta, self.num_thresh)
        return o


def accuracy(model, X, Y):
    Y_pred = []
    for x in X:
        Y_pred.append(model.predict(x))
    count = 0
    for i, j in zip(Y, Y_pred):

        if i == j: count += 1
    return count / len(Y)

def cross_val(X, Y):
    Xs = np.split(X, 5)
    Ys = np.split(Y, 5)
    o_acc = []
    m_acc = []
    m_loss = []
    o_loss = []
    baseline_acc = []
    baseline_loss = []

    ordinal_betas = []
    bsl_p = [0.15, 0.1, 0.05, 0.4, 0.3]
    print('Beginning cross eval')
    num_classes = len(np.unique(Y))
    for i in range(5):
        print(f'Now running pass {i+1}/5')
        testX, testY = Xs[i], Ys[i]
        trainX, trainY = np.concatenate([Xs[j] for j in range(5) if j != i]), \
                         np.concatenate([Ys[j] for j in range(5) if j != i])

        m = MultinomialLogReg(trainX, trainY, num_classes).build()
        ma = accuracy(m, testX, testY)
        m_acc.append(ma)
        ml = ll_multinomial(m.beta, testX, testY, num_classes, optimizing=False)
        if not np.isnan(ml): m_loss.append(ml/len(testY))
        o = OrdinalLogReg(trainX, trainY, num_classes).build()
        oa = accuracy(o, testX, testY)
        o_acc.append(oa)
        ol = ll_ordinal(o.oldbeta, testX, testY, num_classes)
        o_loss.append(ol/len(testY))
        ordinal_betas.append(o.beta)
        b = BaselinePredictor(bsl_p)
        baseline_acc.append(accuracy(b, testX, testY))
        baseline_loss.append(ll_baseline(bsl_p, testY) / len(testY))

    print('***MULTINOMIAL***')
    print(f'm_acc = {m_acc}, average = {sum(m_acc) / len(m_acc)}')
    print(f'm_loss = {m_loss}, average = {sum(m_loss) / len(m_loss)}')
    print('***ORDINAL***')
    print(f'o_acc = {o_acc}, average = {sum(o_acc) / len(o_acc)}')
    print(f'o_loss = {o_loss}, average = {sum(o_loss) / len(o_loss)}')

    acc = pd.DataFrame({'Baseline' : [i*100 for i in baseline_acc], 'Multinomial LR' : [i*100 for i in m_acc], 'Ordinal LR' : [i*100 for i in o_acc]})
    acci = acc.melt()
    acci.columns = ['Method', 'Accuracy']
    acci.to_csv('data/acci.csv')
    loss = pd.DataFrame({'Baseline' : baseline_loss, 'Multinomal LR': m_loss, 'Ordinal LR' : o_loss, })
    lossi = loss.melt()
    lossi.columns = ['Method', 'Log Loss']
    lossi.to_csv('data/lossi.csv')
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    sns.barplot(data = acci, y='Method', x='Accuracy', ax=ax[0])
    sns.barplot(data = lossi, y='Method', x='Log Loss', ax=ax[1])
    ax[0].set_title('Classification accuracy', fontsize=14)
    ax[1].set_title('Mean Log Loss', fontsize=14)
    ax[1].set_xlabel('Log Loss', fontsize=12)
    ax[0].set_xlabel('Accuracy [%]', fontsize=12)
    ax[0].set_ylabel('Method', fontsize=12)
    ax[1].set_ylabel('Method', fontsize=12)
    fig.tight_layout()
    plt.savefig('Loss plot2')
    plt.show()

    return ordinal_betas


def bs():
    df = pd.read_csv('data/dataset.csv', delimiter=';')
    df.sex = df.sex.astype('category').cat.codes
    df.response = df.response.astype('category').cat.reorder_categories(
        ['very poor', 'poor', 'average', 'good', 'very good']).cat.codes

    ordinal_betas = []
    num_classes = 5
    for i in range(100):
        print(f'Ordinal Iteration {i+1}/100')
        df2 = df.sample(frac=1, replace=True)
        X, Y = np.array(df2.drop('response', axis=1)), np.array(df2.response)
        X = scale_data(X)
        trainX, testX = X[:200], X[200:]
        trainY, testY = Y[:200], Y[200:]
        o = OrdinalLogReg(trainX, trainY, num_classes).build()
        ordinal_betas.append(o.beta)
    coeffs = {}
    for i in range(len(ordinal_betas)):
        coeffs[i] = ordinal_betas[i]
    betas = pd.DataFrame(coeffs).transpose()
    betas.columns = ['age', 'sex', 'year', 'X.1', 'X.2', 'X.3', 'X.4', 'X.5', 'X.6', 'X.7', 'X.8', 'B0']
    sns.barplot(data = betas, x='B0', ci='sd')
    plt.xlim(left=0, right=4)
    plt.xticks([0, 1, 2, 3, 4], ['very poor', 'poor', 'average', 'good', 'very good'])
    plt.xlabel('Class')
    plt.ylabel(r'$\beta_{0}$')
    b0mean = betas.B0.mean()
    plt.title('Intercept value ~ {:.2f}'.format(b0mean))
    sns.despine()
    plt.savefig('beta-sd', bbox_inches='tight')
    plt.show()
    df23 = betas.drop('B0', axis=1).melt()
    df23.columns = ['Feature', 'Coefficients']
    print(df23)
    plt.subplots(figsize=(8, 6))
    sns.barplot(data = df23, y= 'Feature', x='Coefficients', ci='sd')
    plt.vlines(x=0, ymin=-1, ymax=+11, ls='--')
    plt.title('Value of Beta Parameters in Ordinal Logistic Regression')
    plt.savefig('Coefficients-Bootstrap', bbox_inches='tight')
    plt.show()






if __name__ == "__main__":

    df = pd.read_csv('data/dataset.csv', delimiter=';')
    #print(df)
    """
    df.sex = df.sex.astype('category').cat.codes
    df.response = df.response.astype('category').cat.reorder_categories(
        ['very poor', 'poor', 'average', 'good', 'very good']).cat.codes

    X, Y = np.array(df.drop('response', axis=1)), np.array(df.response)
    X_scaled = scale_data(X)
    ordinal_betas = cross_val(X_scaled, Y)
    coeffs = {}
    for i in range(len(ordinal_betas)):
        coeffs[i] = ordinal_betas[i]
    betas = pd.DataFrame(coeffs).transpose()
    betas.columns = ['age', 'sex', 'year', 'X.1', 'X.2', 'X.3', 'X.4', 'X.5', 'X.6', 'X.7', 'X.8', 'B0']
    print(betas)
    betas.to_csv('data/betas.csv')
    """
    bs()