import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss
import seaborn as sns
from matplotlib import pyplot as plt

plt.style.use('seaborn')
import math
from pytictoc import TicToc

"""
NOTE - THE NUMBER OF EPOCHS (!), LEARNING RATE, REGULARIZATION AND MOMENTUM PARAMETERS
ARE CURRENTLY SET TO CONFORM TO THE PROVIDED UNIT TESTS. 
CONSIDER SIGNIFICANTLY CHANGING THEM FOR IMPLEMENTATION OR ELSE RESULTS FROM THE REPORT WILL NBOT
BE REPRODUCED
"""

EPOCHS = 50*1000
LAMBDA = 1e-5
ALPHA = 1e-1
MU = 0.7


def softmax(Z):
    e_x = np.exp(np.array(Z).astype(float) - np.max(np.array(Z).astype(float)))
    return e_x / e_x.sum(axis=0)


def ReLU(Z):
    return np.maximum(0, Z)


def deriv_relu(Z):
    return Z > 0


def log_loss(pred, y):
    loss = 0
    for i in range(len(pred)):
        id = np.argmax(y[i])
        loss += np.log(pred[i][id])
    return -loss


def get_predictions(A2):
    return np.argmax(A2, 0)


def rmse(pred, y):
    return math.sqrt(np.mean((np.reshape(pred, (pred.shape[0],)) - y) ** 2))


def onehot(Y):
    l = len(np.unique(Y))
    oh = np.zeros((len(Y), l))
    oh[np.arange(Y.size), Y] = 1
    return oh.T


class predictor():
    def __init__(self, weights, biases):
        self.weights_ = weights
        self.biases = biases

    def predict(self, X_test):
        X_test = ss().fit_transform(X_test).T
        A2 = self.forward_prop(X_test)
        A2 = A2.T
        return A2

    def forward_prop(self, X):  # Dummy method, overwritten in children
        a = np.zeros((10, 10))
        return a

    def weights(self):
        ret = [weight.T for weight in self.weights_]
        biases = [bias.T for bias in self.biases]
        fin = [np.append(weight, bias, axis=0) for (weight, bias) in zip(ret, biases)]
        return fin


class classifier(predictor):

    def forward_prop(self, X):
        activated = [X]
        unactivated = []
        for i in range(len(self.weights_) - 1):
            Zx = self.weights_[i].dot(activated[i]) + self.biases[i]
            unactivated.append(Zx)
            Ax = ReLU(Zx)
            activated.append(Ax)
        Z_fin = self.weights_[-1].dot(activated[-1]) + self.biases[-1]
        A_fin = softmax(Z_fin)
        return A_fin

    def set_encoders(self, encoder):
        self.encoder = encoder
        self.decoder = {value: key for (key, value) in encoder.items()}

    def predict_most_prob(self, X_test):
        A2 = self.predict(X_test).T
        indices = get_predictions(A2)
        A2 = A2.T
        fin = np.zeros(A2.shape)
        for i in range(len(fin)):
            fin[i, indices[i]] = 1
        return fin


class regressor(predictor):

    def predict(self, X_test):
        X_test = ss().fit_transform(X_test).T
        A2 = self.forward_prop(X_test)
        fin = A2
        return fin[0]

    def forward_prop(self, X):
        activated = [X]
        unactivated = []
        for i in range(len(self.weights_) - 1):
            Zx = self.weights_[i].dot(activated[i]) + self.biases[i]
            unactivated.append(Zx)
            Ax = ReLU(Zx)
            activated.append(Ax)
        Z_fin = self.weights_[-1].dot(activated[-1]) + self.biases[-1]
        A_fin = Z_fin
        return A_fin


class ANN:
    def __init__(self, units, lambda_=LAMBDA, momentum=MU, verbose=False):
        self.regularization = lambda_
        self.dims = units
        self.momentum = momentum
        self.verbose = verbose

    def fit(self, X_train, Y_train, lr=ALPHA, epochs=EPOCHS):
        self.X_train = ss().fit_transform(X_train).T

        Y_train = Y_train.T

        self.y_size = Y_train.size
        self.lr = lr
        self.set_dims(Y_train)
        self.weights = [np.random.rand(self.dims[i], self.dims[i - 1]) - 0.5 for i in range(1, len(self.dims))]
        self.old_weights = self.weights.copy()
        self.biases = [np.random.rand(self.dims[i], 1) - 0.5 for i in range(1, len(self.dims))]
        self.old_biases = self.biases.copy()
        # self.encode(Y_train)
        self.encode(Y_train)
        loss = self.gradient_descent(lr, epochs)
        return self.make_predictor()

    def set_dims(self, Y_train):
        self.dims = self.dims  # dummy method, to be overridden in children

    def encode(self, Y):
        self.Y_train = Y
        pass  # dummy method, to be overridden in children

    def get_loss(self):  ##Dummy method, to be overridden in children
        def dummy_loss(a, b): return np.inf

        return dummy_loss

    def make_predictor(self):
        return predictor(self.weights, self.biases)  # dummy method, to be overridden in children

    def forward_prop(self, X):
        activated = [X]
        unactivated = []
        for i in range(len(self.weights) - 1):
            Zx = self.weights[i].dot(activated[i]) + self.biases[i]
            unactivated.append(Zx)
            Ax = ReLU(Zx)
            activated.append(Ax)
        Z_fin = self.weights[-1].dot(activated[-1]) + self.biases[-1]
        A_fin = self.process_Afin(Z_fin)
        return activated, unactivated, A_fin

    def backprop(self, activated, unactivated, A_fin):
        a = activated[::-1]
        print(activated[0].shape)
        u = unactivated[::-1]
        w = self.weights[::-1]
        dZx = A_fin - self.Y_train
        dWs = [1 / self.y_size * dZx.dot(a[0].T)]
        dbs = [1 / self.y_size * np.sum(dZx)]
        for i in range(1, len(a)):
            dZx = w[i - 1].T.dot(dZx) * deriv_relu(u[i - 1])
            dWs.append(1 / self.y_size * dZx.dot(a[i].T))
            dbs.append(1 / self.y_size * np.sum(dZx))
        return dWs[::-1], dbs[::-1]

    def process_Afin(self, Z_fin):
        return Z_fin

    def update_params(self, dws, dbs, alpha):
        w = list(zip(self.weights, dws))
        b = list(zip(self.biases, dbs))
        ow = self.weights.copy()
        ob = self.biases.copy()
        for i in range(len(w)):
            self.weights[i] = w[i][0] - alpha * (
                    w[i][1] + self.regularization * w[i][0]) + self.momentum * (
                                      self.weights[i] - self.old_weights[i])
        for i in range(len(b)):
            self.biases[i] = b[i][0] - alpha * b[i][1] + self.momentum * (self.biases[i] - self.old_biases[i])
        self.old_weights = ow.copy()
        self.old_biases = ob.copy()
        return

    def gradient_descent(self, alpha, iterations):
        last_results = []
        best_weights = []
        best_biases = []
        best_result = np.inf
        loss_f = self.get_loss()
        for i in range(iterations):
            a, u, a_fin = self.forward_prop(self.X_train)
            dws, dbs = self.backprop(a, u, a_fin)
            self.update_params(dws, dbs, alpha)
            if i % 10 == 0:
                if self.verbose: print("Iteration: ", i)

                _, _, a_fin_val = self.forward_prop(self.X_train)
                try:
                    loss = loss_f(a_fin_val.T, self.Y_train)
                except IndexError:
                    loss = loss_f(a_fin_val.T, self.Y_train.T)
                if loss < 1e-7: break
                if loss < best_result:
                    best_weights = self.weights
                    best_biases = self.biases
                    best_result = loss
                if self.verbose: print(f'loss : {loss:.2f}')
                if all(i <= loss for i in last_results) and len(last_results) == 20:
                    if self.momentum > 1e-3:
                        self.momentum /= 10
                        if len(last_results) >= 20: last_results.pop()
                        last_results.insert(0, loss)
                        continue
                    if alpha > 1e-8:
                        alpha /= 10
                        if self.verbose: print(f'Alpha updated! Now {alpha}')
                    else:
                        break  ## change to validation
                if len(last_results) >= 20: last_results.pop()
                last_results.insert(0, loss)
        self.weights = best_weights
        self.biases = best_biases
        return best_result


class ANNClassification(ANN):
    def process_Afin(self, Z_fin):
        return softmax(Z_fin)

    """
    Gradient checking method 
    (draft, not working completely)
    
    def check_grads(self, dws):
        eps = 1e-2
        weights = self.get_weights()
        perturb = [np.zeros(weight.shape) for weight in weights]
        numGrad = [np.zeros(weight.shape) for weight in weights]
        for i in range(len(weights)):
            for p in range(len(weights[i])):
                for j in range(len(weights[i][p])):
                    perturb[i][p][j] = eps
                    #print(perturb)
                    #print('\n\n\n')
                    self.set_weights([weight + perturb for (weight, perturb) in zip(weights, perturb)])
                    _,_, a = self.forward_prop()
                    Jpos = log_loss(a.T, self.encoded_Y.T)
                    self.set_weights([weight - perturb for (weight, perturb) in zip(weights, perturb)])
                    _,_, a = self.forward_prop()

                    Jneg = log_loss(a.T, self.gts)
                    numGrad[i][p][j] = (Jpos - Jneg) / (2*eps)
                    perturb[i][p][j] = 0
            #print(dws[i])
            #print('\n\n')
            #print(numGrad[i])
            #print(f'Numeric gradient difference in dimension [{i}] ={np.linalg.norm(dws[i] - numGrad[i]) / np.linalg.norm(dws[i] + numGrad[i])}')

        self.set_weights(weights)
    """

    def set_dims(self, Y_train):
        self.dims = [self.X_train.shape[0]] + self.dims + [len(np.unique(Y_train))]

    def encode(self, Y):
        self.dictionary = {i[1]: i[0] for i in enumerate(np.unique(Y))}
        self.Y_train = onehot(np.array([self.dictionary[i] for i in Y]))
        self.gts = [self.dictionary[i] for i in Y]
        pass  # dummy method, to be overridden in children

    def get_loss(self):  ##Dummy method, to be overridden in children
        def normed_log_loss(pred, y):
            return log_loss(pred, y) / self.y_size

        return normed_log_loss

    def make_predictor(self):
        c = classifier(self.weights, self.biases)
        c.set_encoders(self.dictionary)
        return c


class ANNRegression(ANN):
    def set_dims(self, Y_train):
        self.dims = [self.X_train.shape[0]] + self.dims + [1]

    def get_loss(self): return rmse

    def make_predictor(self):
        return regressor(self.weights, self.biases)  # dummy method, to be overridden in children


"""
Assignment nr. 3
"""


def housing():
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    regr = pd.DataFrame({'RMSE': [], 'units': []})
    for unit in [[], [10], [10, 10], [200], [10, 30, 10], [30, 10, 30]]:
        print(f'Now computing {unit} for regression')
        losses = check(unit, method='Regression')
        if unit == []: unit = 'No Hidden Layers'
        regr = regr.append(pd.DataFrame({'RMSE': losses, 'units': 5 * [f'{unit}']}))
    sns.barplot(data=regr, x='RMSE', y='units', ax=ax[0], ci='sd')
    ax[0].set_title('Regression')
    ax[0].set_xlabel('Average RMSE')
    classif = pd.DataFrame({'mislas': [], 'units': []})
    for unit in [[10], [10, 10], [200], [10, 30, 10], [30, 10, 30]]:
        print(f'Now computing {unit} for classification')
        losses = check(unit, method='Classification')
        if unit == []: unit = 'No Hidden Layers'
        classif = classif.append(pd.DataFrame({'mislas': losses, 'units': 5 * [f'{unit}']}))
    classif.mislas = [i * 100 for i in classif.mislas]
    sns.barplot(data=classif, x='mislas', y='units', ax=ax[1], ci='sd')
    ax[1].set_title('Classification')
    ax[1].set_xlabel('Average Misclassification rate [%]')
    plt.subplots_adjust(hspace=0.4, left=0.25)
    plt.savefig('CV layer size comparison', bbox_inches='tight')
    plt.show()


def check(units, folds=5, method='Regression', ):
    if method == 'Regression':
        df = pd.read_csv('./housing2r.csv')
    elif method == 'Classification':
        df = pd.read_csv('./housing3.csv')
    else:
        raise NameError('Please input relevant method (\'Regression\', \'Classification\')')
    data = np.array(df)
    np.random.seed(1)
    np.random.shuffle(data)
    Xs = np.array_split(data[:, :-1], folds)
    Ys = np.array_split(data[:, -1], folds)
    losses = []
    for fold in range(folds):
        print(f'Fold {fold + 1}/{folds}')
        trainX, trainY = np.array([x for i, x in enumerate(Xs) if i != fold][0]), np.array(
            [x for i, x in enumerate(Ys) if i != fold][0])
        testX, testY = Xs[fold], Ys[fold]
        if method == 'Regression':
            r = ANNRegression(verbose=False, units=units).fit(trainX, trainY)
            pred = r.predict(testX)
            print(pred)
            loss = rmse(pred, testY)
            losses.append(loss)

        elif method == 'Classification':
            c = ANNClassification(verbose=False, units=units).fit(trainX, trainY)
            gts = [c.encoder[i] for i in testY.T]
            pred = [np.argmax(i) for i in c.predict(testX)]
            misclassification = 1 - sum([p == y for (p, y) in zip(pred, gts)]) / len(gts)
            losses.append(misclassification)
    return losses


"""
Asssignment nr. 4
"""


def grid_search():
    tt = TicToc()
    classif = pd.DataFrame({'misclas': [], 'units': [], 'lambda': [], })
    for unit in [[10], [50], [5, 5], [5, 10, 5], [10, 5, 10]]:
        print(f'Running for unit {unit}')
        for lambda_ in [1e-3, 1e-4, 1e-5]:
            print(f'Running for lambda {lambda_}')
            tt.tic()
            misclas, lls = make(unit, lambda_)
            tt.toc()
            classif = classif.append(
                pd.DataFrame({'misclas': misclas, 'log_loss': lls, 'units': 5 * [f'{unit}'], 'lambda': 5 * [lambda_]}))
    return classif


def make(unit, lambda_):
    df = pd.read_csv('train.csv')
    df = df.drop(df.columns[0], axis=1)
    data = np.array(df)
    np.random.seed(1)
    np.random.shuffle(data)
    folds = 5
    Xs = np.array_split(data[:, :-1], folds)
    Ys = np.array_split(data[:, -1], folds)
    misclas = []
    for fold in range(folds):
        print(f'FOLD {fold + 1}/{folds}')
        trainX, trainY = np.array([x for i, x in enumerate(Xs) if i != fold][0]), np.array(
            [x for i, x in enumerate(Ys) if i != fold][0])
        testX, testY = Xs[fold], Ys[fold]
        c = ANNClassification(verbose=False, units=unit, lambda_=lambda_).fit(trainX, trainY, lr=0.1)
        pred = c.predict(testX)
        gts = [c.encoder[i] for i in testY.T]
        pred = [np.argmax(i) for i in pred]
        misclassification = 1 - sum([p == y for (p, y) in zip(pred, gts)]) / len(gts)
        misclas.append(misclassification)
    return misclas

def plot_big(df=None):
    """
    This file assumes that the output of grid_search is saved in the local directory as ('./cv_df_output.pkl') or
    passed as a parameter.
    """
    if df==None:
        df = pd.read_pickle('./cv_df_output.pkl')
        try:
            df = df.drop('log_loss', axis=1)
        except IndexError: print('No redundant loss column in df')
    fig, ax = plt.subplots(figsize=(6,6))
    pal = sns.color_palette()
    sns.lineplot(x='lambda', y='misclas', hue='units', data=df, palette=[pal[i] for i in list(range(4))+[5]])
    ax.set_xscale('log')
    ax.set_xlabel(r'Regularization coefficient $\lambda$')
    ax.set_yticks([0.23, 0.24, 0.25, 0.26])
    ax.set_yticklabels([23, 24, 25, 26])
    ax.set_ylabel('Misclassification rate [%]')
    #ax.grid(False)
    leg = ax.get_legend()
    leg.set_title('Hidden Layer Arrangement')
    plt.savefig('Internal CV', bbox_inches='tight')
    plt.show()
    return


def create_final_predictions(units, lambda_=1e-3, alpha=0.1, mu=0.3):
    df = pd.read_csv('train.csv')
    df = df.drop(df.columns[0], axis=1)
    data = np.array(df)
    trainX = data[:, :-1]
    trainY = data[:, -1]
    classifier = ANNClassification(verbose=True, units=units, lambda_=lambda_, momentum=mu).fit(trainX, trainY,
                                                                                                 lr=alpha)
    df = pd.read_csv('test.csv')
    df = df.drop(df.columns[0], axis=1)
    testX = np.array(df)
    pred = classifier.predict(testX)
    df = pd.DataFrame(pred)
    df.to_csv('./final_no_colnames.txt')
    df.columns = [f'Class_{i}' for i in range(1, 10)]
    df.to_csv('./final.txt')
    reformat_file()
    return df

def reformat_file(df=None):
    if df==None:
        df = pd.read_csv('./final.txt')
    df.columns = ['id'] + [f'Class_{i}' for i in range(1,10)]
    df.id = [i+1 for i in df.id]
    df.to_csv('./final.txt', index=False)


if __name__ == "__main__":
    """
    Uncomment the following line to output Figure 1 (fit on hosuing)
    """
    housing()

    """
    Uncomment following lines to get dataframe on which Figure 2 is drawn (Grid search CV for
    big dataset (train)
    """
    #classif = grid_search()
    #classif.to_pickle('./cv_df_output.pkl')
    #plot_big()

    """
    Uncomment the following lines to make prediction (4)
    """
    #df = create_final_predictions(units=[50])
