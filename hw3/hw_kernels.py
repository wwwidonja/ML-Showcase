import numpy as np
import pandas as pd #used only for reading
from sklearn.preprocessing import StandardScaler #Used only to scale data
import seaborn as sns #used only to get prettier plots :)
from matplotlib import pyplot as plt

plt.style.use('seaborn-deep')
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
from sklearn.metrics import mean_squared_error #used for internavl CV


class Predictor:

    def __init__(self, alpha, X, kernel):
        self.kernel = kernel
        self.alpha = alpha
        self.X = X

    def predict(self, Xprime):
        kx = self.kernel(Xprime, self.X)
        return np.dot(kx, self.alpha)


class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        self.lambda_ = lambda_
        self.kernel = kernel

    def fit(self, X, Y):
        self.X = X
        K = np.array(self.kernel(X, X))
        alpha = np.dot(np.linalg.inv(K + np.dot(self.lambda_, np.identity(len(X)))), Y)
        self.Y = Y
        return Predictor(alpha, X, self.kernel)


class RBF:
    def __init__(self, sigma):
        self.sigma = sigma
        pass

    def __call__(self, A, B):
        if A.ndim == 1:
            A = np.array([A])
        if B.ndim == 1:
            B = np.array([B])

        first_term = np.tile(np.sum(np.multiply(A, A), axis=1), (B.shape[0], 1)).T

        third_term = np.tile(np.sum(np.multiply(B, B), axis=1), (A.shape[0], 1))

        second_term = 2 * A.dot(B.T)

        e = (first_term - second_term + third_term) / (2 * pow(self.sigma, 2))

        ret = np.exp(e)
        if ret.shape == (1, 1):
            ret = ret[0][0]
        elif ret.shape[1] == 1 or ret.shape[0] == 1:
            ret = ret[0]
        return ret


class Polynomial:
    def __init__(self, M):
        self.M = M
        pass

    def __call__(self, A, B):
        e = np.array(pow((1 + np.dot(A, B.T)), self.M))
        return e


def cross_validation(Xstar, Y):
    best_lambda = []
    poly_rmse = []
    xstars = np.array_split(Xstar, 5)
    ys = np.array_split(Y, 5)
    for M in range(1, 5):
        curr_best_lambda = 0
        best_score = np.inf
        for l in np.arange(0.05, 10, 0.05):
            rmses = []
            for j in range(5):

                trainXstar, trainY = np.array([x for i, x in enumerate(xstars) if i != j][0]), np.array(
                    [x for i, x in enumerate(ys) if i != j][0])
                testXstar, testY = xstars[j], ys[j]
                krr = KernelizedRidgeRegression(Polynomial(M), l).fit(trainXstar, trainY)
                ypred = krr.predict(testXstar)
                rmse = mean_squared_error(ypred, testY, squared=False)
                rmses.append(rmse)
            mean_rmse = np.mean(rmses)
            if mean_rmse < best_score:
                best_score = mean_rmse
                curr_best_lambda = l
        poly_rmse.append(best_score)
        best_lambda.append(curr_best_lambda)
    rbf_rmse = []
    best_lambda_rbf = []
    for sigma in np.arange(5, 10.5, 0.5):
        curr_best_lambda = 0
        best_score = np.inf
        for l in np.arange(0.05, 10, 0.05):
            rmses = []
            for j in range(5):
                trainXstar, trainY = np.array([x for i, x in enumerate(xstars) if i != j][0]), np.array(
                    [x for i, x in enumerate(ys) if i != j][0])
                testXstar, testY = xstars[j], ys[j]
                krr = KernelizedRidgeRegression(RBF(sigma), l).fit(trainXstar, trainY)
                ypred = krr.predict(testXstar)
                rmse = mean_squared_error(ypred, testY, squared=False)
                rmses.append(rmse)
            mean_rmse = np.mean(rmses)
            if mean_rmse < best_score:
                best_score = mean_rmse
                curr_best_lambda = l
        rbf_rmse.append(best_score)
        best_lambda_rbf.append(curr_best_lambda)
    return best_lambda, best_lambda_rbf


def rmse_M(Xstar, Y):
    trainXstar, testXstar = Xstar[:160], Xstar[160:]
    trainY, testY = Y[:160], Y[160:]
    poly_rmse = []
    poly_rmse_best = []
    fig, ax = plt.subplots(2, 1)
    lambdas1, lambdas2 = cross_validation(trainXstar, trainY)

    #UNCOMMENT HERE IF YOU WANT TO SEE THE VALUES OF LAMBDAS, PRESENTED IN THE REPORT.
    print(f'lambdas1 = {lambdas1}')
    print(f'lambdas2 = {lambdas2}')
    for M in range(1, 5):
        krr = KernelizedRidgeRegression(Polynomial(M), 1).fit(trainXstar, trainY)
        krr_best = KernelizedRidgeRegression(Polynomial(M), lambdas1[M - 1]).fit(trainXstar, trainY)
        ypred = krr.predict(testXstar)
        ypred_best = krr_best.predict(testXstar)
        poly_rmse.append(mean_squared_error(ypred, testY, squared=False))
        poly_rmse_best.append(mean_squared_error(ypred_best, testY, squared=False))

    sns.lineplot(x=range(1, 5), y=poly_rmse, ax=ax[0], label=r'$\lambda=1$')
    sns.lineplot(x=range(1, 5), y=poly_rmse_best, ax=ax[0], label=r'best $\lambda$')
    ax[0].set_xlabel('M value', )
    ax[0].set_ylabel('RMSE', )
    ax[0].set_title('Polynomial kernel', )

    rbf_rmse = []
    rbf_rmse_best = []
    count = 0
    for sigma in np.arange(5, 10.5, 0.5):
        krr = KernelizedRidgeRegression(RBF(sigma), 1).fit(trainXstar, trainY)
        krr_best = KernelizedRidgeRegression(RBF(sigma), lambdas2[count]).fit(trainXstar, trainY)
        count += 1
        ypred2 = krr.predict(testXstar)
        ypred2_best = krr_best.predict(testXstar)
        rbf_rmse.append(mean_squared_error(ypred2, testY, squared=False))
        rbf_rmse_best.append(mean_squared_error(ypred2_best, testY, squared=False))
    df = pd.DataFrame({'x': rbf_rmse, 'y': np.arange(5, 10.5, 0.5)})
    df2 = pd.DataFrame({'x': rbf_rmse_best, 'y': np.arange(5, 10.5, 0.5)})
    df.y = df.y.astype('category')
    df2.y = df2.y.astype('category')
    sns.lineplot(x='y', y='x', data=df, ax=ax[1], label=r'$\lambda=1$')
    sns.lineplot(x='y', y='x', data=df2, ax=ax[1], label=r'best $\lambda$')
    ax[1].set_xlabel(r'$\sigma$ value')
    ax[1].set_ylabel(r'RMSE')
    ax[1].set_title('RBF kernel')
    sns.despine()

    fig.subplots_adjust(hspace=0.5)
    plt.savefig('parameter selection')
    plt.show()


def sine():
    sin = pd.read_csv('data/sine.csv')
    s = np.array(sin)
    X = np.array([s[:, 0]]).T
    scaler = StandardScaler().fit(X)
    Xstar = scaler.transform(X)
    Y = np.array([s[:, 1]]).T
    pol_M, pol_lambda = 10, 1
    krr = KernelizedRidgeRegression(Polynomial(10), 1).fit(Xstar, Y)
    rbf_sigma, rbf_lambda = 1.1, 1e-5
    krr2 = KernelizedRidgeRegression(RBF(rbf_sigma), rbf_lambda).fit(Xstar, Y)
    xt = np.array([np.arange(0, 20, 0.001)]).T
    yprime = krr.predict(scaler.transform(xt))
    yprime2 = krr2.predict(scaler.transform(xt))
    sns.scatterplot(x=X.T[0], y=Y.T[0], label='Sine dataset', alpha=0.4)
    sns.lineplot(x=xt.T[0], y=yprime.T[0], label=f'Polynomial (M={pol_M}), lambda = {pol_lambda}')
    sns.lineplot(x=xt.T[0], y=yprime2.T[0], label=f'RBF (sigma={rbf_sigma}), lambda = {rbf_lambda}')
    plt.savefig('Sine')
    plt.show()


if __name__ == '__main__':
    sine()
    housing = pd.read_csv('data/housing2r.csv')
    h = np.array(housing)
    X = np.array(h[:, :5])
    scaler = StandardScaler().fit(X)
    Xstar = scaler.transform(X)
    Y = np.array([h[:, 5]]).T
    rmse_M(Xstar, Y)
