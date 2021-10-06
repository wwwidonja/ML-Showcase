import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from cvxopt import solvers
from cvxopt import matrix
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
from sklearn.metrics import mean_squared_error #used for internavl CV
plt.style.use('seaborn-deep')


def transformToMatrix(npobject):
    return matrix(npobject, tc='d')


class Predictor:

    def __init__(self, kernel, X, alphas, b):
        self.kernel = kernel
        self.X = X
        self.alphas = alphas
        self.b = b
        pass

    def get_alpha(self):
        return self.alphas

    def get_b(self):
        return self.b[0]

    def predict(self, Xprime):
        K = self.kernel(self.X, Xprime)
        adj_alphas = np.dot(self.alphas, np.array([1, -1]))
        return np.dot(K.T, adj_alphas) + self.b


class SVR:
    def __init__(self, kernel, lambda_, epsilon=1, small_eps=1e-6):
        self.lambda_, self.kernel, self.epsilon = lambda_, kernel, epsilon
        self.vectors = []
        self.small_eps = small_eps
        pass

    def fit(self, X, Y):
        C = 1 / self.lambda_

        K = np.array(self.kernel(X, X))
        prefix_array = np.array([[1, -1], [-1, 1]])
        k02 = K.shape[0]*2
        y0 = Y.shape[0]
        prefixes = np.tile(prefix_array, K.shape)  # constructs an array of interchanging prefixes the size of K.
        shape = (K.shape[1], k02)
        pp = np.repeat(np.repeat(K, 2).reshape(shape), 2,axis=0)
        P = np.multiply(pp, prefixes)

        eps = self.epsilon * np.ones(P.shape[0])
        yy = np.multiply(np.repeat(Y, 2).reshape(1, y0 * 2), np.tile(prefix_array[0], (1, y0)))
        q = transformToMatrix((eps - yy).T)

        h = transformToMatrix(np.repeat([C, 0], k02))

        b = transformToMatrix(0.)

        G = transformToMatrix(np.vstack([np.identity(k02), -np.identity(k02)]))

        A = transformToMatrix(np.tile(prefix_array[0], (1, y0)))

        solvers.options['show_progress'] = False
        res = solvers.qp(transformToMatrix(P), q, G, h, A, b)
        alphas = np.reshape(np.array(res['x']), (len(X), 2))

        vector_diff = np.dot(alphas, np.array([1, -1]))
        YW = Y - np.array([np.dot(K, vector_diff)]).T

        alphai_1 = alphas < C
        alphaistar_1 = alphas > 0
        indices_box_1 = np.logical_or(alphai_1[:, 1], alphaistar_1[:, 0])


        alphai_2 = alphas > 0
        alphaistar_2 = alphas < C
        indices_box_2 = np.logical_or(alphai_2[:, 1], alphaistar_2[:, 0])

        lower = max((self.epsilon+YW)[indices_box_1])
        upper = min((-self.epsilon+YW)[indices_box_2])

        b_fin = (lower[0] + upper[0]) / 2  #get b as the mean of the bounds
        self.vectors = np.where(abs(vector_diff) >= self.small_eps)
        return Predictor(self.kernel, X, alphas, np.array([b_fin]))

def cross_validation(Xstar, Y):
    best_lambda = []
    poly_rmse = []
    xstars = np.array_split(Xstar, 5)
    ys = np.array_split(Y, 5)
    for M in range(1, 11):
        curr_best_lambda = 0
        best_score = np.inf
        for l in np.arange(0.1, 10.1, 0.1):
            rmses = []
            for j in range(5):
                trainXstar, trainY = np.array([x for i, x in enumerate(xstars) if i != j][0]), np.array(
                    [x for i, x in enumerate(ys) if i != j][0])
                testXstar, testY = xstars[j], ys[j]
                krr = SVR(Polynomial(M), l, epsilon=8).fit(trainXstar, trainY)
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
    for sigma in np.arange(2, 10.5, 0.5):
        curr_best_lambda = 0
        best_score = np.inf
        for l in np.arange(0.1, 10.1, 0.1):
            rmses = []
            for j in range(5):
                trainXstar, trainY = np.array([x for i, x in enumerate(xstars) if i != j][0]), np.array(
                    [x for i, x in enumerate(ys) if i != j][0])
                testXstar, testY = xstars[j], ys[j]
                krr = SVR(RBF(sigma), l,epsilon=10).fit(trainXstar, trainY)
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
    fig, ax = plt.subplots(2, 2)
    lambdas1, lambdas2 = cross_validation(trainXstar, trainY)

    #UNCOMMENT HERE IF YOU WANT TO SEE THE VALUES OF LAMBDAS, PRESENTED IN THE REPORT.
    #print(f'lambdas1 = {lambdas1}')
    #print(f'lambdas2 = {lambdas2}')
    vecs = []
    vecs_best = []
    for M in range(1, 11):
        svr_fitter = SVR(Polynomial(M), 1, epsilon=8, small_eps=1e-6)
        svr = svr_fitter.fit(trainXstar, trainY)
        svr_fitter_best = SVR(Polynomial(M), lambdas1[M - 1], epsilon=8, small_eps=1e-6)
        svr_best = svr_fitter_best.fit(trainXstar, trainY)
        vecs.append(len(svr_fitter.vectors[0]))
        print(vecs)
        vecs_best.append(len(svr_fitter_best.vectors[0]))
        print(vecs_best)
        ypred = svr.predict(testXstar)
        ypred_best = svr_best.predict(testXstar)
        poly_rmse.append(mean_squared_error(ypred, testY, squared=False))
        poly_rmse_best.append(mean_squared_error(ypred_best, testY, squared=False))

    sns.lineplot(x=range(1, 11), y=poly_rmse, ax=ax[0][0], label=r'$\lambda=1$', color=sns.color_palette()[1])
    sns.lineplot(x=range(1, 11), y=poly_rmse_best, ax=ax[0][0], label=r'best $\lambda$', color=sns.color_palette()[2])
    sns.lineplot(x=range(1, 11), y=vecs, ax=ax[0][1], label=r'$\lambda=1$', color=sns.color_palette()[1])
    sns.lineplot(x=range(1, 11), y=vecs_best, ax=ax[0][1], label=r'best $\lambda$', color=sns.color_palette()[2])

    ax[0][0].set_xlabel('M value', )
    ax[0][0].set_ylabel('RMSE', )
    ax[0][0].set_title('Polynomial kernel', )
    ax[0][1].set_xlabel('M value', )
    ax[0][1].set_ylabel('Support vector count', )

    vecsi = []
    vecs_besti = []
    rbf_rmse = []
    rbf_rmse_best = []
    count = 0
    for sigma in np.arange(2, 10.5, 0.5):
        print(f'sigma = {sigma}/2')
        svr_fitter = SVR(RBF(sigma), 1, epsilon=10, small_eps=1e-6)
        svr = svr_fitter.fit(trainXstar, trainY)
        vecsi.append(len(svr_fitter.vectors[0]))
        svr_fitter_best = SVR(RBF(sigma), lambdas2[count], epsilon=10, small_eps=1e-6)
        svr_best = svr_fitter_best.fit(trainXstar, trainY)
        vecs_besti.append(len(svr_fitter_best.vectors[0]))
        print(vecsi)
        print(vecs_besti)
        count += 1
        ypred2 = svr.predict(testXstar)
        ypred2_best = svr_best.predict(testXstar)
        rbf_rmse.append(mean_squared_error(ypred2, testY, squared=False))
        rbf_rmse_best.append(mean_squared_error(ypred2_best, testY, squared=False))

    df = pd.DataFrame({'x': rbf_rmse, 'y': np.arange(2, 10.5, 0.5)})
    df2 = pd.DataFrame({'x': rbf_rmse_best, 'y': np.arange(2, 10.5, 0.5)})
    df.y = df.y.astype('category')
    df2.y = df2.y.astype('category')
    sns.lineplot(x='y', y='x', data=df, ax=ax[1][0], label=r'$\lambda=1$', color = sns.color_palette()[1])
    sns.lineplot(x='y', y='x', data=df2, ax=ax[1][0], label=r'best $\lambda$', color = sns.color_palette()[2])
    sns.lineplot(x=df['y'], y=vecsi, ax=ax[1][1], label=r'$\lambda=1$', color = sns.color_palette()[1])
    sns.lineplot(x=df2['y'], y=vecs_besti, ax=ax[1][1], label=r'best $\lambda$', color = sns.color_palette()[2])
    ax[1][0].set_xlabel(r'$\sigma$ value', )
    ax[1][0].set_ylabel('RMSE', )
    ax[1][0].set_title('RBF kernel', )
    ax[1][1].set_xlabel(r'$\sigma$ value', )
    ax[1][1].set_ylabel('Support vector count', )
    sns.despine()
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig('parameter selection')
    plt.show()


def standin_norm(a, b):
    return np.sum(np.multiply(a, b), axis=1)

class RBF():

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, A, B):
        normA = np.array([np.linalg.norm(A, axis=1) ** 2])
        normB = np.array([np.linalg.norm(B, axis=1) ** 2])

        first_term = np.repeat(normA.T, B.shape[0], axis=1)
        third_term = np.repeat(normB, A.shape[0], axis=0)
        second_term = 2 * (A.dot(B.T))
        e = - 1/ (2* pow(self.sigma, 2)) * (first_term - second_term + third_term)

        k = np.squeeze(np.exp(e))
        return k


class Polynomial:
    def __init__(self, M):
        self.M = M
        pass

    def __call__(self, A, B):
        e = np.array(pow((1 + np.dot(A, B.T)), self.M))
        return e


def sine():
    sin = pd.read_csv('sine.csv')
    s = np.array(sin)
    X = np.array([s[:, 0]]).T
    scaler = StandardScaler().fit(X)
    Xstar = scaler.transform(X)
    Y = np.array([s[:, 1]]).T
    rbf_sigma, rbf_lambda = 0.3, 1e-3
    pol_M, pol_lambda = 10, 1
    fitter1 = SVR(Polynomial(pol_M), pol_lambda, epsilon=0.6, small_eps=1e-6)
    svr1 = fitter1.fit(Xstar, Y, )
    fitter2 = SVR(RBF(rbf_sigma), rbf_lambda, epsilon=0.6, small_eps=1e-6)
    svr2 = fitter2.fit(Xstar, Y)
    vectors1 = fitter1.vectors
    vectors2 = fitter2.vectors
    xt = np.array([np.arange(0, 20, 0.001)]).T
    yprime2 = svr2.predict(scaler.transform(xt))
    yprime1 = svr1.predict(scaler.transform(xt))
    fig, ax = plt.subplots()
    sns.scatterplot(x=X.T[0], y=Y.T[0], label='Sine dataset', alpha=0.5, palette='blue', color=sns.color_palette()[1])
    rcParams["scatter.marker"] = "1"
    sns.scatterplot(x=X.T[0][vectors2], y=Y.T[0][vectors2], label='Support vectors RBF', color=sns.color_palette()[2],)
    rcParams["scatter.marker"] = "2"
    sns.scatterplot(x=X.T[0][vectors1], y=Y.T[0][vectors1], label='Support vectors Polynomial',color=sns.color_palette()[3], markers=X.T[0][vectors1])
    sns.lineplot(x=xt.T[0], y=yprime2, label=f'RBF', color=sns.color_palette()[2])
    sns.lineplot(x=xt.T[0], y=yprime1, label=f'Polynomial',color=sns.color_palette()[3])
    sns.despine()
    plt.legend(loc=3, bbox_to_anchor=(0, -0.7))
    ax.set_title('Support vector regression on sine dataset')
    plt.subplots_adjust(bottom=0.4)
    plt.savefig('Sine')
    plt.show()


if __name__ == '__main__':
    sine()
    housing = pd.read_csv('housing2r.csv')
    h = np.array(housing)
    X = np.array(h[:, :5])
    scaler = StandardScaler().fit(X)
    Xstar = scaler.transform(X)
    Y = np.array([h[:, 5]]).T
    rmse_M(Xstar, Y)