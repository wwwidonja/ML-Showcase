import numpy as np
import pandas as pd


class classifier():
    def __init__(self):
        pass

    def predict(self):
        pass


class regressor():
    def __init__(self):
        pass

    def predict(self):
        pass


class ANNClassification:
    def __init__(self, X, Y, h1_dim=10, l2_dim=2):
        ###Initiate A0 - input layer
        ###Initate Z1 - unactivated 1st layer - weight.dot(A0) + bias
        ##Apply activation ie softmax A1 = relu(Z1)
        ### Z2 = W2A1 + b2
        ### A2 = softmax(z2)

        self.m = Y.size
        self.w1 = np.random.rand(h1_dim, X.shape[0]) - 0.5 #check if values ok
        self.bias_1 = np.random.rand(h1_dim, 1) - 0.5 #check if values ok
        self.w2 = np.random.rand(l2_dim, h1_dim) - 0.5 #check if values ok
        self.bias_2 = np.random.rand(l2_dim, 1) - 0.5 #check if values ok
        self.X = X
        self.dictionary = {i[1] : i[0] for i in enumerate(np.unique(Y))}
        self.encoded_Y = self.onehot(np.array([self.dictionary[i] for i in Y]))
        pass


    def onehot(self, Y):
        l = len(np.unique(Y))
        oh = np.zeros((len(Y), l))
        print(Y)
        oh[np.arange(Y.size), Y] = 1
        return oh.T


    def ReLU(self, Z):
        return np.maximum(0, Z)


    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))


    def forward_prop(self):
        Z1 = self.w1.dot(self.X) + self.bias_1
        A1 = self.ReLU(Z1)
        Z2 = self.w2.dot(A1) + self.bias_2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2


    def deriv_relu(self, Z):
        return Z>0

    def backprop(self, Z1, A1, Z2, A2):
        dZ2 = A2 - self.encodedY
        dW2 = 1/self.m * dZ2.dot(A1.T)
        db2 = 1/self.m * np.sum(dZ2, 2)
        dZ1 = self.w2.T.dot(dZ2) * self.deriv_relu(Z1)

        dW1 = 1/self.m * dZ1.dot(self.X.T)
        db1 = 1/self.m * np.sum(dZ1,2)

        return dW1, db1, dW2, db2


    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1-alpha*dW1
        b1 = b1-alpha*db1
        W2 = W2-alpha*dW2
        b2 = b2 - alpha*db2
        return W1, b1, W2, b2

    def fit(self):
        pass


class ANNRegression:
    def __init__(self):
        pass

    def fit(self):
        pass


if __name__ == "__main__":
    df = pd.read_csv('./housing3.csv')
    ###Take just a couple of examples
    df = df[:3]
    data = np.array(df).T
    X = data[:-1]
    Y = data[-1]
    print(Y.shape)
    c = ANNClassification(X, Y)