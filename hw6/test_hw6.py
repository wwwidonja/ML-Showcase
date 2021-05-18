import unittest
import numpy as np
import csv

from hw6 import ANNClassification, ANNRegression


class HW6Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 1, 2, 3])
        self.hard_y = np.array([0, 1, 1, 0])

    def test_ann_classification_no_hidden_layer(self):
        fitter = ANNClassification(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 4))
        np.testing.assert_allclose(pred, np.identity(4), atol=0.01)

    def test_ann_classification_no_hidden_layer_hard(self):
        # aiming to solve a non linear problem without hidden layers
        fitter = ANNClassification(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, 0.5, atol=0.01)

    def test_ann_classification_hidden_layer_hard(self):
        # with hidden layers we can solve an non-linear problem
        fitter = ANNClassification(units=[10], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, [[1, 0], [0, 1], [0, 1], [1, 0]], atol=0.01)

    def test_ann_classification_hidden_layers_hard(self):
        # two hidden layers
        fitter = ANNClassification(units=[10, 20], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, [[1, 0], [0, 1], [0, 1], [1, 0]], atol=0.01)

    def test_ann_regression_no_hidden_layer(self):
        fitter = ANNRegression(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.y, atol=0.01)

    def test_ann_regression_no_hidden_layer_hard(self):
        # aiming to solve a non linear problem without hidden layers
        fitter = ANNRegression(units=[], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, 0.5, atol=0.01)

    def test_ann_regression_hidden_layer_hard(self):
        # one hidden layer
        fitter = ANNRegression(units=[10], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.hard_y, atol=0.01)

    def test_ann_regression_hidden_layer_hard(self):
        # two hidden layers
        fitter = ANNRegression(units=[13, 6], lambda_=0.0001)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.hard_y, atol=0.01)

    def test_predictor_get_info(self):
        fitter = ANNRegression(units=[10, 5], lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        lw = m.weights()  # a list of weight matrices that include intercept biases

        self.assertEqual(len(lw), 3)  # two hidden layer == three weight matrices

        self.assertEqual(lw[0].shape, (3, 10))
        self.assertEqual(lw[1].shape, (11, 5))
        self.assertEqual(lw[2].shape, (6, 1))


class TestFinalSubmissions(unittest.TestCase):

    def test_format(self):
        """ Tests format of your final predictions. """

        with open("final.txt", "rt") as f:
            content = list(csv.reader(f))
            content = [l for l in content if l]

            ids = []

            for i, l in enumerate(content):
                # each line contains 10 columns
                self.assertEqual(len(l), 10)

                # first line is just a description line
                if i == 0:
                    self.assertEqual(l, ['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4',
                                         'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
                else:
                    ids.append(int(l[0]))  # first element is an id
                    probs = np.array([float(f) for f in l[1:]])  # the rest are probabilities
                    self.assertLessEqual(np.max(probs), 1)
                    self.assertGreaterEqual(np.min(probs), 0)

            # ids covered the whole range
            self.assertEqual(set(ids), set(range(1, 11878+1)))

    def test_function(self):
        try:
            from hw6 import create_final_predictions
        except ImportError:
            self.fail("Function create_final_predictions does not exists.")


if __name__ == "__main__":
    import unittest
    unittest.main()