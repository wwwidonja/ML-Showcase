import unittest

import numpy as np
import random

from hw_tree import Tree, Bagging, RandomForest, \
    hw_tree_full, hw_cv_min_samples, hw_bagging, hw_randomforests


def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]


class HWTreeTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        self.y = np.array([0, 0, 1, 1])
        self.train = self.X[:3], self.y[:3]
        print(self.train)
        self.test = self.X[3:], self.y[3:]

    def test_call_tree(self):
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=random_feature,
                 min_samples=2)
        p = t.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_bagging(self):
        t = Tree(rand=random.Random(1),
             get_candidate_columns=random_feature,
             min_samples=2)
        b = Bagging(rand=random.Random(0),
                    tree_builder=t,
                    n=20)
        p = b.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_randomforest(self):
        rf = RandomForest(rand=random.Random(0),
                          n=20,
                          min_samples=2)
        p = rf.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_signature_hw_tree_full(self):
        train, test = hw_tree_full(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)

    def test_signature_hw_bagging(self):
        train, test = hw_bagging(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)

    def test_signature_hw_randomforests(self):
        train, test = hw_randomforests(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)

    def test_signature_hw_cv_min_samples(self):
        train, test, best_min_samples = hw_cv_min_samples(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(best_min_samples, int)


if __name__ == "__main__":
    unittest.main()
