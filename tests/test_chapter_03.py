import unittest

import numpy as np

from chapter_03.bai_3_26_3_30 import (
    Perceptron,
    gradient_descent,
    perceptron_score,
    perceptron_update,
    sign_label,
)


class Chapter03Tests(unittest.TestCase):
    def test_gradient_descent_four_updates(self) -> None:
        history = gradient_descent()
        expected_x = [5.0, 3.8, 3.08, 2.648, 2.3888]
        expected_f = [10.0, 4.24, 2.1664, 1.419904, 1.15116544]
        np.testing.assert_allclose([step.x for step in history], expected_x)
        np.testing.assert_allclose([step.value for step in history], expected_f)

    def test_exercise_3_27(self) -> None:
        score = perceptron_score([1, 2, -10], [3, 4, 1])
        self.assertEqual(score, 1)
        self.assertEqual(sign_label(score), 1)
        self.assertNotEqual(sign_label(score), -1)

    def test_exercise_3_28(self) -> None:
        updated = perceptron_update([-2, 1, 0], [2, 3, 1], label=1)
        np.testing.assert_array_equal(updated, [0, 4, 1])
        self.assertEqual(perceptron_score(updated, [2, 3, 1]), 13)

    def test_custom_perceptron_learns_and(self) -> None:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, -1, -1, 1])
        model = Perceptron(learning_rate=0.1, n_epochs=100).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)


if __name__ == "__main__":
    unittest.main()

