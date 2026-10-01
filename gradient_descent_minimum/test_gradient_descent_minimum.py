import unittest

from gradient_descent_minimum import df1, df2, f1, f2, gradient_descent


class GradientDescentMinimumTests(unittest.TestCase):
    def test_first_function_converges_to_global_minimum(self) -> None:
        result = gradient_descent(f1, df1, x0=5, learning_rate=0.1)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.x, 0.0, delta=5e-4)
        self.assertAlmostEqual(result.value, -2.0, places=6)

    def test_second_function_converges_from_x0_2(self) -> None:
        result = gradient_descent(f2, df2, x0=2, learning_rate=0.1)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.x, 1.0, delta=5e-4)
        self.assertAlmostEqual(result.value, -2 / 3, places=6)

    def test_second_function_converges_from_x0_5(self) -> None:
        result = gradient_descent(f2, df2, x0=5, learning_rate=0.1)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.x, 1.0, delta=5e-4)

    def test_rejects_invalid_learning_rate(self) -> None:
        with self.assertRaises(ValueError):
            gradient_descent(f1, df1, x0=5, learning_rate=0)


if __name__ == "__main__":
    unittest.main()
