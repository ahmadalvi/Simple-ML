import unittest
from losses import losses


class TestLosses(unittest.TestCase):

    def setUp(self):
        self.y_true = [1, 0, 1, 1]
        self.y_pred = [0.9, 0.1, 0.8, 0.7]
        self.ce_y_true = [1, 0, 1, 0]
        self.ce_y_pred = [0.9, 0.2, 0.8, 0.4]

    def test_mse(self):
        result = losses.mse(self.y_true, self.y_pred)
        expected = (
            (1 - 0.9) ** 2 + (0 - 0.1) ** 2 + (1 - 0.8) ** 2 + (1 - 0.7) ** 2
        ) / 4
        self.assertAlmostEqual(result, expected)

    def test_mae(self):
        result = losses.mae(self.y_true, self.y_pred)
        expected = (abs(1 - 0.9) + abs(0 - 0.1) + abs(1 - 0.8) + abs(1 - 0.7)) / 4
        self.assertAlmostEqual(result, expected)

    def test_cross_entropy_binary(self):
        result = losses.cross_entropy(self.ce_y_true, self.ce_y_pred, type="binary")
        import math

        expected = (
            -(
                (1 * math.log(0.9) + (1 - 1) * math.log(1 - 0.9))
                + (0 * math.log(0.2) + (1 - 0) * math.log(1 - 0.2))
                + (1 * math.log(0.8) + (1 - 1) * math.log(1 - 0.8))
                + (0 * math.log(0.4) + (1 - 0) * math.log(1 - 0.4))
            )
            / 4
        )
        self.assertAlmostEqual(result, expected)

    def test_cross_entropy_categorical(self):
        y_true_cat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        y_pred_cat = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        result = losses.cross_entropy(y_true_cat, y_pred_cat, type="categorical")
        import math

        expected = (
            -(
                (1 * math.log(0.9) + 0 * math.log(0.05) + 0 * math.log(0.05))
                + (0 * math.log(0.1) + 1 * math.log(0.8) + 0 * math.log(0.1))
                + (0 * math.log(0.2) + 0 * math.log(0.2) + 1 * math.log(0.6))
            )
            / 3
        )
        self.assertAlmostEqual(result, expected)
