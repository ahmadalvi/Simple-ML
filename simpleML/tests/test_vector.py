import unittest
from core.vector import Vector
from core.exceptions import DimensionError

class TestVector(unittest.TestCase):

    def test_dot(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        self.assertEqual(v1.dot(v2), 32)

        v3 = Vector([1, 2])
        v4 = Vector([3, 4])
        self.assertEqual(v3.dot(v4), 11)

        v5 = Vector([])
        v6 = Vector([])
        self.assertEqual(v5.dot(v6), 0)

    def test_addition(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        self.assertEqual(v1 + v2, Vector([5, 7, 9]))

        v3 = Vector([])
        v4 = Vector([])
        self.assertEqual(v3 + v4, Vector([]))

    def test_subtract_positive_vectors(self):
        v1 = Vector([5, 7, 9])
        v2 = Vector([1, 2, 3])
        self.assertEqual(v1 - v2, Vector([4, 5, 6]))

    def test_subtract_with_negatives(self):
        v1 = Vector([3, -2, 7])
        v2 = Vector([-1, 4, -3])
        self.assertEqual(v1 - v2, Vector([4, -6, 10]))

    def test_subtract_empty_vectors(self):
        v1 = Vector([])
        v2 = Vector([])
        self.assertEqual(v1 - v2, Vector([]))

    def test_subtract_self(self):
        v = Vector([1, 2, 3])
        self.assertEqual(v - v, Vector([0, 0, 0]))

    def test_subtract_raises_on_dimension_mismatch(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        with self.assertRaises(DimensionError):
            _ = v1 - v2

    def test_length_mismatch_add(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4, 5])
        with self.assertRaises(DimensionError):
            _ = v1 + v2 

    def test_length_mismatch_dot(self):
        v1 = Vector([1])
        v2 = Vector([1, 2])
        with self.assertRaises(DimensionError):
            _ = v1.dot(v2)

    def test_scalarmult(self):
        v1 = Vector([1, 2, 3])
        scalar = 2
        self.assertEqual(v1.scalarmult(scalar), Vector([2, 4, 6]))

        v2 = Vector([])
        self.assertEqual(v2.scalarmult(scalar), Vector([]))

        self.assertEqual(v1.scalarmult(0), Vector([0,0,0]))

    def test_norm(self):
        v1 = Vector([3, 4])
        self.assertEqual(v1.norm("euclidean"), 5.0)
        self.assertEqual(v1.norm("manhattan"), 7.0)
        self.assertEqual(v1.norm("inf"), 4)

        v2 = Vector([])
        self.assertEqual(v2.norm("euclidean"), 0.0)
        self.assertEqual(v2.norm("manhattan"), 0.0)
        self.assertEqual(v2.norm("inf"), 0)

    def test_cross_product(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        result = v1.cross(v2)
        self.assertEqual(result, Vector([-3, 6, -3]))

        v3 = Vector([0, 0, 0])
        self.assertEqual(v3.cross(v3), Vector([0, 0, 0]))

        v4 = Vector([1, 2])
        with self.assertRaises(DimensionError):
            _ = v1.cross(v4)

    def test_valid_projection(self):
        v1 = Vector([3, 4])
        v2 = Vector([1, 0])
        self.assertEqual(v1.proj(v2), Vector([3, 0]))

    def test_projection_onto_self(self):
        v = Vector([2, 5])
        self.assertEqual(v.proj(v), v)

    def test_projection_onto_zero_vector_raises(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([0, 0, 0])
        with self.assertRaises(DimensionError):
            v1.proj(v2)

    def test_projection_with_negative_components(self):
        v1 = Vector([-2, 3])
        v2 = Vector([1, -1])
        self.assertEqual(v1.proj(v2), Vector([-2.5, 2.5]))

    def test_projection_of_zero_vector(self):
        v1 = Vector([0, 0, 0])
        v2 = Vector([2, 2, 1])
        self.assertEqual(v1.proj(v2), Vector([0, 0, 0]))

if __name__ == '__main__':
    unittest.main()
