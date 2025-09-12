import unittest
from core.matrix import Matrix
from core.vector import Vector
from core.exceptions import DimensionError


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.m1 = Matrix([[1, 2], [3, 4]])
        self.m2 = Matrix([[5, 6], [7, 8]])
        self.v = Vector([1, 2])

    # ---- trace() ----
    def test_trace_square_matrix(self):
        self.assertEqual(self.m1.trace(), 5)  # 1 + 4

    def test_trace_non_square_raises(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionError):
            m.trace()

    # ---- transpose() ----
    def test_transpose(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.transpose(), Matrix([[1, 3], [2, 4]]))

    # ---- add() ----
    def test_add_valid(self):
        self.assertEqual(self.m1.add(self.m2), Matrix([[6, 8], [10, 12]]))

    def test_add_dimension_mismatch(self):
        m3 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionError):
            self.m1.add(m3)

    # ---- sub() ----
    def test_sub_valid(self):
        self.assertEqual(self.m2.sub(self.m1), Matrix([[4, 4], [4, 4]]))

    def test_sub_dimension_mismatch(self):
        m3 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            self.m1.sub(m3)

    # ---- scalar_mult() ----
    def test_scalar_mult(self):
        self.assertEqual(self.m1.scalar_mult(2), Matrix([[2, 4], [6, 8]]))

    def test_scalar_mult_zero(self):
        self.assertEqual(self.m1.scalar_mult(0), Matrix([[0, 0], [0, 0]]))

    # ---- vector_mult() ----
    def test_vector_mult_valid(self):
        # [ [1,2], [3,4] ] * [1,2] = [1*1+2*2, 3*1+4*2] = [5, 11]
        self.assertEqual(self.m1.vector_mult(self.v), Vector([5, 11]))

    def test_vector_mult_dimension_mismatch(self):
        v = Vector([1, 2, 3])
        with self.assertRaises(DimensionError):
            self.m1.vector_mult(v)

    # ---- matrix_mult() ----
    def test_matrix_mult_valid(self):
        # [ [1,2], [3,4] ] * [ [5,6], [7,8] ]
        # = [ [1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8] ]
        # = [ [19, 22], [43, 50] ]
        self.assertEqual(self.m1.matrix_mult(self.m2), Matrix([[19, 22], [43, 50]]))
