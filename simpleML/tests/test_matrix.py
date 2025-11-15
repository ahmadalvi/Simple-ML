import unittest
from core.matrix import Matrix
from core.vector import Vector
from core.exceptions import DimensionError


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.m1 = Matrix([[1, 2], [3, 4]])
        self.m2 = Matrix([[5, 6], [7, 8]])
        self.m3 = Matrix([[1, 2], [0, 0]])
        self.m4 = Matrix([[0, 0], [0, 0]])
        self.v = Vector([1, 2])

    # ---- trace() ----
    def test_trace_square_matrix(self):
        self.assertEqual(self.m1.trace(), 5)  # 1 + 4

    def test_trace_non_square_raises(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionError):
            m.trace()

    # ---- rank() ----
    def test_rank(self):
        self.assertEqual(self.m1.rank(), 2)
        self.assertEqual(self.m3.rank(), 1)
        self.assertEqual(self.m4.rank(), 0)

    # ---- transpose() ----
    def test_transpose(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.transpose().mat, Matrix([[1, 3], [2, 4]]).mat)

    # ---- add() ----
    def test_add_valid(self):
        self.assertEqual(self.m1.add(self.m2).mat, Matrix([[6, 8], [10, 12]]).mat)

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
        self.assertEqual(self.m1.matrix_mult(self.m2), Matrix([[19, 22], [43, 50]]))

    # ---- lu_decomposition() ----
    def test_lu_decomposition_2x2(self):
        A = Matrix([[4, 3], [6, 3]])
        L_expected = Matrix([[1, 0], [1.5, 1]])
        U_expected = Matrix([[4, 3], [0, -1.5]])

        L, U = A.lu_decomposition()
        self.assertEqual(L, L_expected)
        self.assertEqual(U, U_expected)

    def test_lu_decomposition_identity(self):
        A = Matrix([[1, 0], [0, 1]])
        L, U = A.lu_decomposition()
        self.assertEqual(L, Matrix([[1, 0], [0, 1]]))
        self.assertEqual(U, Matrix([[1, 0], [0, 1]]))

    def test_lu_decomposition_non_square_raises(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionError):
            A.lu_decomposition()

    # ---- forward_substitution() ----
    def test_forward_substitution_basic(self):
        L = Matrix([[1, 0, 0], [2, 1, 0], [-1, 4, 1]])
        b = Vector([1, 2, 3])

        y_expected = Vector([1, 0, 4])
        y = L.forward_substitution(b)
        self.assertEqual(y, y_expected)

    def test_forward_substitution_identity(self):
        L = Matrix([[1, 0], [0, 1]])
        b = Vector([7, 9])
        self.assertEqual(L.forward_substitution(b), b)

    # ---- backward_substitution() ----
    def test_backward_substitution_basic(self):
        U = Matrix([[2, 3, -1], [0, 4, 2], [0, 0, 5]])
        y = Vector([3, 6, 10])

        x_expected = Vector([1.75, 0.5, 2])
        x = U.backward_substitution(y)
        self.assertEqual(x, x_expected)

    def test_backward_substitution_identity(self):
        U = Matrix([[1, 0], [0, 1]])
        y = Vector([5, -3])
        self.assertEqual(U.backward_substitution(y), y)

    # ---- inverse() ----
    def test_inverse_2x2(self):
        A = Matrix([[4, 7], [2, 6]])
        A_inv_expected = Matrix([[0.6, -0.7], [-0.2, 0.4]])
        self.assertEqual(A.inverse().mat, A_inv_expected.mat)

    def test_inverse_3x3(self):
        A = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        A_inv_expected = Matrix([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]])
        self.assertEqual(A.inverse().mat, A_inv_expected.mat)

    def test_inverse_identity(self):
        I = Matrix([[1, 0], [0, 1]])
        self.assertEqual(I.inverse().mat, I.mat)

    def test_inverse_non_invertible_raises(self):
        A = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(DimensionError):
            A.inverse()
