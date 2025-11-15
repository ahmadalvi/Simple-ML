from core.exceptions import DimensionError
from core.vector import Vector


class Matrix:
    def __init__(self, mat):
        self.mat = mat
        self.rows = len(mat)
        self.columns = len(mat[0]) if mat else 0

    def __eq__(self, value):
        if not isinstance(value, Matrix):
            return False
        return self.mat == value.mat
    
    def disp(self):
        rows = int(len(self.mat) / self.columns)
        for i in range(rows):
            row = self.mat[i * self.columns : (i + 1) * self.columns]
            print(" ".join(f"{x:4}" for x in row))

    def trace(self) -> int:
        """Calculates the trace of a matrix

        Args:
            m: A Matrix object

        Returns:
            Returns an integer object, the sum of the diagonal elements of the matrix

        Raises:
            DimensionError: The matrix provided must be a square matrix

        """
        if self.rows != self.columns:
            raise DimensionError("Matrix must be square to calculate trace.")

        return sum(self.mat[i][i] for i in range(self.rows))
    
    def rank(self, tol=1e-9) -> int:
        """Calculate the rank of a matrix.

        Args:
            self: A Matrix object

        Returns:
            Returns the rank of the matrix as an integer
        """
        if self.is_diagonal():
            return self.rows
        
        matrix = self.mat.copy()
        rows = len(self.mat)
        cols = len(self.mat[0])
        rank = 0
        row = 0

        for col in range(cols):
            pivot = None
            max_val = tol
            for r in range(row, rows):
                if abs(matrix[r][col]) > max_val:
                    max_val = abs(self.mat[r][col])
                    pivot = r
            
            if pivot is None:
                continue

            matrix[row], matrix[pivot] = matrix[pivot], matrix[row]

            pivot_val = matrix[row][col]
            matrix[row] = [x / pivot_val for x in matrix[row]]

            for r in range(row + 1, rows):
                factor = matrix[r][col]
                matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[row])]
            
            row += 1 
            rank += 1
        return rank
    
    def transpose(self) -> "Matrix":
        """Transpose a matrix

        Args:
            m: A Matrix object

        Returns:
            Returns the transpose of a matrix
        """

        transposed = [
            [self.mat[j][i] for j in range(self.rows)] for i in range(self.columns)
        ]
        return Matrix(transposed)

    def add(self, other) -> "Matrix":
        """Add 2 matrices together

        Args:
            self: A Matrix object
            other: A Matrix object

        Returns:
            Returns the sum of the 2 matrices as a new matrix

        Raises:
            DimensionError: If the vecotrs are not the same size.

        """
        if (self.rows != other.rows) or (self.columns != other.columns):
            raise DimensionError("Matrices must have the same dimension")

        matrix = self.mat.copy()

        for i in range(len(self.mat)):
            for j in range(len(self.mat[i])):
                matrix[i][j] += other.mat[i][j]

        return Matrix(matrix)

    def sub(self, other) -> "Matrix":
        """Subtract 2 matrices

        Args:
            m1: A Matrix object
            m2: A Matrix object

        Returns:
            Returns the difference of the 2 matrices as a new matrix

        Raises:
            DimensionError: If the vecotrs are not the same size.

        """
        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Matrices must have the same dimension")

        matrix = self.mat.copy()

        for i in range(len(self.mat)):
            for j in range(len(self.mat[i])):
                matrix[i][j] -= other.mat[i][j]

        return Matrix(matrix)

    def scalar_mult(self, scalar: int) -> "Matrix":
        """Multiply a matrix with a scalar

        Args:
            self: a Matrix object
            scalar: and integer

        Returns:
            Returns a new Matrix object with each element multiplied by the scalar
        """

        return Matrix(
            [
                [(self.mat[i][j] * scalar) for j in range(self.columns)]
                for i in range(self.rows)
            ]
        )

    def vector_mult(self, vector: Vector):
        """Multiply a matrix with a vector

        Args:
            self: a Matrix object
            vector: a Vector object

        Returns:
            Returns a new Vector object with the result of the multiplication
        """

        if self.columns != vector.n:
            raise DimensionError(
                "Matrix columns must match vector size for multiplication."
            )

        result = [0] * self.rows

        for i in range(self.rows):
            for j in range(self.columns):
                result[i] += self.mat[i][j] * vector.arr[j]

        return Vector(result)

    def matrix_mult(self, other) -> "Matrix":
        """Multiply 2 matrices together

        Args:
            self: A Matrix object
            other: A Matrix object

        Returns:
            Returns the product of the 2 matrices as a new matrix

        Raises:
            DimensionError: If the matrices are not compatible for multiplication.

        """
        if self.columns != other.rows:
            raise DimensionError("Matrices are not compatible for multiplication.")

        result = [[0 for _ in range(other.columns)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.columns):
                for k in range(self.columns):
                    result[i][j] += self.mat[i][k] * other.mat[k][j]

        return Matrix(result)

    def is_diagonal(self) -> bool: 
        """Check if a matrix is diagonal
        Args:
            self: A Matrix object

        Returns:
            Returns True if the matrix is diagonal, False otherwise

        """
        for i in range(self.rows):
            for j in range(self.columns):
                if i != j and self.mat[i][j] != 0:
                    return False
        return True
    
    def is_upper_triangular(self) -> bool:
        """Check if a matrix is upper triangular

        Args:
            self: A Matrix object

        Returns:
            Returns True if the matrix is diagonal, False otherwise
        """
        for i in range(1, self.rows):
            for j in range(min(i, self.columns)):
                if self.mat[i][j] != 0:
                    return False
        return True
    
    def lu_decomposition(self) -> tuple["Matrix", "Matrix"]:
        """Perform LU Decomposition on a matrix
        
        Args:
            self: A matrix object
        
        Returns:
            Returns a tuple containing the lower and uppoer triangular matrices aa Matrix Objects
            
        Raises:
            DimensionError: If the matrix is not square.
        """
    
        if self.rows != self.columns:
            raise DimensionError("Matrix must be square for LU Decomposition.")

        L = identity(self.rows).mat
        U = self.mat.copy()
        n = self.rows

        for i in range(n):
            for j in range(i+1, n):
                m = -1 * (U[j][i] / U[i][i])
                U[j][i] = 0
                for k in range(i+1, n):
                    U[j][k] += m * U[i][k]
                L[j][i] = -1 * m
        
        return (Matrix(L), Matrix(U))

    def forward_substitution(self, b: Vector) -> Vector:
        """Perform Forward Substitution on a matrix, given a vector b
        
        Args:
            self: A matrix object
            b: A Vector object
        
        Returns:
            Returns a Vector object y such that Ly = b
            
        Raises:
            DimensionError: If the matrix is not square.
        """
        y = b.arr.copy()

        for i in range(self.rows):
            for j in range(i):
                y[i] = round(y[i] - self.mat[i][j] * y[j], 6)
        
        return Vector(y)

    def backward_substitution(self, y: Vector) -> Vector:
        """Perform Backward Substitution on a matrix, given a vector y
        
        Args:
            self: A matrix object
            y: A Vector object
        
        Returns:
            Returns a Vector object x such that Ux = y
            
        Raises:
            DimensionError: If the matrix is not square.
        """
        x = y.arr.copy()

        for i in range(self.rows - 1, -1, -1):
            for j in range(i+1, self.columns):
                x[i] -= self.mat[i][j] * x[j]
            x[i] = round(x[i] / self.mat[i][i], 6)
        
        return Vector(x)

    def inverse(self) -> "Matrix":
        """Calculate the inverse of a matrix

        Args:
            self: A Matrix object

        Returns:
            Returns the inverse of the matrix as a new Matrix object

        Raises:
            DimensionError: If the matrix is not square or not invertible.

        """
        if self.rows != self.columns:
            raise DimensionError("Matrix must be square to calculate inverse.")

        if self.rank() < self.rows:
            raise DimensionError("Matrix is not invertible, rank of matrix is less than its size.")
    
        L, U = self.lu_decomposition()
        n = self.rows

        inverse_mat = [[0.0 for _ in range(n)] for _ in range(n)]

        for k in range(n):
            e_k = Vector([1 if i == k else 0 for i in range(n)])

            y = L.forward_substitution(e_k)

            x = U.backward_substitution(y)

            for i in range(n):
                inverse_mat[i][k] = x.arr[i]

        return Matrix(inverse_mat)


    def determinant(self) -> int:
        """Calculate the determinant of a matrix
        Args:
            self: A Matrix object
        Returns:
            Returns the determinant as an integer
        Raises:
            DimensionError: If the matrix is not square.
        """

        if self.rows != self.columns:
            raise DimensionError("Matrix must be square to calculate determinant.")

        if self.rows == 2:
            return self.mat[0][0] * self.mat[1][1] - self.mat[0][1] * self.mat[1][0]

        if self.is_upper_triangular():
            det = 1
            for i in range(self.rows):
                det *= self.mat[i][i]
            return det

        det = 0
        for j in range(self.columns):
            sub_matrix = [
                [self.mat[i][k] for k in range(self.columns) if k != j]
                for i in range(1, self.rows)
            ]
            sign = (-1) ** j
            det += sign * self.mat[0][j] * Matrix(sub_matrix).determinant()

        return det

def identity(n: int) -> Matrix:
    """Create an nxn identity matrix

    Args:
        n: A positive integer

    Returns:
        Returns a Matrix object

    """

    return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])


def zeros(n: int) -> Matrix:
    """Creates a matrix full of zeroes

    Args:
        n: A positive integer

    Returns:
        Returns a Matrix Object

    """

    return Matrix([[0 for _ in range(n)] for _ in range(n)])


def ones(n: int) -> Matrix:
    """Creates a matrix full of ones

    Args:
        n: A positive integer

    Returns:
        Returns a Matrix Object

    """
    return Matrix([[1 for _ in range(n)] for _ in range(n)])
