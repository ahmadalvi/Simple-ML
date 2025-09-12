from core.exceptions import DimensionError
from core.vector import Vector


class Matrix:
    def __init__(self, mat: list[list[int]]):
        self.mat = mat
        self.columns = len(mat[0])
        self.rows = len(mat)
    
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

        pass

    def is_orthogonal(self) -> bool:
        """Check if a matrix is orthogonal (a square matrix whose columns (and rows)
            form an orthonormal set of vectors, meaning each column has a unit length
            and is orthogonal to all other columns)

        Args:
            self: A Matrix object

        Returns:
            True if the matrix is orthogonal, False otherwise

        """
        pass

    def is_singular(self) -> bool:
        """Check if a matrix is singular (i.e. it does not have an inverse)

        Args:
            self: A Matrix object

        Returns:
            True if the matrix is singular, False otherwise

        """
        pass

    def is_idempotent(self) -> bool:
        """Check if a matrix is idempotent (i.e. multiplying the matrix by itself
            yields the same matrix)

        Args:
            self: A Matrix object
        Returns:
            True if the matrix is idempotent, False otherwise
        """
        pass

    def is_involutary(self) -> bool:
        """Check if a matrix is involutary (i.e. its own inverse)
        Args:
            self: A Matrix object
        Returns:
            True if the matrix is involutary, False otherwise
        """
        pass

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

        det = 0
        for j in range(self.columns):
            sub_matrix = [
                [self.mat[i][k] for k in range(self.columns) if k != j]
                for i in range(1, self.rows)
            ]
            sign = (-1) ** j
            det += sign * self.mat[0][j] * Matrix(sub_matrix).determinant()

        return det

    def eigenvalues(self) -> list[float]:
        """Calculate the eigenvalues of a matrix
        Args:
            self: A Matrix object
        Returns:
            Returns a list of eigenvalues as floats
        Raises:
            DimensionError: If the matrix is not square.
        """
        pass

    def eigenvectors(self) -> list[Vector]:
        """Calculate the eigenvectors of a matrix
        Args:
            self: A Matrix object
        Returns:
            Returns a list of eigenvectors as Vector objects
        Raises:
            DimensionError: If the matrix is not square.
        """
        pass

    def rank(self) -> int:
        """Calculate the rank of a matrix
        Args:
            self: A Matrix object
        Returns:
            Returns the rank as an integer
        """

        pass

    def nullity(self) -> int:
        """Calculate the nullity of a matrix
        Args:
            self: A Matrix object
        Returns:
            Returns the rank as an integer
        """
        pass

    def conjugate_transpose(self) -> "Matrix":
        """Calculate the conjugate transpose of a matrix
        Args:
            self: A Matrix object
        Returns:
        """
        pass

    def is_square(self) -> bool:
        """Check if a matrix is square (i.e. has the same number of rows and columns)
        Args:
            self: A Matrix object
        Returns:
            True if the matrix is square, False otherwise
        """
        return self.rows == self.columns


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
