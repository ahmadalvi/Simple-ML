class Matrix:
    def __init__(self, n: int, mat):
        self.n = n
        self.mat = mat
    
    def disp(self):
        if len(self.mat) == 0:
            print([])
        elif len(self.mat) == 1:
            print(self.mat)
        else:
            for i in range(len(self.mat)):
                if i % self.n == 0:
                    print("[", end = " ")
                    print(self.mat[i], end = " ")
                elif i % self.n == self.n-1:
                    print(self.mat[i], end = " ")
                    print("]")
                else:
                    print(self.mat[i], end = " ")

    def identity(n):
        m = Matrix(n, [])
        length = n * n
        for i in range(length):
            if i % (n + 1) == 0:
                m.mat.append(1)
            else:
                m.mat.append(0)
        return m
        
    def zeros(n):
        m = Matrix(n, [])
        for i in range(n * n):
            m.mat.append(0)
        return m
    
    def transpose(matrix):
        new_n = len(matrix.mat) / matrix.n
        m = Matrix(new_n, [])

        # for i in range(len(matrix.mat)):

    def trace(matrix) -> int:
        diagonal_entries = len(matrix.mat) / matrix.n
        tr = 0
        for i in range(0, len(matrix.mat), matrix.n + 1):
            tr += matrix.mat[i]
        return tr
    
    def add(matrix1, matrix2):
        if (matrix1.n != matrix2.n) or (len(matrix1.mat) != len(matrix2.mat)):
            return "Matrices are not the same size"
        
        matrix = Matrix(matrix1.n, [])
        for i in range(len(matrix1.mat)):
            matrix.mat.append(matrix1.mat[i] + matrix2.mat[i])

        return matrix
    
    def sub(matrix1, matrix2):
        if (matrix1.n != matrix2.n) or (len(matrix1.mat) != len(matrix2.mat)):
            return "Matrices are not the same size"
        
        matrix = Matrix(matrix1.n, [])
        for i in range(len(matrix1.mat)):
            matrix.mat.append(matrix1.mat[i] - matrix2.mat[i])

        return matrix
    
    