#Hardcoding linear algebra since I can't download anaconda on the bus wifi :(

import numpy as np

########## VECTOR STUFF ##########
#vector class: takes array as vals and crystalizes it
#.space property tells what R^n it's in
class Vector: 
    def __init__(self, vals):
        self.vals = vals
        self.space = len(vals)

def Vector_Haddamard_Product (vec1, vec2):
    space = vec1.space
    if space != vec2.space:
        return("domain error")
    else:
        array = [None] * space
        for i in range(space):
            array[i] = vec1.vals[i] * vec2.vals[i]
        return(Vector(array))
    
def Dot_Product (vec1,vec2):
    space = vec1.space
    if space != vec2.space:
        return("error")
    else:
        sum = 0
        i = 0
        while i < space:
            sum += vec1.vals[i] * vec2.vals[i]
            i +=1
        return(sum)


########## MATRIX STUFF ##########


#matrix class: takes array of arrays as vals and crystalizes it
#.space property holds tuple (m,n) for M in R^m x R^n
#.vals holds the array of arrays
#.rows holds m
#.columns holds n
#.Transpose() returns the transpose
#.PrintMatrix() prints the matrix line by line
#.getVal(i,j) returns the val in the ith row, jth column
#.setVal(i,j,val) sets the entry in the ith row, jth column to val
class Matrix:
    def __init__(self, vals):
        self.vals = vals
        self.rows = len(vals)
        self.columns = len(vals[0])
        self.space = (self.rows,self.columns)

    def Transpose(self):
        empty_matrix = [[0] * self.rows for _ in range(self.columns)]
        for i in range(self.rows):
            for j in range (self.columns):
                empty_matrix[i][j] += self.vals[j][i]
        return(Matrix(empty_matrix))
    
    def PrintMatrix(self):
        for i in range(self.rows):
            print(self.vals[i])

    def getVal(self, i, j):
        return(self.vals[i-1][j-1])
    
    def setVal(self, i, j, value):
        self.vals[i-1][j-1] = value

    def Determinant(self):
        rows = self.rows
        if rows != self.columns:
            return("Need a Square matrix")
        else:
            det = 0
            for j in range(rows):
                if rows == 2:
                    return(self.getVal(1,1)*self.getVal(2,2))-(self.getVal(1,2)*self.getVal(2,1))
                else:
                    new_vals = [row[:] for row in self.vals]
                    del new_vals[0]
                    for i in range(rows-1):
                        del new_vals[i][j]
                    new_matrix = Matrix(new_vals)
                    det += ((-1)**(j+2)) * (self.getVal(1,j+1)) * new_matrix.Determinant()
            return(det)



# haddamard product for matrices
def Haddamard_Product (matrix1, matrix2):
    space = matrix1.space
    if space != matrix2.space:
        return("domain error")
    else:
        empty_matrix = [[None] * space[1] for _ in range(space[0])]
        for i in range(space[0]):
            for j in range(space[1]):
                empty_matrix[i-1][j-1] = matrix1.getVal(i,j) * matrix2.getVal(i,j)
        return(Matrix(empty_matrix))

# add two matrices
def Matrix_Sum (matrix1, matrix2):
    space = matrix1.space
    if space != matrix2.space:
        return("domain error")
    else:
        empty_matrix = [[None] * space[1] for _ in range(space[0])]
        for i in range(space[0]):
            for j in range(space[1]):
                empty_matrix[i-1][j-1] = matrix1.getVal(i,j) + matrix2.getVal(i,j)
        return(Matrix(empty_matrix))

# standard matrix product
def Matrix_Product (matrix1, matrix2):
    length = matrix1.columns
    if length != matrix2.rows:
        return("domain error")
    else:
        empty_matrix = [[0] * matrix2.columns for _ in range(matrix1.rows)]
        for i in range(matrix1.rows):
            for j in range(matrix2.columns):
                for n in range(length):
                    empty_matrix[i-1][j-1] += matrix1.getVal(i,n) * matrix2.getVal(n,j)
        return(Matrix(empty_matrix))
 

print("Hello World")
print()



"""m = Matrix([[-100,2,3,6,6],[2,2,3,6,2],[1,1,5,1,7],[1,1,1,1,0],[1,3,3,4,1]])
m.PrintMatrix()
print()
print(m.Determinant())"""

"""vec1 = Vector([1,9])
vec2 = Vector([2,1])

vec3 = Haddamard_Product(vec1,vec2)

print(vec3.vals)
print(Dot_Product(vec1,vec2))
"""

"""matrix2 = Matrix([[1,2],[99,1]])
matrix1 = Matrix([[1,0],[0,1],[1,1]])
print("Matrix 1 =")
matrix1.PrintMatrix()
print("Matrix 2 =")
matrix2.PrintMatrix()
print("Matrix 1 x Matrix 2 =")
matrix3 = Matrix_Product(matrix1,matrix2)
matrix3.PrintMatrix()
matrix3.setVal(1,1,69)
print("Matrix 3 =")
matrix3.PrintMatrix()"""

"""m = Matrix([[1,1],[2,2]])
n = Matrix([[4,4],[1,1]])
p = Haddamard_Product(m,n)
print("p = ")
p.PrintMatrix()
q = Matrix_Sum(m,n)
print("q = ")
q.PrintMatrix()"""