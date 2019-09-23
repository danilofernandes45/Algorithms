from math import sqrt

def cholesky_decomp(matrix):

    n = len(matrix)
    tri = [[0]*n for i in range(n)]

    for i in range(n):
        for j in range(i, n):
            if(i == j):
                tri[i][i] = matrix[i][i]
                for k in range(i):
                    tri[i][i] -= tri[k][i] ** 2
                tri[i][i] = sqrt(tri[i][i])

            else:
                tri[i][j] = matrix[i][j]
                for k in range(i):
                    tri[i][j] -= tri[k][i]*tri[k][j]
                tri[i][j] /= tri[i][i]

    return tri

#É assumido que a matriz é positiva e semidefinida

matrix = [[1,1,1],[1,2,2],[1,2,2]]
print(cholesky_decomp(matrix))
