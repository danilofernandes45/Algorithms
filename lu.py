def lu_decomposition(matrix):

    l = [ [0] * len(matrix) for i in range(len(matrix)) ]
    u = [ [0] * len(matrix) for i in range(len(matrix)) ]

    for i in range(len(matrix)):
        for j in range(len(matrix)):

            if( j < i ):
                l[i][j] = matrix[i][j]
                for k in range(j):
                    l[i][j] -= l[i][k] * u[k][j]
                l[i][j] /= u[j][j]

            else:
                if(i == j):
                    l[i][j] = 1

                u[i][j] = matrix[i][j]
                for k in range(i):
                    u[i][j] -= l[i][k] * u[k][j]

    return(l, u)


matrix = [
    [1, 2, 3],
    [3, 2, 1],
    [2, 1, 3]
]

l, u = lu_decomposition(matrix)
print("L")
print(l)
print("U")
print(u)
