from math import sqrt

def qr_decomposition(matrix):

    q = [[] for i in range(len(matrix))]
    r = [[0]*len(matrix) for i in range(len(matrix))]

    for i in range(len(matrix)):
        vec = []
        norm = 0
        for j in range(len(matrix)):
            value = matrix[j][i]
            for k in range(i):
                value -= r[k][i] * q[j][k]
            vec.append(value)
            norm += value ** 2

        norm = sqrt(norm)

        for l in range(len(vec)):
            q[l].append( vec[l] / norm )
            for m in range(i, len(matrix)):
                r[i][m] += q[l][i]*matrix[l][m]

    return (q, r)

#O(N³)
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

#O(N³)
def determinant(matrix):

    # det(A) = det(Q.R) = det(L.U.R) = det(L).det(U).det(R) = PROD_i{ L_{ii} }.PROD_i{ U_{ii} }.PROD_i{ R_{ii} }
    #Since det(L) = PROD_i{ L_{ii} } = 1, det(A) = PROD_i{ U_{ii}.R_{ii} }

    q, r = qr_decomposition(matrix)
    l, u = lu_decomposition(q)

    det = 1
    for i in range(len(matrix)):
        det *= u[i][i] * r[i][i]
    return det

matrix = [
    [1, 2, 3],
    [3, 2, 1],
    [2, 1, 3]
]

print("Determinant: %.2f"%determinant(matrix))
