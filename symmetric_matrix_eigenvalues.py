from math import sqrt
#O(N²)
def max_non_diag_position(matrix):
    p = -1
    q = -1
    max = -1
    for j in range(1, len(matrix)):
        for i in range(j):
            if( abs(matrix[i][j]) > max ):
                max = abs(matrix[i][j])
                p = i
                q = j
    return (p, q)

def getIdent(N):
    matrix_U = []
    for i in range(N):
        vector = []
        for j in range(N):
            x = 0
            if(i == j):
                x = 1
            vector.append(x)
        matrix_U.append(vector)
    return matrix_U

def signal(x):
    if(x < 0):
        return -1
    return 1

def jacobi_method(matrix_A, max_iter, eps):

    N = len(matrix_A)
    p, q = max_non_diag_position(matrix_A)
    error = N * abs(matrix_A[p][q])
    iter = 0
    #
    matrix_U = getIdent(N)
    #
    while(iter < max_iter and error >= eps):
        #COMPUTE ROTATION
        phi = ( matrix_A[p][p] - matrix_A[q][q] ) / ( 2 * matrix_A[p][q] )
        t = 1 / ( phi + signal(phi)*sqrt( phi**2 + 1 ) )
        cos = 1 / sqrt(1 + t**2)
        sin = t * cos
        #UPDATE A
        aux_pp = matrix_A[p][p] * cos**2 + 2*matrix_A[p][q]*cos*sin + matrix_A[q][q] * sin**2
        aux_qq = matrix_A[p][p] * sin**2 - 2*matrix_A[p][q]*cos*sin + matrix_A[q][q] * cos**2
        #
        matrix_A[p][p] = aux_pp
        matrix_A[q][q] = aux_qq
        matrix_A[p][q] = 0
        matrix_A[q][p] = 0
        #
        for k in range(N):
            #UPDATE A
            if(k != p and k != q):
                aux_p =   matrix_A[k][p] * cos + matrix_A[k][q] * sin
                aux_q = - matrix_A[k][p] * sin + matrix_A[k][q] * cos
                #
                matrix_A[k][p] = aux_p
                matrix_A[p][k] = aux_p
                matrix_A[k][q] = aux_q
                matrix_A[q][k] = aux_q
            #UPDATE U
            aux_p =   matrix_U[k][p] * cos + matrix_U[k][q] * sin
            aux_q = - matrix_U[k][p] * sin + matrix_U[k][q] * cos
            matrix_U[k][p] = aux_p
            matrix_U[k][q] = aux_q
        #COMPUTE ERROR
        p, q = max_non_diag_position(matrix_A)
        error = N * abs(matrix_A[p][q])
        iter += 1
    return matrix_A, matrix_U, iter, error

# matrix_A = [[4, 2, 0],
#             [2, 5, 3],
#             [0, 3, 6]]

matrix_A = [[3  , 0.4, 5],
            [0.4, 4  , 0.1],
            [5  , 0.1, -2]]

matrix_A, matrix_U, iter, error = jacobi_method(matrix_A, 10, 0.01)

print(matrix_A)
print()
print(matrix_U)
