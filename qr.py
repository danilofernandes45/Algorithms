qr_decomposition(matrix):

    q = [[] for i in range(len(matrix))]
    r = [[0]*len(matrix) for i in range(len(matrix))]

    for i in range(len(matrix)):
        vec = []
        norm = 0
        for j in range(len(matrix)):
            value = matrix[j][i]
            for k in range(j):
                value -= r[k][i] * q[j][k]
            vec.append(value)
            norm += value ** 2

        norm = sqrt(norm)

        for l in range(len(vec)):
            q[l][i] = vec[l] / norm
            for m in range(i, len(matrix)):
                r[i][m] += q[l][i]*matrix[l][m]

    return(q, r)
