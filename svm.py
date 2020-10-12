from math import exp
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

def gaussianClassifier(X, y, alpha, b, gamma, new_x):
    nrow = len(X)
    ncol = len(X[0])
    sum = -b
    for i in range(nrow):
        value_i = 0
        for l in range(ncol):
            value_i += ( X[i][l] - new_x[l] ) ** 2
        sum += alpha[i] * y[i] * exp( -gamma * value_i )

    if(sum < 0):
        return -1
    return +1

def computeKernelMatrix(X, kernel, gamma = 1):
    nrow = len(X)
    ncol = len(X[0])
    K = []
    for i in range(nrow):
        K.append([])
        for j in range(nrow):
            value = 0
            if(kernel == "linear"):
                for l in range(ncol):
                    value += X[i][l] * X[j][l]

            elif(kernel == "gaussian"):
                for l in range(ncol):
                    value += ( X[i][l] - X[j][l] ) ** 2
                value = exp( -gamma * value )

            K[i].append(value)
    return K

def zeroVector(N):
    vector = []
    for i in range(N):
        vector.append(0)
    return vector

def scalarVector(scalar, vector):
    N = len(vector)
    new_vector = []
    for i in range(N):
        new_vector.append(scalar * vector[i])
    return new_vector

def checkKKTConditions(E_i, y_i, alpha_i, C, eps):
    if (alpha_i == 0 and y_i*E_i >= 0):
        return True
    if (0 < alpha_i and alpha_i < C and abs(y_i*E_i) < eps):
        return True
    if (alpha_i == C and y_i*E_i <= 0):
        return True
    return False

def pickRandomAlphas(E, y, alpha, C, eps):
    p = -1
    q = -1
    N = len(E)
    available_alpha = []
    for d in range(N):
        if( not checkKKTConditions(E[d], y[d], alpha[d], C, eps) ):
            available_alpha.append(d)

    if( len(available_alpha) == 1 ):
        p = available_alpha[0]
        q = p
        count = 0
        while( abs(E[p] - E[q]) < eps and count < N ):
            q = random.randint(0, N - 1)
            count += 1

    elif( len(available_alpha) > 1):
        id = random.randint(0, len(available_alpha) - 1)
        p = available_alpha[id]
        available_alpha.pop(id)
        q = p
        count = 0
        while( abs(E[p] - E[q]) < eps and count < len(available_alpha) ):
            id = random.randint(0, len(available_alpha) - 1)
            q = available_alpha[id]
            count += 1

    return (p, q)

def pickAlphas(E, y, alpha, C, eps):
    p = -1
    q = -1
    N = len(E)
    for d in range(N):
        if( not checkKKTConditions(E[d], y[d], alpha[d], C, eps) ):
            if(p == -1): p = d
            elif(q == -1):
                if( E[d] > E[p] ):
                    q = p
                    p = d
                elif( E[d] < E[p] ):
                    q = d

            elif( E[d] > E[p] ): p = d
            elif( E[d] < E[q] ): q = d

    if( p != -1 and q == -1 ):
        q = 0
        for d in range(1, N):
            if( abs( E[p] - E[d] ) > abs( E[p] - E[q] ) ):
                q = d

    return (p, q)

def SequencialMinimalOptimization(X, y, kernel, C, eps, gamma = 1, max_iter = 1000):
    #Initialization
    nrow = len(X)
    ncol = len(X[0])
    K = computeKernelMatrix(X, kernel, gamma)
    alpha = zeroVector(nrow)
    W = None
    b = 0
    E = scalarVector(-1, y)
    L = 0
    H = C

    num_iter = 0

    if(kernel == "linear"):
         W = zeroVector(ncol)

    #Loop
    p, q = pickAlphas(E, y, alpha, C, eps)
    # delta_Q = 2*eps
    while( p != -1 and q != -1 and num_iter <= max_iter):
        print("(%d, %d)"%(p,q))
        const = 0
        for i in range(nrow):
            if(i != p and i != q):
                const +=  alpha[i] * y[i]
        const = -y[q] * const

        if(y[p] != y[q]):
            L = max(0, -const)
            H = min(C, C - const)
        else:
            L = max(0, const - C)
            H = min(const, C)

        delta_alpha_p = y[p] * (E[p] - E[q]) / (-K[p][p] + 2*K[p][q] -K[q][q])

        if( (alpha[p] + delta_alpha_p) > H ):
            delta_alpha_p = H - alpha[p]
        elif( (alpha[p] + delta_alpha_p) < L ):
            delta_alpha_p = L - alpha[p]

        alpha[p] = alpha[p] + delta_alpha_p
        alpha[q] = alpha[q] - y[p]*y[q]*delta_alpha_p

        if(kernel == "linear"):
            for i in range(ncol):
                W[i] = W[i] + delta_alpha_p * y[p] * (X[p][i] - X[q][i])

        delta_b = 0
        if( 0 < alpha[p] and alpha[p] < C):
            delta_b = E[p] + delta_alpha_p * y[p] * (K[p][p] - K[p][q])
        elif( 0 < alpha[q] and alpha[q] < C):
            delta_b = E[q] + delta_alpha_p * y[p] * (K[p][q] - K[q][q])
        else:
            delta_b = ( E[p] + E[q] + delta_alpha_p * y[p] * ( K[p][p] - K[q][q] ) ) / 2

        b = b + delta_b

        for i in range(nrow):
            E[i] = E[i] + delta_alpha_p * y[p] * ( K[p][i] - K[q][i] ) - delta_b

        delta_Q = ( 1 - y[p]*y[q] ) * delta_alpha_p - 0.5 * (K[p][p] - 2*K[p][q] + K[q][q]) * delta_alpha_p ** 2
        print(delta_Q)

        if(delta_Q < eps):
            p, q = pickRandomAlphas(E, y, alpha, C, eps)
        else:
            p, q = pickAlphas(E, y, alpha, C, eps)
        num_iter += 1

    count  = 0
    for d in range(nrow):
        if(not checkKKTConditions(E[d], y[d], alpha[d], C, eps)):
            count += 1
    print(count)

    print(num_iter)

    return (W, b, alpha, num_iter)

#=============================================TEST=====================================================

def linear_test():

    x1 = np.concatenate( ( np.arange(0, 1, 0.02), np.arange(0, 1, 0.02) ) )
    np.random.seed(0)
    y1 = ( 0.5*x1 + 0.25 ) + 0.5*np.random.random(100)
    x1_noise = np.arange(0, 1, 0.1)
    np.random.seed(1)
    y1_noise = ( 0.5*x1_noise + 0.25 ) - 0.5*np.random.random(10)

    x1 = np.concatenate( (x1, x1_noise) )
    y1 = np.concatenate( (y1, y1_noise) )

    plt.scatter(x1, y1, color = "red")

    x2 = np.concatenate( ( np.arange(0, 1, 0.02), np.arange(0, 1, 0.02) ) )
    np.random.seed(2)
    y2 = ( 0.5*x2 + 0.25 ) - 0.5*np.random.random(100)
    x2_noise = np.arange(0, 1, 0.1)
    np.random.seed(3)
    y2_noise = ( 0.5*x2_noise + 0.25 ) + 0.5*np.random.random(10)

    x2 = np.concatenate( (x2, x2_noise) )
    y2 = np.concatenate( (y2, y2_noise) )

    plt.scatter(x2, y2, color = "blue")

    X = np.zeros((220, 2))
    X[:,0] = np.concatenate((x1, x2))
    X[:,1] = np.concatenate((y1, y2))
    y = np.zeros(220)
    y[  0:110] = +1
    y[110:220] = -1

    W, b, alpha, num_iter = SequencialMinimalOptimization(X, y, kernel = "linear", C = 1, eps = 0.001)

    x = np.arange(0, 1, 0.01)
    y = ( -W[0]*x + b ) / W[1]
    print("y = %.3f x + %.3f"%(-W[0]/W[1], b/W[0]))
    plt.plot(x, y)
    plt.show()

def non_linear_test():

    x1 = np.concatenate( ( np.arange(0, 1, 0.02), np.arange(0, 1, 0.02) ) )
    np.random.seed(0)
    y1 = 0.25*np.sin(2*np.pi*x1) + 0.25*np.random.random(100)
    x1_noise = np.arange(0, 1, 0.1)
    np.random.seed(1)
    y1_noise = 0.25*np.sin(2*np.pi*x1_noise) - 0.25*np.random.random(10)

    x1 = np.concatenate( (x1, x1_noise) )
    y1 = np.concatenate( (y1, y1_noise) )

    x2 = np.concatenate( ( np.arange(0, 1, 0.02), np.arange(0, 1, 0.02) ) )
    np.random.seed(2)
    y2 = 0.25*np.sin(2*np.pi*x2) - 0.25*np.random.random(100)
    x2_noise = np.arange(0, 1, 0.1)
    np.random.seed(3)
    y2_noise = 0.25*np.sin(2*np.pi*x2_noise) + 0.25*np.random.random(10)

    x2 = np.concatenate( (x2, x2_noise) )
    y2 = np.concatenate( (y2, y2_noise) )

    x_y_1, _ = make_blobs(n_samples = 25, n_features = 2, cluster_std = 0.05, centers = [[0.25, -0.25]], shuffle=False, random_state = 123)

    X = np.zeros((245, 2))
    X[:,0] = np.concatenate((x1, x2, x_y_1[:,0]))
    X[:,1] = np.concatenate((y1, y2, x_y_1[:,1]))
    y = np.zeros(245)
    y[  0:110] = +1
    y[110:220] = -1
    y[220:245] = +1

    W, b, alpha, num_iter = SequencialMinimalOptimization(X, y, kernel = "gaussian", C = 1, gamma = 10, eps = 0.001)

    x_test_1 =[]
    y_test_1 = []
    x_test_2 =[]
    y_test_2 = []

    for i in np.arange(0, 1, 0.005):
        for j in np.arange(-0.5, 0.5, 0.005):
            class_test = gaussianClassifier(X, y, alpha, b, gamma = 10, new_x = [i,j])
            if(class_test == +1):
                x_test_1.append(i)
                y_test_1.append(j)
            else:
                x_test_2.append(i)
                y_test_2.append(j)

    plt.scatter(x_test_1, y_test_1, color = "red", alpha = 0.1)
    plt.scatter(x_test_2, y_test_2, color = "blue", alpha = 0.1)

    plt.scatter(x1, y1, color = "red")
    plt.scatter(x2, y2, color = "blue")
    plt.scatter(x_y_1[:,0], x_y_1[:,1], color = "red")

    plt.show()

#linear_test()
non_linear_test()
