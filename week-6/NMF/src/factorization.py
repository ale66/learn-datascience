#!/usr/bin/python
#
# Direct NMF implementation.
#
# Created by Albert Au Yeung (2010)
#
# An implementation of NMF
#
try:
    import numpy as np
except:
    print("This implementation requires the numpy module.")
    exit(0)

#####################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):

    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
               
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
               
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

        if e < 0.001:
            break

    return P, Q.T

#########################################################

if __name__ == "__main__":

    ratings  = [[1, 1, 1, 0, 0],
                [3, 3, 3, 0, 0],
                [4, 4, 4, 0, 0],
                [5, 5, 5, 0, 0],
                [0, 0, 0, 4, 4],
                [0, 0, 0, 5, 5],
                [0, 0, 0, 2, 2]]


    R = np.array(ratings)

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)

    print(nP)

    print(nQ.T)

    print(np.dot(nP, nQ.T))

    print(np.rint(np.dot(nP, nQ.T)))
