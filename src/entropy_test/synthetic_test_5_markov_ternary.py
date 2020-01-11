#############################################################
#
# Test 5. regression model for a fixed input string size N, Markov (Ternary)
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import copy
import pickle
from math import log, e
from sklearn.linear_model import LinearRegression
from numpy.linalg import eig

from utils import *

def get_P(n=2):
    P = np.zeros((n,n))
    for i in range(n):
        p=[]
        for j in range(n):
            p.append(np.random.uniform(0,1,1)[0])
        
        p = p/np.sum(p)
        P[i,:] = p
    return P

def get_next(this_obs, P):
    p = P[this_obs,:].flatten()
    next_obs = multinomial(1,p)
    return next_obs

def markov(len, P, initial = 0):
    this_obs = initial
    observations = [this_obs]
    for i in range(len):
        this_obs = get_next(this_obs, P)
        observations.append(this_obs[0])
    return observations

def entropy_rate(P):
    # P = transition matrix (n by n)
    # mu = asymptotic distribution (1 by n)
    n = P.shape[0]

    evals, evecs = eig(P.T)
    loc = np.where(np.abs(evals - 1.) < 0.0001)[0]
    stationary = evecs[:,loc].T
    mu = stationary / np.sum(stationary)
    mu = mu.real

    # print("evals")
    # print(evals)
    # print("evecs")
    # print(evecs)
    # print("stationary")
    # print(stationary)
    # print("mu")
    # print(mu)

    ent = 0
    for i in range(n):
        for j in range(n):
            ent = ent - mu[:,i] * P[i,j] * log(P[i,j],2)
    return ent[0]

# def g(p):
#     return entropy([p,1-p]) + p

# def dg(p):
#     return -log(p/(1-p),2) + 1

# def g_inverse(H, a=0.001):
#     # from entropy value, get p s.t. 0 < p < 0.5
#     # a = accuracy
#     p_hat = 0.33
#     err = np.abs(g(p_hat) - H)
#     while(err > a):
#         err = np.abs(g(p_hat) - H)
#         p_hat = p_hat - 0.01* (g(p_hat) - H) * dg(p_hat)
#         if (p_hat < 0):
#             p_hat = 0
#         if (p_hat > 2/3):
#             p_hat = 2/3
    
#     return p_hat

def test(size, name, plot, verbose):
    P = get_P(n=3)
    
    # sample binary string
    print("size: ", size)
    print("P   : ")
    print(P)
    uncompressed = markov(size, P, initial = 0)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1), np.mean(np.array(uncompressed)==2)]

    # compression
    uncompressed = list_to_string(uncompressed)
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)

    # entropy
    estimated_ent = get_entropy_ternary(size, compression_ratio, name, plot=True)
    theoretical_ent = entropy_rate(P)
    empirical_ent = entropy(p_tilda)

    print("Compression ratio  : ", compression_ratio)
    print("Estimated entropy  : ", estimated_ent)
    print("Theoretical entropy: ", theoretical_ent)
    print("Empirical entropy  : ", empirical_ent)

    return compression_ratio, estimated_ent, theoretical_ent, empirical_ent

    # print("experiment")
    # init_p = np.array([0.1,0.2,0.7])
    # for i in range(10):
    #     init_p = np.dot(init_p, P)
    #     print(init_p)


def main():
#### edit here ####
    power = 10
    size = 2 ** power
    name = "a ternary Markov process"
    plot = True
    verbose = True
    ###################
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    theo_ent = []
    est_ent = []
    for num in range(2):
        compression_ratio, estimated_ent, theoretical_ent, empirical_ent = test(size, name, plot, verbose)
        theo_ent.append(theoretical_ent)
        est_ent.append(estimated_ent)

    error = np.abs(np.array(theo_ent) - np.array(est_ent))
    print(np.mean(error))
    plt.xlabel("Absolute discrepancy between theoretical and estimated entropy")
    plt.title("Error distribution, string length={}".format(size))
    plt.hist(error)
    plt.axvline(np.mean(error), color="red", label="mean={}".format(np.mean(error).round(3)))
    plt.legend()
    plt.show()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
