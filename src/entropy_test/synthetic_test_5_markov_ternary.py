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

    # print("evals")
    # print(evals)
    # print("evecs")
    # print(evecs)
    # print("stationary")
    # print(stationary)
    # print("mu")
    # print(mu)
    --> mu 에서 허수 파트만 없애고 가꼬오면 됨

    ent = 0
    for i in range(n):
        for j in range(n):
            ent = ent - mu[:,i] * P[i,j] * log(P[i,j],2)
    return ent[0]


def regression_model_ternary(size, number_of_p=30, verbose=False):
    ratio_list =[]
    true_entropy = []
    
    probabilities=[]
    for i in range(number_of_p):
        P = get_P(n=3)
        probabilities.append(P)
            
    for P in probabilities:
        # alpha, beta = P[0,1],P[1,0]
        # true_entropy.append(entropy([beta/(alpha+beta), alpha/(alpha+beta)]))
        true_entropy.append(entropy_rate(P))

        uncompressed = markov(size, P, initial = 0)
        uncompressed = list_to_string(uncompressed)
        # print(uncompressed)
        compressed = compress(uncompressed)
        compression_ratio = len(compressed)/len(uncompressed)
        ratio_list.append(compression_ratio)
        
        if verbose:
            print("p : ", p)
            print("theoretical entropy: ", true_entropy[-1])
            print("compression ratio: ", compression_ratio)
            print()

    # linear regression
    print(true_entropy)
    reg = LinearRegression().fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(ratio_list[:]))
    print("y = ax + b model")
    print("a = ", reg.coef_)
    print("b = ", reg.intercept_)

    # plt.scatter(true_entropy, ratio_list, marker='.', label = "LZW compressor")
    # plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")

    # plt.title("Compression Ratio for ternary Markov chain with Size {}".format(size))
    # plt.xlabel("theoretical entropy")
    # plt.ylabel("compression ratio")
    # plt.legend()
    # plt.show()

    return reg, ratio_list, true_entropy

def get_entropy(size, compression_ratio):
    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model_ternary(size, number_of_p=10, verbose=False)
    reg_inv = LinearRegression().fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
    ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))

    plt.scatter(true_entropy, ratio_list, marker='.')
    plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")
    plt.axvline(ent, color="grey", alpha=0.5)
    plt.axhline(compression_ratio, color="grey", alpha=0.5)
    plt.scatter(ent, compression_ratio, color="red")

    plt.title("Compression Ratio for a ternary Markov chain with Size {}".format(size))
    plt.xlabel("theoretical entropy")
    plt.ylabel("compression ratio")
    plt.legend()
    plt.show()
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

def test():

    ### edit here ####
    power = 10
    size = 2 ** power
    P = get_P(n=3)
    ###################

    # sample ternary string
    print("size: ", size)
    print("P   : ")
    print(P)
    uncompressed = markov(size, P, initial = 0)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1)]
    uncompressed = list_to_string(uncompressed)
    compressed = compress(uncompressed)
    
    # # compression ratio
    # compression_ratio = len(compressed)/len(uncompressed)
    # print(uncompressed)
    # print("Compression ratio: ", compression_ratio)
    # print()

    # entropy
    # estimated_ent = get_entropy(size, compression_ratio)    
    theoretical_entropy = entropy_rate(P)

    print("experiment")
    init_p = np.array([0.1,0.2,0.7])
    for i in range(10):
        init_p = np.dot(init_p, P)
        print(init_p)


    # theo_ent = []
    # est_ent = []
    # for test in range(1):
    #     print()
    #     print("*", test+1)

    #     #### edit here ####
    #     power = 10
    #     size = 2 ** power
    #     P = get_P(n=3)
    #     ###################

    #     # sample ternary string
    #     print("size: ", size)
    #     print("P   : ")
    #     print(P)
    #     uncompressed = markov(size, P, initial = 0)
    #     p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1)]
    #     uncompressed = list_to_string(uncompressed)
    #     compressed = compress(uncompressed)
        
    #     # compression ratio
    #     compression_ratio = len(compressed)/len(uncompressed)
    #     # print(uncompressed)
    #     print("Compression ratio: ", compression_ratio)
    #     print()

    #     # entropy
    #     estimated_ent = get_entropy(size, compression_ratio)    
    #     theoretical_entropy = entropy_rate(P)

    #     print("Estimated entropy: ", estimated_ent)
    #     print("Theoretical entropy: ", theoretical_entropy)
    #     print("Empirical entropy: ", entropy(p_tilda))
    #     theo_ent.append(theoretical_entropy)
    #     est_ent.append(estimated_ent)

    # error = np.abs(np.array(theo_ent) - np.array(est_ent))
    # print(np.mean(error))
    # plt.title("Absolute discrepancy between theoretical and estimated entropy, string length={}".format(size))
    # plt.hist(error)
    # plt.axvline(np.mean(error), color="red", label="mean={}".format(np.mean(error)))
    # plt.legend()
    # plt.show()


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    test()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
