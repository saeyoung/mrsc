#############################################################
#
# Test 4. regression model for a fixed input string size N, Markov (Binary)
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
from binary import *

def test(n,size, name="a binary Markov chain", plot=True, verbose=True):
    P = get_P(n)

    # sample binary string
    uncompressed = markov(size, P, initial = 0)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1)]
    uncompressed = list_to_string(uncompressed)
    compressed = compress(uncompressed)
    
    # compression ratio
    compression_ratio = len(compressed)/len(uncompressed)

    # entropy
    estimated_ent = get_entropy(size, compression_ratio, name, plot)    
    theoretical_ent = entropy_rate(P)
    # empirical_ent = entropy(p_tilda)
    empirical_ent = 0

    if verbose:
        print()
        print("size: ", size)
        print("P   : ")
        print(P)
        print("Compression ratio: ", compression_ratio)
        print("Estimated entropy: ", estimated_ent)
        print("Theoretical entropy: ", theoretical_ent)
        print("Empirical entropy: ", empirical_ent)
    return compression_ratio, estimated_ent, theoretical_ent, empirical_ent

def main():
    #### edit here ####
    power = 10
    size = 2 ** power
    n = 2
    name = "a binary Markov chain"
    plot = False
    verbose = True
    ###################

    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")
    
    for size in [64,256,450,1024,4096]:
        theo_ent = []
        est_ent = []
        for num in range(50):
            compression_ratio, estimated_ent, theoretical_ent, empirical_ent = test(n, size, name, plot, verbose)
            theo_ent.append(theoretical_ent)
            est_ent.append(estimated_ent)

        error = np.abs(np.array(theo_ent) - np.array(est_ent))
        print(np.mean(error))
        plt.xlabel("Absolute discrepancy between theoretical and estimated entropy")
        plt.title("Error distribution, string length={}".format(size))
        plt.hist(error)
        plt.axvline(np.mean(error), color="red", label="mean={}".format(np.mean(error).round(3)))
        plt.legend()
        plt.savefig("result/binary_markov_{}.png".format(size))
        # plt.show()
        plt.clf()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()


# def regression_model_binary(size, number_of_p=30, verbose=False):
#     ratio_list =[]
#     true_entropy = []
    
#     probabilities=[]
#     for i in range(number_of_p):
#         P = get_P(n=2)
#         probabilities.append(P)
            
#     for P in probabilities:
#         # alpha, beta = P[0,1],P[1,0]
#         # true_entropy.append(entropy([beta/(alpha+beta), alpha/(alpha+beta)]))
#         true_entropy.append(entropy_rate(P))

#         uncompressed = markov(size, P, initial = 0)
#         uncompressed = list_to_string(uncompressed)
#         # print(uncompressed)
#         compressed = compress(uncompressed)
#         compression_ratio = len(compressed)/len(uncompressed)
#         ratio_list.append(compression_ratio)
        
#         if verbose:
#             print("p : ", p)
#             print("theoretical entropy: ", true_entropy[-1])
#             print("compression ratio: ", compression_ratio)
#             print()

#     # linear regression
#     reg = LinearRegression().fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(ratio_list[:]))
#     print("y = ax + b model")
#     print("a = ", reg.coef_)
#     print("b = ", reg.intercept_)

#     # plt.scatter(true_entropy, ratio_list, marker='.', label = "LZW compressor")
#     # plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")

#     # plt.title("Compression Ratio for binary Markov chain with Size {}".format(size))
#     # plt.xlabel("theoretical entropy")
#     # plt.ylabel("compression ratio")
#     # plt.legend()
#     # plt.show()

#     return reg, ratio_list, true_entropy

# def get_entropy(size, compression_ratio):
#     # mapping compression ratio to entropy
#     reg, ratio_list, true_entropy = regression_model_binary(size, number_of_p=100, verbose=False)
#     reg_inv = LinearRegression().fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
#     ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))

#     # plt.scatter(true_entropy, ratio_list, marker='.')
#     # plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")
#     # plt.axvline(ent, color="grey", alpha=0.5)
#     # plt.axhline(compression_ratio, color="grey", alpha=0.5)
#     # plt.scatter(ent, compression_ratio, color="red")

#     # plt.title("Compression Ratio for a binary Markov chain with Size {}".format(size))
#     # plt.xlabel("theoretical entropy")
#     # plt.ylabel("compression ratio")
#     # plt.legend()
#     # plt.show()
#     return ent[0]
