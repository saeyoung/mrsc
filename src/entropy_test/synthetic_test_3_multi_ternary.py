#############################################################
#
# Test 2. regression model for a fixed input string size N, Ternary
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

from utils import *

# regression model for a fixed input string size N
def regression_model_ternary(size, number_of_p=30, verbose=False):
    ratio_list =[]
    true_entropy = []
    
    probabilities=[]
    for i in range(number_of_p):
        p1= np.random.uniform(0,1,1)[0]
        p2 = np.random.uniform(0,1,1)[0]
        p3 = np.random.uniform(0,1,1)[0]
        
        probabilities.append([p1,p2,p3]/(p1+p2+p3))
            
    for p in probabilities:
        true_entropy.append(entropy(p))
        uncompressed = list_to_string(multinomial(size, p))
        compressed = compress(uncompressed)
        compression_ratio = len(compressed)/len(uncompressed)
        ratio_list.append(compression_ratio)
        
        if verbose:
            print("p : ", p)
            print("theoretical entropy: ", entropy(p))
            print("compression ratio: ", compression_ratio)
            print()

    # linear regression
    reg = LinearRegression().fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(ratio_list[:]))
    print("y = ax + b model")
    print("a = ", reg.coef_)
    print("b = ", reg.intercept_)

    plt.scatter(true_entropy, ratio_list, marker='.', label = "LZW compressor")
    plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")

    plt.title("Compression Ratio for Multinomial(3,p) with Size {}".format(size))
    plt.xlabel("theoretical entropy")
    plt.ylabel("compression ratio")
    plt.legend()
    plt.show()

    return reg, ratio_list, true_entropy

def get_entropy(size, compression_ratio):
    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model_ternary(size, number_of_p=30, verbose=False)
    reg_inv = LinearRegression().fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
    ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))

    plt.scatter(true_entropy, ratio_list, marker='.')
    plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")
    plt.axvline(ent, color="grey", alpha=0.5)
    plt.axhline(compression_ratio, color="grey", alpha=0.5)
    plt.scatter(ent, compression_ratio, color="red")

    plt.title("Compression Ratio for Multinomial(3,p) with Size {}".format(size))
    plt.xlabel("theoretical entropy")
    plt.ylabel("compression ratio")
    plt.legend()
    plt.show()
    return ent

def g(p):
    return entropy([p,1-p]) + p

def dg(p):
    return -log(p/(1-p),2) + 1

def g_inverse(H, a=0.001):
    # from entropy value, get p s.t. 0 < p < 0.5
    # a = accuracy
    p_hat = 0.33
    err = np.abs(g(p_hat) - H)
    while(err > a):
        err = np.abs(g(p_hat) - H)
        p_hat = p_hat - 0.01* (g(p_hat) - H) * dg(p_hat)
        if (p_hat < 0):
            p_hat = 0
        if (p_hat > 2/3):
            p_hat = 2/3
    
    return p_hat

def test():
    #### edit here ####
    power = 10
    size = 2 ** power
    p = [1/6, 2/3, 1/6]
    ###################

    # sample binary string
    print("size: ", size)
    print("p   : ", p)
    uncompressed = multinomial(size, p)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1), np.mean(np.array(uncompressed)==2)]

    uncompressed = list_to_string(uncompressed)
    compressed = compress(uncompressed)
    
    # len()
    compression_ratio = len(compressed)/len(uncompressed)
    print("Compression ratio: ", compression_ratio)
    print()

    # entropy
    estimated_ent = get_entropy(size, compression_ratio)
    print("Estimated entropy: ", estimated_ent)
    print("Theoretical entropy: ", entropy(p))
    print("Empirical entropy: ", entropy(p_tilda))

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
