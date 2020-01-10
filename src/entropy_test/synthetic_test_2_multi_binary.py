#############################################################
#
# Test 2. regression model for a fixed input string size N, Binary
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
def regression_model_binary(size, number_of_p=30, verbose=False):
    ratio_list =[]
    true_entropy = []
    for p0 in np.linspace(1e-5,0.5,number_of_p):
        p = [p0, 1-p0]
        true_entropy.append(entropy(p))
        uncompressed = multinomial(size, p)
        uncompressed = list_to_string(uncompressed)
        compressed = compress(uncompressed)
        compression_ratio = len(compressed)/len(uncompressed)
        ratio_list.append(compression_ratio)
        
        if verbose:
            print("p : ", p)
            print("theoretical entropy: ", entropy([p,1-p]))
            print("compression ratio: ", compression_ratio)
            print()

    # linear regression
    reg = LinearRegression().fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(ratio_list[:]))
    print("y = ax + b model")
    print("a = ", reg.coef_)
    print("b = ", reg.intercept_)

    plt.plot(true_entropy, ratio_list, marker='.', label = "LZW compressor")
    plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression")

    plt.title("Compression Ratio of size {} Bernoulli(p), 0<p<0.5".format(size))
    plt.xlabel("theoretical entropy")
    plt.ylabel("compression ratio")
    plt.legend()
    plt.show()

    return reg, ratio_list, true_entropy

def get_entropy(size, compression_ratio):
    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model_binary(size, number_of_p=30, verbose=False)
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

def f(p):
    return -p*log(p,2) - (1-p)*log(1-p,2)

def df(p):
    return -log(p/(1-p),2) 

def f_inverse(H, a=0.001):
    # from entropy value, get p s.t. 0 < p < 0.5
    # a = accuracy
    p_hat = 0.25
    err = np.abs(f(p_hat) - H)
    while(err > a):
        err = np.abs(f(p_hat) - H)
        p_hat = p_hat - 0.01* (f(p_hat) - H) * df(p_hat)
        if (p_hat<0):
            p_hat = e-15
        if (p_hat>0.5):
            p_hat = 0.5
    
    return p_hat
def test():
    #### edit here ####
    power = 10
    size = 2 ** power
    p = [2/5, 3/5]
    ###################

    # sample binary string
    print("size: ", size)
    print("p   : ", p)
    uncompressed = multinomial(size, p)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1)]

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
