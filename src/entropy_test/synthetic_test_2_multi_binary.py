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
def test(size, name="a Bernoulli process", plot=True, verbose=True):
    p = random_p()

    # sample binary string
    uncompressed = multinomial(size, p)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1)]

    # compression
    uncompressed = list_to_string(uncompressed)
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)
    
    # entropy
    estimated_ent = get_entropy(size, compression_ratio, name, plot)
    theoretical_ent = entropy(p)
    empirical_ent = entropy(p_tilda)
    
    if verbose:
        print()
        print("size: ", size)
        print("p   : ", p)
        print("Compression ratio: ", compression_ratio)
        print("Estimated entropy: ", estimated_ent)
        print("Theoretical entropy: ", theoretical_ent)
        print("Empirical entropy: ", empirical_ent)
    return compression_ratio, estimated_ent, theoretical_ent, empirical_ent

def main():
    #### edit here ####
    power = 10
    size = 2 ** power 
    name = "a Bernoulli process"
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
