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
from ternary import *

def test(p, size, name, plot, verbose):
    # sample binary string
    print("size: ", size)
    print("p   : ", p)
    uncompressed = multinomial(size, p)
    p_tilda = [np.mean(np.array(uncompressed)==0), np.mean(np.array(uncompressed)==1), np.mean(np.array(uncompressed)==2)]

    # compression
    uncompressed = list_to_string(uncompressed)
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)

    # entropy
    estimated_ent = get_entropy_ternary(size, compression_ratio, name, plot)
    theoretical_ent = entropy(p)
    # empirical_ent = entropy(p_tilda)
    empirical_ent = 0

    # lower bound
    # lb = g_inverse(estimated_ent, a=0.005)

    if verbose:
        print("p_tilda            : ", np.round(p_tilda,3))
        print("Compression ratio: ", compression_ratio)
        print("Estimated entropy: ", estimated_ent)
        print("Theoretical entropy: ", theoretical_ent)
        # print("Empirical entropy: ", empirical_ent)
        # print("P(e) lower bound   : ", lb)
        print()

    return compression_ratio, estimated_ent, theoretical_ent, empirical_ent

def main():
    #### edit here ####
    # powers = [6,7,8]
    powers = [6,7,8,9,10,11]
    power = 9
    size = 2 ** power
    name = "a ternary multinomial process"
    plot = False
    verbose = True
    ###################
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    data = np.zeros([1,len(powers)])
    for num in range(50):
        theo_ent = []
        est_ent = []
        p = random_p(n=3)

        for power in powers:
            size = 2 ** power
            compression_ratio, estimated_ent, theoretical_ent, empirical_ent = test(p, size, name, plot, verbose)
            theo_ent.append(theoretical_ent)
            est_ent.append(estimated_ent)

        error = np.abs(np.array(theo_ent) - np.array(est_ent))
        data = np.vstack((data, error))
    data = data[1:,:]
    # print(data)
    # print(np.log2(data))
    plt.title("Absolute discrepancy between theoretical and estimated entropy")
    plt.xlabel("log length (base=2)")
    plt.ylabel("log absolute error")
    plt.boxplot(np.log2(data))
    plt.xticks(np.arange(1,len(powers)+1),powers)
    # plt.axhline(np.mean(error), color="red", label="mean={}".format(np.mean(error).round(3)))
    # plt.legend()
    # plt.savefig("result/ternary_multinomial_{}.png".format(size))
    plt.savefig("result/ternary_multinomial_boxplot.png")
    plt.show()
    plt.clf()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
