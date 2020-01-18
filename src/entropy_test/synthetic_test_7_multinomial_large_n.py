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

def test(n, p, size, name, plot, verbose):
    # make dictionary
    dict_size = n
    dictionary = {i : chr(i) for i in range(dict_size)}
    
    # make a list of numbers
    uncomp_numbers = multinomial(size, p)
    p_tilda = get_p_tilda(uncomp_numbers, n)

    # convert number list to string
    uncompressed = str()        
    for i in uncomp_numbers:
        uncompressed = uncompressed + dictionary[i]

    # compression
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)
    # print("compression ratio: ",compression_ratio)

    # entropy
    estimated_ent = get_entropy(n, size, compression_ratio, name=name, plot=plot)
    theoretical_ent = entropy(p)
    # empirical_ent = entropy(p_tilda)
    empirical_ent = 0

    # # lower bound
    # # lb = g_inverse(estimated_ent, a=0.005)

    if verbose:
        print("p_tilda            : ", np.round(p_tilda,3))
        print("Compression ratio  : ", compression_ratio)
        print("Estimated entropy  : ", estimated_ent)
        print("Theoretical entropy: ", theoretical_ent)

    return compression_ratio, estimated_ent, theoretical_ent, empirical_ent

def experiment():
    #### edit here ####
    n = 3 # number of states
    power = 10
    size = 2 ** power
    name = "Multinomial process with {} states".format(n)
    plot = True
    verbose = True
    ###################

    theo_ent=[]
    est_ent=[]
    for num in range(10):
        p = random_p(n=n)
        compression_ratio, estimated_ent, theoretical_ent, empirical_ent = test(n, p, size, name, plot, verbose)
        theo_ent.append(theoretical_ent)
        est_ent.append(estimated_ent)

    error = np.array(theo_ent)-np.array(est_ent)
    print("theoretical entropy")
    print(theo_ent)
    print("estimated entropy")
    print(est_ent)

def experiment_1():

    for n in [2,3,5,8,13,21]:
        #### edit here ####
        # n = 3 # number of states
        power = 10
        size = 2 ** power
        name = "Multinomial process with {} states".format(n)
        plot = False
        verbose = True
        ###################

        theo_ent=[]
        est_ent=[]
        for num in range(100):
            p = random_p(n=n)
            compression_ratio, estimated_ent, theoretical_ent, empirical_ent = test(n, p, size, name, plot, verbose)
            theo_ent.append(theoretical_ent)
            est_ent.append(estimated_ent)

        error = np.array(theo_ent)-np.array(est_ent)
        plt.title("Theoretical entropy - estimated entropy distribution \n {}".format(name))
        plt.hist(error)
        # plt.axhline(np.mean(error), color="red", label="mean={}".format(np.mean(error).round(3)))
        # plt.legend()
        plt.savefig("result/err_dist/{}_states_multinomial_error_distribution_{}.png".format(n, size))
        # plt.show()
        plt.clf()

def experiment_2():
    #### edit here ####
    powers = [9,10,11,12]
    # powers = [6,8]
    samples = 50
    plot = False
    verbose = True
    ###################

    for n in [2,3,5,8,13,21]:
        name = "Multinomial process with {} states".format(n)
        data = np.zeros([1,len(powers)])
        for num in range(samples):
            theo_ent = []
            est_ent = []
            p = random_p(n=n)

            for power in powers:
                size = 2 ** power
                compression_ratio, estimated_ent, theoretical_ent, empirical_ent = test(n, p, size, name, plot, verbose)
                theo_ent.append(theoretical_ent)
                est_ent.append(estimated_ent)

            error = np.abs(np.array(theo_ent) - np.array(est_ent))
            data = np.vstack((data, error))
        data = data[1:,:]

        # regression line
        y = np.median(np.log2(data),axis=0).reshape(len(powers),1)
        X = np.arange(len(powers)).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)

        plt.title("Absolute discrepancy between theoretical and estimated entropy \n {}".format(name))
        plt.xlabel("log length (base=2)")
        plt.ylabel("absolute error")
        plt.boxplot(data)
        plt.xticks(np.arange(1,len(powers)+1),powers)
        plt.savefig("result/boxplot/{}_states_multinomial_boxplot_semilog_{}_samples.png".format(n, samples))
        # plt.show()
        plt.clf()

        plt.title("Absolute discrepancy between theoretical and estimated entropy \n {}".format(name))
        plt.xlabel("log length (base=2)")
        plt.ylabel("log absolute error")
        plt.boxplot(np.log2(data))
        plt.plot(np.arange(1,len(powers)+1),reg.predict(X), color="red", label="r2 score={}".format(reg.score(X,y).round(3)))
        plt.xticks(np.arange(1,len(powers)+1),powers)
        plt.legend()
        plt.savefig("result/boxplot/{}_states_multinomial_boxplot_loglog_{}_samples.png".format(n, samples))
        # plt.show()
        plt.clf()

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    experiment_2()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
