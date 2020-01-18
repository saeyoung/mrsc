######################################################
#
# Functions for markov processes with large state space
#
######################################################
import numpy as np
import pandas as pd
import random
import copy
import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.linalg import eig

# for entropy
# import zlib
import re
from math import log, e
from io import StringIO
from utils import *

# def regression_model(n, size, number_of_p=30, verbose=False, plot=True):
#     # n = the number of states
#     dict_size = n
#     dictionary = {i : chr(i) for i in range(dict_size)}
    
#     ratio_list =[]
#     true_entropy = []
    
#     probabilities=[]
#     for i in range(number_of_p):
#         probabilities.append(random_p(n))
            
#     for p in probabilities:
#         true_entropy.append(entropy(p))
        
#         uncompressed = str()        
#         uncomp_numbers = multinomial(size, p)
#         for i in uncomp_numbers:
#             uncompressed = uncompressed + dictionary[i]

#         compressed = compress(uncompressed)
#         compression_ratio = len(compressed)/len(uncompressed)
#         ratio_list.append(compression_ratio)
        
#         if verbose:
#             print("p : ", p)
#             print("theoretical entropy: ", entropy(p))
#             print("compression ratio: ", compression_ratio)
#             print()

#     # linear regression
#     reg = LinearRegression(fit_intercept=False).fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(ratio_list[:]))
#     print("y = ax + b model")
#     print("a = ", reg.coef_)
#     print("b = ", reg.intercept_)

#     if plot:
#         plt.scatter(true_entropy, ratio_list, marker='.', label = "LZW compressor")
#         plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")

#         plt.title("Compression ratio - entropy regression model \n Ternary multinomial, size={}".format(size))
#         plt.xlabel("entropy")
#         plt.ylabel("compression ratio")
#         plt.legend()
#         plt.show()

#     return reg, ratio_list, true_entropy

def get_entropy(n, size, compression_ratio, name="a Markov process", plot=True):
    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model(n, size, number_of_p=100, verbose=False, plot=plot)
    reg_inv = LinearRegression(fit_intercept=False).fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
    ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))[0]

    if plot:
        print("plot")
        print("estimated entropy = ", ent)
        print("compression ratio = ", compression_ratio)
        print("reg.predict(est)  = ", reg.predict([[ent]]))
        plt.scatter(true_entropy, ratio_list, marker='.')
        plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")
        plt.axvline(ent, color="grey", alpha=0.5)
        plt.axhline(compression_ratio, color="grey", alpha=0.5)
        plt.scatter(ent, compression_ratio, color="red", label="Estimated entropy={}".format(ent.round(3)))

        plt.title("Estimated entropy of {} with Size {}".format(name, size))
        plt.xlabel("entropy")
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

