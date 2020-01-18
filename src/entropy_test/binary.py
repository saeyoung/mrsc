######################################################
#
# Utility functions
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

def get_delta(x):
    delta = np.zeros(len(x)-1)
    delta[(x - x.shift(+1))[1:]>0] = 1
    # 1 = increase
    # 0 = not increase (decrease or stays the same)
    delta = delta.astype(int)
    return list(delta)

def regression_model_binary(size, number_of_p=30, verbose=False, plot=True):
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

    if plot:
        plt.plot(true_entropy, ratio_list, marker='.', label = "LZW compressor")
        plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression")

        plt.title("Compression ratio - entropy regression model \n Bernoulli(p) with 0<p<0.5, size={}".format(size))
        plt.xlabel("entropy")
        plt.ylabel("compression ratio")
        plt.legend()
        plt.show()

    return reg, ratio_list, true_entropy

def get_entropy(size, compression_ratio, name="a random process", plot=True):
    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model_binary(size, number_of_p=30, verbose=False, plot=plot)
    reg_inv = LinearRegression().fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
    ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))[0]

    if plot:
        plt.scatter(true_entropy, ratio_list, marker='.')
        plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression model", color="orange")
        plt.axvline(ent, color="grey", alpha=0.5)
        plt.axhline(compression_ratio, color="grey", alpha=0.5)
        plt.scatter(ent, compression_ratio, color="red", label="estimated entropy ={}".format(ent.round(3)))

        plt.title("Estimated entropy of {} with Size {}".format(name, size))
        plt.xlabel("entropy")
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

