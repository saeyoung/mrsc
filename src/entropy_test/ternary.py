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

def mean_thus_far(x):
    mean=[]
    for i in range(1, len(x)+1):
        mean.append(x[:i].mean())
    return np.array(mean)

def std_thus_far(x):
    std=[]
    for i in range(1, len(x)+1):
        std.append(x[:i].std())
    return np.array(std)

def get_gamma(y, x, alpha, window):
    # y : refence to calculate the mean/std
    # x : evaluate this based on men/std(y)
    # window = rolling window size
    # alpha = +- alpha * std
    
    roll_mean = y.rolling(window).mean()[window:]
    roll_std = y.rolling(window).std()[window:]
    thus_mean = mean_thus_far(y)[:window]
    thus_std = std_thus_far(y)[:window]
    thus_std[0]=0

    # upper boundary (0, 1)
    pre = thus_mean + thus_std * alpha
    post = np.array(roll_mean + roll_std * alpha)
    upper = np.hstack((pre, post))

    # lower boundary (-1, 0)
    pre = thus_mean - thus_std * alpha
    post = np.array(roll_mean - roll_std * alpha)
    lower = np.hstack((pre, post))
 
    gamma = np.zeros(len(x))
    gamma[x > upper] = 1
    gamma[x < lower] = -1

    # 1 = above mean + alpha*std
    # -1 = below mean - alpha*std
    # 0 = between mean +- alpha*std
    gamma = gamma.astype(int)
    return list(gamma), upper, lower

def regression_model_ternary(size, number_of_p=30, verbose=False, plot=True):
    ratio_list =[]
    true_entropy = []
    
    probabilities=[]
    for i in range(number_of_p):
        probabilities.append(random_p(3))
            
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

    if plot:
        plt.scatter(true_entropy, ratio_list, marker='.', label = "LZW compressor")
        plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")

        plt.title("Compression ratio - entropy regression model \n Ternary multinomial, size={}".format(size))
        plt.xlabel("entropy")
        plt.ylabel("compression ratio")
        plt.legend()
        plt.show()

    return reg, ratio_list, true_entropy

def get_entropy_ternary(size, compression_ratio, name="a random process", plot=True):
    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model_ternary(size, number_of_p=30, verbose=False, plot=plot)
    reg_inv = LinearRegression().fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
    ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))[0]

    if plot:
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

