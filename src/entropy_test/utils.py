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

# Using LZW compression
def compress(uncompressed):
    """Compress a string to a list of output symbols."""
 
    # Build the dictionary.
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    
    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
 
    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result

def decompress(compressed):
    """Decompress a list of output ks to a string."""
 
    dict_size = 256
    dictionary = dict((i, chr(i)) for i in range(dict_size))
 
    # use StringIO, otherwise this becomes O(N^2)
    # due to string concatenation in a loop
    result = StringIO()
    w = chr(compressed.pop(0))
    result.write(w)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result.write(entry)
 
        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
 
        w = entry
    return result.getvalue()

def entropy(prob):
    ent = 0.
    for p in prob:
        ent = ent - p*log(p,2)
    return ent

def list_to_string(a):
    return re.sub('\W+','', str(a) )

def lzw_test(delta):
    delta_string = list_to_string(delta)
    delta_compressed = compress(delta_string)
    delta_decompressed = decompress(delta_compressed)
    ratio = len(delta_compressed)/len(delta_string)
    error = np.sum(np.array([int(i) for i in delta_string]) != np.array([int(i) for i in delta_decompressed]))
    
    print("- using lzw")
    print("original size   : ", len(delta))
    print("compressed size : ", len(delta_compressed))
    print("ratio           : ", ratio)
    print("error           : ", error/len(delta_string))

def multinomial(n,p = [1/3,1/3,1/3]):
    final = []
    for i in range(n):
        result = np.random.multinomial(1, p)
        final.append(np.array(range(len(p)))[result == 1][0])
    return list(np.array(final).T)

def random_p(n=2):
    p = []
    for i in range(n):
        p0 = np.random.uniform(0,1,1)[0]
        p.append(p0)
    p = np.array(p)/np.sum(p)
    return list(p)

def get_p_tilda(uncompressed, n):
    p_tilda=[]
    for i in range(n):
        p_i = np.mean(np.array(uncompressed)==i)
        p_tilda.append(p_i)
    return p_tilda

# Markov
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
    mu = mu.real

    # print("evals")
    # print(evals)
    # print("evecs")
    # print(evecs)
    # print("stationary")
    # print(stationary)
    # print("mu")
    # print(mu)

    ent = 0
    for i in range(n):
        for j in range(n):
            ent = ent - mu[:,i] * P[i,j] * log(P[i,j],2)
    return ent[0]

def regression_model(n, size, number_of_p=30, verbose=False, plot=True):
    # n = the number of states
    dict_size = n
    dictionary = {i : chr(i) for i in range(dict_size)}
    
    ratio_list =[]
    true_entropy = []
    
    probabilities=[]
    for i in range(number_of_p):
        probabilities.append(random_p(n))
            
    for p in probabilities:
        true_entropy.append(entropy(p))
        
        uncompressed =str()        
        uncomp_numbers = multinomial(size, p)
        for i in uncomp_numbers:
            uncompressed = uncompressed + dictionary[i]

        compressed = compress(uncompressed)
        compression_ratio = len(compressed)/len(uncompressed)
        ratio_list.append(compression_ratio)
        
        if verbose:
            print("p : ", p)
            print("theoretical entropy: ", entropy(p))
            print("compression ratio: ", compression_ratio)
            print()

    # linear regression
    reg = LinearRegression(fit_intercept=True).fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(ratio_list[:]))
    print("y = ax + b model")
    print("a = ", reg.coef_)
    print("b = ", reg.intercept_)

    if plot:
        plt.scatter(true_entropy, ratio_list, marker='.', label = "LZW compressor")
        plt.plot(true_entropy, reg.predict(np.array(true_entropy).reshape(-1,1)), label="regression", color="orange")
        plt.title("Compression ratio - entropy regression model \n Multinomial with {} states, size={}".format(n, size))
        plt.xlabel("entropy")
        plt.ylabel("compression ratio")
        plt.legend()
        plt.show()

    return reg, ratio_list, true_entropy

def get_entropy(n, size, compression_ratio, name="a Markov process", plot=True):
    # n = number of states
    # size = length of a sequence
    # compression ratio = comp.ratio of the sequence of interest

    # mapping compression ratio to entropy
    reg, ratio_list, true_entropy = regression_model(n, size, number_of_p=100, verbose=False, plot=plot)
    reg_inv = LinearRegression(fit_intercept=True).fit(np.array(ratio_list[:]).reshape(-1, 1), np.array(true_entropy[:]))
    ent = reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))[0]

    if plot:
        print("plot")
        print("estimated entropy = ", ent)
        print("compression ratio = ", compression_ratio)
        print("reg.predict(est)  = ", reg.predict([[ent]]))
        plt.scatter(true_entropy, ratio_list, marker='.')
        plt.plot(reg_inv.predict(np.array(ratio_list).reshape(-1,1)), ratio_list, label="regression", color="orange")
        plt.axvline(ent, color="grey", alpha=0.5)
        plt.axhline(compression_ratio, color="grey", alpha=0.5)
        plt.scatter(ent, compression_ratio, color="red", label="Estimated entropy={}".format(ent.round(3)))

        plt.title("Estimated entropy of {} with Size {}".format(name, size))
        plt.xlabel("entropy")
        plt.ylabel("compression ratio")
        plt.legend()
        plt.show()
    return ent



# Fano's ternary (alphabet size = 3)
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