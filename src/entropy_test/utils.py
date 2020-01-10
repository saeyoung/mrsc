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

def get_delta(x):
    delta = np.zeros(len(x)-1)
    delta[(x - x.shift(+1))[1:]>0] = 1
    # 1 = increase
    # 0 = not increase (decrease or stays the same)
    delta = delta.astype(int)
    return list(delta)

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



