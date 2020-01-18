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

