import pandas as pd
import numpy as np
from sklearn import linear_model

# personal libraries
from mrsc.src.predictions import cvxRegression

""" Exponential Weights Moving Average (EWMA) """
def applyEWMA(series, param=0.5):
    return series.ewm(param).mean()


""" Learn Weights for Predictions """
def learnWeights(x, y, method='convex', alpha=0.1):
    if method.lower() == 'linear':
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(x, y)
        weights = regr.coef_
    elif method.lower() == 'ridge':
        regr = linear_model.Ridge(fit_intercept=False, alpha=alpha)
        regr.fit(x, y)
        weights = regr.coef_
    else:
        regr = cvxRegression.ConvexRegression(x, y)
        weights = regr.x
    return weights


""" Multiplicative Weights Update (MWU) """
def MWU(preds, obs, weights, learningRate=0.2):
    weights = weights * (1 - learningRate * predLoss(preds, obs))
    #weights = weights * np.exp(-learningRate * predLoss(preds, obs))
    return weights / np.sum(weights)


""" Loss Function for MWU """
def predLoss(pred, obs):
    return np.abs(pred - obs)
    #return (pred - obs) ** 2


""" Combine Predictions into Single Prediction """
def combinePredictions(predictions, weights): 
    return np.dot(predictions, weights)