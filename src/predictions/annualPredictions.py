# third party libraries
import sys, os
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import copy
import pickle
from statsmodels.tsa.arima_model import ARIMA
from sklearn import linear_model
from itertools import combinations, product

# personal libraries
from mrsc.src.model.SVDmodel import SVDmodel
from mrsc.src.model.Target import Target
from mrsc.src.model.Donor import Donor
from mrsc.src.synthcontrol.mRSC import mRSC
from mrsc.src.dataPrep.importData import *
import mrsc.src.utils as utils

# dennis libraries
from mrsc.src.dataPrep import plotData, gameData, annualData
from mrsc.src.predictions import cvxRegression, predictionMethods, SLA

""" 
Predict via MRSC
intuition: use other players (historical and current) to create a synthetic version of 'targetPlayer' to forecast
"""
def mrscPredict(targetPlayer, allPivotedTableDict, donor, pred_interval, metric, pred_metrics,
                      threshold, donorSetup, denoiseSetup, regression_method, verbose):
    
    # create target object
    target = Target(targetPlayer, allPivotedTableDict)
    
    # create mrsc model 
    mrsc = mRSC(donor, target, pred_interval, probObservation=1)
    
    # fit model
    mrsc.fit_threshold(metric, threshold, donorSetup, denoiseSetup, regression_method, verbose)
    
    # predict model for 'pred_metric'
    pred = mrsc.predict()
    pred = pred[pred.index.isin(pred_metrics)]
    
    return pred

"""
Predict via ARIMA
"""
def arimaPredict(train, test, params=(1,1,0)): 
    predictions = list()
    history = [y for y in train]
    for t in range(len(test)): 
        model = ARIMA(history, order=params)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    return predictions

"""
Get predictions from all models and true value
"""
def getPredictions(targetPlayer, allPivotedTableDict, donor, pred_interval, metric, pred_metrics,
                      threshold, donorSetup, denoiseSetup, regression_method, verbose, arima_params, ewm_param):
    # get MRSC prediction
    pred1 = mrscPredict(targetPlayer, allPivotedTableDict, donor, pred_interval, metric, pred_metrics,
                      threshold, donorSetup, denoiseSetup, regression_method, verbose)
    pred1 = pred1.values.flatten()[0]
        
    # get ARIMA prediction
    df = allPivotedTableDict[metric[0]]
    df = df[df.index == targetPlayer]
    df.dropna(axis='columns', inplace=True)
    num_years = df.shape[1] - 1
    train_years = np.arange(num_years)
    test = df.loc[:, num_years].values.flatten()
    
    # w/o ewm
    train = df.loc[:, train_years].values.flatten()
    pred2 = arimaPredict(train, test, arima_params)
    pred2 = pred2[0][0]
    
    # w/ ewm
    train_ewm = df.loc[:, train_years].T.ewm(com=ewm_param).mean().values.flatten()
    pred3 = arimaPredict(train_ewm, test, arima_params)
    pred3 = pred3[0][0]
    
    # concatenate predictions
    pred = np.array([pred1, pred2, pred3]).reshape((1,3))
    
    return pred, test

