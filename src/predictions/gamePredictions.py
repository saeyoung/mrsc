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
from mrsc.src.predictions import cvxRegression, predictionMethods, SLA, annualPredictions

""" hard singular value thresholding """ 
def hsvt(X, rank): 
    u, s, v = np.linalg.svd(X, full_matrices=False)
    s[rank:].fill(0)
    return np.dot(u*s, v)

""" project feature onto de-noised feature space """
def projectFeatures(featureMatrix, feature): 
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(featureMatrix.T, feature)
    return np.dot(featureMatrix.T, regr.coef_)

""" Forecast via Supervised Learning Method """
def forecastSLA(df, player, window, catFeatureTypes, sla, featureMatrix, metric='PTS_G', ewmParam=0.5, n=2): 
    predsSLA = np.array([])
    true = np.array([])

    # get player specific data
    dfPlayer = df[df.Player == player]
    dates = dfPlayer.gmDate.values

    for i in range(window, len(dates)):
        # get current game information
        currDate = dates[i]
        currGamePerf = dfPlayer.loc[dfPlayer.gmDate == currDate, metric].values[0]
        true = np.append(true, currGamePerf)

        # get numerical feature
        datesWindow = dates[i-window: i]
        windowVals = dfPlayer.loc[dfPlayer.gmDate.isin(datesWindow), metric].values
        numFeatures = SLA.getWindowFeatures(windowVals, ewmParam)

        # project numerical feature onto de-noised space
        numFeatures = projectFeatures(featureMatrix, numFeatures)

        # get categorical feature
        prevDate = dates[i-1]
        catFeaturesDict = {'teamLoc': {'df': dfPlayer, 'date': currDate}, 
                'gameOutcome': {'df': dfPlayer, 'date': prevDate},
                'teammates': {'df': df, 'player': player, 'date': currDate, 'metric': metric, 'n': n}}
        catFeatures = SLA.getCategoricalFeatures(catFeaturesDict, catFeatureTypes)

        # concatenate features 
        feature = np.append(numFeatures, catFeatures)

        # forecast
        predSLA = sla.model.predict(feature.reshape(1, feature.shape[0]))[0]
        predsSLA = np.append(predsSLA, predSLA)      
    
    return predsSLA, true

""" Train and test via SLA """ 
def SLATrainTest(slaType, paramDict, dfTrain, dfTest, player, catFeatureTypes, metric='PTS_G'):
    # extract hyper-parameters
    windowSize = paramDict['windowSize']
    rank = paramDict['rank']
    ewmParam = paramDict['ewm']
    n = paramDict['n']
        
    """ Train Stage """
    # construct (featureMatrix, labels)

    featureMatrix, labels = SLA.constructFeatureMatrixLabels(dfTrain, player,
                                                             windowSize, catFeatureTypes,
                                                             metric, ewmParam, n)
        
    # de-noise featureMatrix
    numHSVTFeatures = featureMatrix.shape[1] - len(catFeatureTypes)
    featureMatrix[:, :numHSVTFeatures] = hsvt(featureMatrix[:, :numHSVTFeatures], rank=rank)
        
    # create SLA forecaster and fit
    sla = SLA.SLAForecast(slaType, paramDict)
    sla.fit(featureMatrix, labels)
        
    """ CV Stage """
    # SLA forecast
    featureMatrixProj = featureMatrix[:, :numHSVTFeatures]
    predsSLA, true = forecastSLA(dfTest, player, windowSize,
                                catFeatureTypes, sla, featureMatrixProj, 
                                metric, ewmParam, n)
    
    return predsSLA, true

""" Forecast via ARIMA """
def forecastARIMA(df, dates, buffer, metric, history, paramsARIMA):
    predsARIMA = np.array([])
    true = np.array([])
    
    for i in range(buffer, len(dates)): 
        # forecast  
        model = ARIMA(history, order=paramsARIMA)
        model_fit = model.fit(disp=0)
        predARIMA = model_fit.forecast()[0][0]
        predsARIMA = np.append(predsARIMA, predARIMA) 
        
        # update history
        currDate = dates[i]
        currGamePerf = df.loc[df.gmDate == currDate, metric].mean()
        true = np.append(true, currGamePerf)
        history = np.append(history, currGamePerf) 

    return predsARIMA, true

""" Train and Test via ARIMA """ 
def ARIMATrainTest(paramDict, dfTrain, dfTest, player, buffer, metric='PTS_G'): 
    # get player specific information
    dfTrainPlayer = dfTrain[dfTrain.Player == player]
    dfTestPlayer = dfTest[dfTest.Player == player]
    datesTest = dfTestPlayer.gmDate.values

    """ extract hyper-parameters """
    d = int(paramDict['d'])
    p = int(paramDict['p'])
    q = int(paramDict['q'])
    arimaParams = (d, p, q)
        
    # forecast
    trainHistory = dfTrainPlayer[metric].values
    testHistory = dfTestPlayer.loc[dfTestPlayer.gmDate.isin(datesTest[:buffer]), metric].values
    history = np.append(trainHistory, testHistory)
    predsARIMA, true = forecastARIMA(dfTestPlayer, datesTest, buffer, metric, history, arimaParams)
    
    return predsARIMA, true


""" Forecast via combined estimators """
def forecastCombined(true, preds, numPredictors, learningRate=0.4):
    predsCombined = np.array([])
    weights = np.ones(numPredictors) / numPredictors
    
    for i in range(len(true)): 
        # combine predictions
        pred = preds[i]
        predCombined = predictionMethods.combinePredictions(pred, weights)
        predsCombined = np.append(predsCombined, predCombined)
        
        # get true value
        obs = np.ones(numPredictors) * true[i]
        
        # update weights
        weights = predictionMethods.MWU(pred, obs, weights, learningRate)
        
    return predsCombined

