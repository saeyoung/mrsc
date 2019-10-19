import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn import linear_model

# dennis libraries
from mrsc.src.predictions import predictionMethods, SLA

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
def forecastSLA(df, teamsDict, player, window, sla, featureMatrix,
    numFeatureTypes, catFeatureTypes, labelType=None, 
    metric='PTS_G', ewmParam=0.5, n=2, project=True): 
    # create prediction and true observations lists
    preds = np.array([])
    true = np.array([])

    # get player specific data
    dfPlayer = df[df.Player == player]
    dates = dfPlayer.gmDate.values

    for i in range(window, len(dates)):
        # get relevant dates
        currDate = dates[i]
        prevDate = dates[i-1]
        datesWindow = dates[i-window: i]

        # get numerical feature
        windowVals = dfPlayer.loc[dfPlayer.gmDate.isin(datesWindow), metric].values
        oppTeam = dfPlayer.loc[dfPlayer.gmDate == currDate, 'opptAbbr'].values[0]
        numFeaturesDict = {'std': {'windowVals': windowVals}, 
                'mean': {'windowVals': windowVals},
                'ewm': {'windowVals': windowVals, 'ewmParam': ewmParam},
                'opp': {'teamsDict': teamsDict, 'oppTeam': oppTeam, 'date': currDate}}
        numFeatures = SLA.getNumFeatures(numFeaturesDict, numFeatureTypes)
        # project numerical feature onto de-noised space
        if project: 
            numFeatures = projectFeatures(featureMatrix, numFeatures)

        # get categorical feature
        catFeaturesDict = {'teamLoc': {'df': dfPlayer, 'date': currDate}, 
                'gameOutcome': {'df': dfPlayer, 'date': datesWindow, 'ewm': ewmParam},
                'teammates': {'df': df, 'player': player, 'date': currDate, 'metric': metric, 'n': n}}
        catFeatures = SLA.getCategoricalFeatures(catFeaturesDict, catFeatureTypes)

        # concatenate features 
        feature = np.append(numFeatures, catFeatures)

        # forecast 
        labelDict = {'mean': {'windowVals': windowVals}, 
                'ewm': {'windowVals': windowVals, 'ewmParam': ewmParam}} 
        pred = sla.model.predict(feature.reshape(1, feature.shape[0]))[0]
        pred = SLA.getLabel(pred, labelDict, labelType, train=False)
        preds = np.append(preds, pred)   

        # get true labels
        currGamePerf = dfPlayer.loc[dfPlayer.gmDate == currDate, metric].values[0]
        true = np.append(true, currGamePerf)  
    
    return preds, true

""" Train and test via SLA """ 
def SLATrainTest(slaType, paramDict, dfTrain, dfTest, teamsTrainDict, teamsTestDict, player, 
                numFeatureTypes, catFeatureTypes, labelType=None, metric='PTS_G', project=True):
    # extract hyper-parameters
    windowSize = paramDict['windowSize']
    rank = paramDict['rank']
    ewmParam = paramDict['ewm']
    n = paramDict['n']
        
    """ Train Stage """
    # construct (featureMatrix, labels)
    featureMatrix, labels = SLA.constructFeatureMatrixLabels(dfTrain, teamsTrainDict, player, windowSize,
                                                             numFeatureTypes, catFeatureTypes, labelType,
                                                             metric, ewmParam, n)
        
    # de-noise featureMatrix
    numHSVTFeatures = len(numFeatureTypes)
    featureMatrix[:, :numHSVTFeatures] = hsvt(featureMatrix[:, :numHSVTFeatures], rank=rank)
        
    # create SLA forecaster and fit
    sla = SLA.SLAForecast(slaType, paramDict)
    sla.fit(featureMatrix, labels)
        
    """ Test Stage """
    featureMatrixProj = featureMatrix[:, :numHSVTFeatures]
    preds, true = forecastSLA(dfTest, teamsTestDict, player, windowSize, sla, featureMatrixProj, 
                                numFeatureTypes, catFeatureTypes, labelType, 
                                metric, ewmParam, n, project)
    
    return preds, true

""" Forecast via ARIMA """
def forecastARIMA(df, dates, buffer, metric, history, paramsARIMA):
    preds = np.array([])
    true = np.array([])
    
    for i in range(buffer, len(dates)): 
        # forecast  
        model = ARIMA(history, order=paramsARIMA)
        model_fit = model.fit(disp=0)
        pred = model_fit.forecast()[0][0]
        preds = np.append(preds, pred) 
        
        # update history
        currDate = dates[i]
        currGamePerf = df.loc[df.gmDate == currDate, metric].mean()
        true = np.append(true, currGamePerf)
        history = np.append(history, currGamePerf) 

    return preds, true

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
    preds, true = forecastARIMA(dfTestPlayer, datesTest, buffer, metric, history, arimaParams)
    
    return preds, true


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


""" Baseline predictor using past 'window' values """ 
def baselineForecast(df, player, window, buffer=0, metric='PTS_G'):
    preds = np.array([])
    true = np.array([])

    # get player specific information
    dfPlayer = df[df.Player == player]
    dates = dfPlayer.gmDate.values 

    # forecast
    for i in range(buffer, len(dates)): 
        # baseline predictor using mean of past 'window' games
        datesWindow = dates[i-window: i]  
        windowVals = dfPlayer.loc[dfPlayer.gmDate.isin(datesWindow), metric].values
        pred = np.mean(windowVals)
        preds = np.append(preds, pred)

        # get true value
        currGamePerf = dfPlayer.loc[dfPlayer.gmDate == dates[i], metric].values[0]
        true = np.append(true, currGamePerf)

    return preds, true





















