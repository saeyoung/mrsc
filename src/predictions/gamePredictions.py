import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn import linear_model
from sklearn.metrics import r2_score

# dennis libraries
from mrsc.src.predictions import predictionMethods, SLA


""" Compute root mean squared error (l2-norm) """ 
def rmse(pred, true): 
    error = (pred - true) ** 2
    return np.sqrt(np.mean(error))

""" Compute mean absolute error (l1-norm) """
def mae(pred, true):
    return np.mean(np.abs(pred - true))

""" compute correlation """
def corr(pred, true): 
    return pd.Series(pred).corr(pd.Series(true))

""" return average error over window """ 
def errorWindowAvg(pred, true, windowSize, errorType='rmse'):
    predWindow = getWindowAvg(pred, windowSize)
    trueWindow = getWindowAvg(true, windowSize)
    if errorType == 'mae':
        error = mae(trueWindow, predWindow)
    elif errorType == 'r2': 
        error = r2_score(trueWindow, predWindow)
    else:
        error = rmse(trueWindow, predWindow)
    return error

""" compute window average """ 
def getWindowAvg(series, window): 
    seriesWindow = np.array([])
    for i in range(0, len(series)-window, window): 
        windowAvg = np.mean(series[i: i+window])
        seriesWindow = np.append(seriesWindow, windowAvg)
    return seriesWindow 

""" create dictionaries of parameters """ 
def getParamDicts(paramDict, infoDict, featureTypes, labelType):
    # features
    statsWindow = paramDict['statsWindow']
    statsCom = paramDict['statsCom']
    gmWindow = paramDict['gmWindow']
    gmCom = paramDict['gmCom']
    opptWindow = paramDict['opptWindow']
    teamWindow = paramDict['teamWindow']
    teamCom = paramDict['teamCom']
    n = paramDict['n']
    
    # model
    rank = paramDict['rank']
    project = paramDict['project']
    updateType = paramDict['updateType']
    updatePeriod = paramDict['updatePeriod']
    
    # sla 
    slaType = paramDict['type']
    alpha = paramDict['alpha']
    radius = paramDict['radius']
    n_neighbors = paramDict['n_neighbors']
    weights = paramDict['weights']
    algo = paramDict['algo']
    leaf_size = paramDict['leaf_size']
    #f_type = paramDict['f_type']
    #f_params = paramDict['f_params']
    tau = paramDict['tau']
    fit_intercept = paramDict['fit_intercept']
    
    # create features dictionaries
    featuresDict = dict()
    for feature in featureTypes:
        if feature == 'std': 
            featuresDict.update({'std': {'window': statsWindow}})
        if feature == 'mean':
            featuresDict.update({'mean': {'window': statsWindow}})
        if feature == 'ewm': 
            featuresDict.update({'ewm': {'window': statsWindow, 'com': statsCom}})
        if feature == 'gmOutcome': 
            featuresDict.update({'gmOutcome': {'window': gmWindow, 'com': gmCom}})
        if feature == 'oppt': 
            featuresDict.update({'oppt': {'window': opptWindow}})
        if feature == 'delta': 
            featuresDict.update({'delta': {'window': statsWindow, 'com': statsCom}})
        if feature == 'teammates': 
            featuresDict.update({'teammates': {'window': teamWindow, 'n': n, 'com': teamCom}})
        if feature == 'teamLoc':
            featuresDict.update({'teamLoc': {}})
            
    # create labels dictionary     
    if labelType == 'mean':
        labelsDict = {'mean': {'window': statsWindow}}
    elif labelType == 'ewm':
        labelsDict = {'ewm': {'window': statsWindow, 'com': statsCom}}
    else:
        labelsDict = {'none': {}}
    
    # create model dictionary
    modelDict = {'rank': rank, 'project': project, 'updateType': updateType, 'updatePeriod': updatePeriod}
    
    # create info dictionary
    #bufferWindow = np.max([statsWindow, gmWindow, opptWindow])
    bufferWindow = statsWindow
    infoDict.update({'buffer': bufferWindow})
    
    # create sla dictionary
    if slaType == 'knn':
        params = {'n_neighbors': n_neighbors, 'weights': weights, 'algo': algo, 'leaf_size': leaf_size}
    elif slaType == 'rnn':
        params = {'radius': radius, 'weights': weights, 'algo': algo, 'leaf_size': leaf_size}
    elif slaType == 'ridge':
        params = {'alpha': alpha}
    elif slaType == 'lwr':
        params = {'tau': tau, 'fit_intercept': fit_intercept}
        #params = {'f_type': f_type, 'f_params': f_params, 'fit_intercept': fit_intercept}
    else:
        params = {}
    slaDict = {'type': slaType, 'params': params}  
    return infoDict, featuresDict, labelsDict, modelDict, slaDict




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
    for i in range(buffer+1, len(dates)):
    #for i in range(buffer, len(dates)): 
        # baseline predictor using mean of past 'window' games
        datesWindow = dates[i-window: i]  
        windowVals = dfPlayer.loc[dfPlayer.gmDate.isin(datesWindow), metric].values
        pred = np.mean(windowVals)
        preds = np.append(preds, pred)

        # get true value
        currGamePerf = dfPlayer.loc[dfPlayer.gmDate == dates[i], metric].values[0]
        true = np.append(true, currGamePerf)

    return preds, true





















