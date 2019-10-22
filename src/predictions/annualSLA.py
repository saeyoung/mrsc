import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import math
import heapq

# personal libraries
from mrsc.src.predictions import predictionMethods, gamePredictions, SLA 

""" hard singular value thresholding """ 
def hsvt(X, rank): 
    X_hsvt = X
    if rank > 0: 
        u, s, v = np.linalg.svd(X, full_matrices=False)
        s[rank:].fill(0)
        X_hsvt = np.dot(u*s, v)
    return X_hsvt 

""" project feature onto de-noised feature space """
def projectFeatures(featureMatrix, feature): 
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(featureMatrix.T, feature)
    return np.dot(featureMatrix.T, regr.coef_)

def getTeam(df, year): 
    return df.loc[df.Year == year, 'Tm'].values[0]

def getTeammates(df, player, team, year): 
    teammates = df.loc[(df.Tm == team) & (df.Year == year), 'Player'].tolist()
    teammates.remove(player)
    return teammates

def getWindowVals(df, years, metric='PTS'): 
    return df.loc[df.Year.isin(years), metric].values

"""def getValidYears(df, testYear): 
    return df.loc[df.Year <= testYear, 'Year'].values"""

def getYears(df): 
    return df.Year.values

def getYearsWindow(years, i, window): 
    return years[i-window: i] if i >= window else years[:i]

def getLabel(df, year, metric='PTS'):
    return df.loc[df.Year == year, metric].values[0]

def topNTeammatesPerf(df, teammates, year, window, n=1, com=0.2, metric='PTS'):
    perfList = np.array([])
    for player in teammates: 
        # get teammates info
        dfPlayer = df.loc[df.Player == player]
        
        # get valid years
        years = dfPlayer.loc[dfPlayer.Year < year, 'Year'].values
        
        # get performance over window
        yearsWindow = years[-window:]
        windowVals = getWindowVals(dfPlayer, yearsWindow, metric)

        # get ewm performance
        ewmPerf = predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values
        ewmPerf = ewmPerf[-1] if ewmPerf.size else 0
        
        # append to performance list
        perfList = np.append(perfList, ewmPerf)
    return heapq.nlargest(n, perfList)

def getFeature(years, i, dfPlayer, dfLeague, featuresDict, metric='PTS'): 
    featureVec = np.array([])
    
    for feature, featureParams in featuresDict.items(): 
        if feature == 'std':
            # unpack parameters
            window = featureParams['window']
            
            # get performance over relevant years
            yearsWindow = getYearsWindow(years, i, window)
            windowVals = getWindowVals(dfPlayer, yearsWindow, metric)
            
            # compute standard deviation
            featureVal = np.std(windowVals)
            
            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            
        if feature == 'mean':
            # unpack parameters
            window = featureParams['window']
            
            # get performance over relevant years
            yearsWindow = getYearsWindow(years, i, window)
            windowVals = getWindowVals(dfPlayer, yearsWindow, metric)
            
            # compute mean 
            featureVal = np.mean(windowVals)
            
            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            
        if feature == 'ewm': 
            # unpack parameters
            window = featureParams['window']
            com = featureParams['com']
            
            # get performance over relevant years
            yearsWindow = getYearsWindow(years, i, window)
            windowVals = getWindowVals(dfPlayer, yearsWindow, metric)
            
            # apply ewm and get most recent value
            ewm_windowVals = predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values
            featureVal = ewm_windowVals[-1]
            
            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            
        if feature == 'teammates': 
            # unpack parameters
            window = featureParams['window']
            com = featureParams['com']
            n = featureParams['n']

            # get current year and player
            currYear = years[i]
            player = dfPlayer.Player.values[0]
            
            # get current team and teammates
            team = getTeam(dfPlayer, currYear)
            teammates = getTeammates(dfLeague, player, team, currYear)
            featureVal = topNTeammatesPerf(dfLeague, teammates, currYear, window, n, com, metric)
            
            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
    return featureVec
    
def updateLabel(label, years, i, df, labelsDict, metric='PTS', train=True):
    # unpack label info
    labelType = list(labelsDict.keys())[0]
    labelParams = list(labelsDict.values())[0]

    # label = perf - mean(window)
    if labelType == 'mean':
        # get parameters
        window = labelParams['window']
            
        # get performance over relevant years
        yearsWindow = getYearsWindow(years, i, window)
        windowVals = getWindowVals(df, yearsWindow, metric)
            
        # compute mean 
        labelShift = np.mean(windowVals)

    # label = perf - ewm(window)[-1]
    elif labelType == 'ewm': 
        # unpack parameters
        window = labelParams['window']
        com = labelParams['com']
            
        # get performance over relevant years
        yearsWindow = getYearsWindow(years, i, window)
        windowVals = getWindowVals(df, yearsWindow, metric)
            
        # apply ewm and get most recent value
        ewm_windowVals = predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values
        labelShift = ewm_windowVals[-1]

    # label = perf
    else: 
        labelShift = 0
    return label - labelShift if train else label + labelShift

    
def getFeaturesLabels(infoDict, dataDict, featuresDict, labelsDict, modelDict):
    # unpack info
    player = infoDict['player']
    metric = infoDict['metric']
    bufferWindow = infoDict['buffer']
    
    # unpack data
    dfLeague = dataDict['df']
    
    # get player specific data
    dfPlayer = dfLeague.loc[dfLeague.Player == player]
    years = dfPlayer.Year.values
    
    # initialize feature matrix and labels
    features = np.array([])
    labels = np.array([])
    
    # iterate through every year
    for i in range(bufferWindow, len(years)): 
        # construct feature
        feature = getFeature(years, i, dfPlayer, dfLeague, featuresDict, metric)
        
        # construct label
        currYear = years[i]
        label = getLabel(dfPlayer, currYear, metric)
        label = updateLabel(label, years, i, dfPlayer, labelsDict, metric, train=True)
        
        # append
        features = np.vstack([features, feature]) if features.size else feature
        labels = np.append(labels, label)

    # unpack model info
    rank = modelDict['rank']
    features = hsvt(features, rank=rank)
    return features, labels 

def trainSLA(infoDict, dataDict, featuresDict, labelsDict, modelDict, slaDict): 
    # construct features and labels
    features, labels = getFeaturesLabels(infoDict, dataDict, featuresDict, labelsDict, modelDict)

    # unpack sla parameters
    slaType = slaDict['type']
    slaParams = slaDict['params']
    slaDict.update({'features': features, 'labels': labels})

    # create and fit model to data
    sla = SLA.SLAForecast(slaType, slaParams)
    sla.fit(features, labels)

    # update sla dictionary
    slaDict.update({'model': sla})

def testSLA(infoDict, dataDict, featuresDict, labelsDict, modelDict, slaDict):
    # unpack info 
    player = infoDict['player']
    metric = infoDict['metric']
    bufferWindow = infoDict['buffer']

    # unpack data
    dfLeague = dataDict['df']

    # unpack sla info
    sla = slaDict['model']
    features = slaDict['features']
    labels = slaDict['labels']

    # unpack model info
    rank = modelDict['rank']
    project = modelDict['project']

    # get player specific data (dataframe + dates played)
    dfPlayer = dfLeague[dfLeague.Player == player]
    years = dfPlayer.Year.values

    # initialize
    preds = np.array([])
    true = np.array([])

    """ PREDICT """ 
    # get previous window years
    i = len(years) - 1

    # get feature
    feature = getFeature(years, i, dfPlayer, dfLeague, featuresDict, metric)
    slaDict['test_feature'] = feature
    if project: 
        feature = projectFeatures(features, feature)
    slaDict['test_featureProj'] = feature 

    # get forecast
    pred = sla.predict(feature.reshape(1, feature.shape[0]))[0]
    pred = updateLabel(pred, years, i, dfPlayer, labelsDict, metric, train=False)
    preds = np.append(preds, pred)

    # get true value
    currYear = years[i]
    label = getLabel(dfPlayer, currYear, metric)
    true = np.append(true, label)
    return preds, true





















    