

""" OLD """ 
""" Forecast via Supervised Learning Method """
def forecastSLA_(df, dfAll, player, window, dates, catFeatureTypes, sla, featureMatrix, metric='PTS_G', ewmParam=0.5, n=2): 
    predsSLA = np.array([])
    true = np.array([])
    
    for i in range(window, len(dates)):
        # get current game information
        currDate = dates[i]
        currGamePerf = df.loc[df.gmDate == currDate, metric].values[0]
        true = np.append(true, currGamePerf)

        # get numerical feature
        datesWindow = dates[i-window: i]
        windowVals = df.loc[df.gmDate.isin(datesWindow), metric].values
        numFeatures = SLA.getWindowFeatures(windowVals, ewmParam)

        # project numerical feature onto de-noised space
        numFeatures = projectFeatures(featureMatrix, numFeatures)

        # get categorical feature
        prevDate = dates[i-1]
        catFeaturesDict = {'teamLoc': {'df': df, 'date': currDate}, 
                'gameOutcome': {'df': df, 'date': prevDate},
                'teammates': {'df': dfAll, 'player': player, 'date': currDate, 'metric': metric, 'n': n}}
        catFeatures = SLA.getCategoricalFeatures(catFeaturesDict, catFeatureTypes)

        # concatenate features 
        feature = np.append(numFeatures, catFeatures)

        # forecast
        predSLA = sla.model.predict(feature.reshape(1, feature.shape[0]))[0]
        predsSLA = np.append(predsSLA, predSLA)      
    
    return predsSLA, true

def SLATrainTest_(slaType, paramDict, dfTrain, dfAllTrain, datesTrain, dfTest, dfAllTest, datesTest, player, catFeatureTypes, metric='PTS_G'): 
    # extract hyper-parameters
    windowSize = paramDict['windowSize']
    rank = paramDict['rank']
    ewmParam = paramDict['ewm']
    n = paramDict['n']
        
    """ Train Stage """
    # construct (featureMatrix, labels)
    featureMatrix, labels = SLA.constructFeatureMatrixLabels(dfTrain, dfAllTrain, player,
                                                             windowSize, datesTrain, catFeatureTypes,
                                                             metric, ewmParam, n)
        
    # de-noise featureMatrix
    numHSVTFeatures = featureMatrix.shape[1] - len(catFeatureTypes)
    featureMatrix[:, :numHSVTFeatures] = hsvt(featureMatrix[:, :numHSVTFeatures], rank=rank)
        
    # create SLA forecaster and fit
    sla = SLA.SLAForecast(slaType, paramDict)
    sla.fit(featureMatrix, labels)
        
    """ CV Stage """
    # SLA forecast
    predsSLA, true = forecastSLA(dfTest, dfAllTest, player, windowSize, datesTest,
                                catFeatureTypes, sla, featureMatrix[:, :numHSVTFeatures], 
                                metric, ewmParam, n)
    
    return predsSLA, true



""" Forecast via Supervised Learning Method """
"""def forecastSLA(df, dates, window, metric, sla, featureMatrix, ewmParam=0.5): 
    predsSLA = np.array([])
    true = np.array([])
    
    for i in range(window, len(dates)):
        # get current game information
        currDate = dates[i]
        currGamePerf = df.loc[df.gmDate == currDate, metric].mean()
        true = np.append(true, currGamePerf)
        teamLoc = df.loc[df.gmDate == currDate, 'teamLoc']
        if (teamLoc == 'Home').bool(): 
            teamLoc = 1
        else:
            teamLoc = -1

        # get previous game information
        prevDate = dates[i-1]
        prevGameOutcome = df.loc[df.gmDate == prevDate, 'teamRslt']
        if (prevGameOutcome == 'Win').bool():
            prevGameOutcome = 1
        else:
            prevGameOutcome = -1

        # create feature and forecast
        datesWindow = dates[i-window: i]
        windowVals = df.loc[df.gmDate.isin(datesWindow), metric].values

        gameInfo = np.array([prevGameOutcome, teamLoc])
        feature = SLA.getWindowFeature(windowVals, ewmParam)

        # project feature onto de-noised space
        feature = projectFeature(featureMatrix, feature)

        feature = np.append(feature, gameInfo)
        predSLA = sla.model.predict(feature.reshape(1, feature.shape[0]))[0]
        predsSLA = np.append(predsSLA, predSLA)      
    
    return predsSLA, true"""

""" OLD """
def constructFeatureMatrixLabels_(df, dfAll, player, window, dates, catFeatureTypes, metric='PTS_G', ewmParam=0.5, n=2): 
    numSamples = len(dates) - window
    labels = np.array([])
    featureMatrix = np.array([])
        
    for i in range(window, len(dates)): 
        # get current date and value (use as label)
        currDate = dates[i]
        currVal = df.loc[df.gmDate == currDate, metric].values[0]

        # get numerical features
        datesWindow = dates[i-window: i]
        windowVals = df.loc[df.gmDate.isin(datesWindow), metric].values
        numFeatures = getWindowFeatures(windowVals, ewmParam)

        # get categorical feature
        prevDate = dates[i-1]
        catFeaturesDict = {'teamLoc': {'df': df, 'date': currDate}, 
                'gameOutcome': {'df': df, 'date': prevDate},
                'teammates': {'df': dfAll, 'player': player, 'date': currDate, 'metric': metric, 'n': n}}
        catFeatures = getCategoricalFeatures(catFeaturesDict, catFeatureTypes)

        # concatenate features
        feature = np.append(numFeatures, catFeatures)

        # append to feature matrix and label vector
        featureMatrix = np.vstack([featureMatrix, feature]) if featureMatrix.size else feature
        labels = np.append(labels, currVal)
    return featureMatrix, labels


# THIS IS OLD! (below)

"""def constructFeatureMatrixLabels(df, window, dates, metric, ewmParam=0.5): 
    numSamples = len(dates) - window
    labels = np.array([])
    featureMatrix = np.array([])
        
    for i in range(window, len(dates)): 
        # get current date and value (used as label)
        currDate = dates[i]
        currVal = df.loc[df.gmDate == currDate, metric].values[0]

        # get home/away information for current date
        teamLoc = df.loc[df.gmDate == currDate, 'teamLoc']
        if (teamLoc == 'Home').bool(): 
            teamLoc = 1
        else:
            teamLoc = -1

        # get previous dates of size 'window' and values
        datesWindow = dates[i-window: i]
        windowVals = df.loc[df.gmDate.isin(datesWindow), metric].values
        
        # get previous game outcome 
        prevDate = dates[i-1]
        prevGameOutcome = df.loc[df.gmDate==prevDate, 'teamRslt']
        if (prevGameOutcome == 'Win').bool():
            prevGameOutcome = 1
        else:
            prevGameOutcome = -1
        
        # create features from window values
        gameInfo = np.array([prevGameOutcome, teamLoc])
        feature = getWindowFeature(windowVals, ewmParam)
        feature = np.append(feature, gameInfo)

        # append to feature matrix and label vector
        featureMatrix = np.vstack([featureMatrix, feature]) if featureMatrix.size else feature
        labels = np.append(labels, currVal)
    return featureMatrix, labels
"""
