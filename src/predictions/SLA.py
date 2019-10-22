import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR
import math
import heapq

# personal libraries
from mrsc.src.predictions import predictionMethods, gamePredictions, local_regression

""" Supervised Learning Forecasting """
class SLAForecast: 
    def __init__(self, method, params):
        self.method = method
        self.params = params
        self.model = []
        
    def fit(self, featureMatrix, labels):
        # ridge regression
        if self.method.lower() == 'ridge': 
            alpha = self.params['alpha']
            self.model = linear_model.Ridge(fit_intercept=False, alpha=alpha)

        # random forest regression
        elif self.method.lower() == 'randomforest':
            n_estimators = self.params['n_estimators']
            max_depth = self.params['max_depth']
            min_samples_split = self.params['min_samples_split']
            min_samples_leaf = self.params['min_samples_leaf']
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                min_samples_leaf=min_samples_leaf, min_samples_split = min_samples_split)

        # support vector regression
        elif self.method.lower() == 'svr': 
            C = self.params['C']
            gamma = self.params['gamma']
            degree = self.params['degree']
            epsilon = self.params['epsilon']
            self.model = SVR(C=C, gamma=gamma, degree=degree, epsilon=epsilon)

        # locally weighted regression
        elif self.method.lower() == 'lwr': 
            #f_type = self.params['f_type']
            #f_params = self.params['f_params']
            tau = self.params['tau']
            fit_intercept = self.params['fit_intercept']
            self.model = local_regression.LWRegressor(tau=tau, fit_intercept=fit_intercept) 

        # radius neighbors regression
        elif self.method.lower() == 'rnn':
            radius = self.params['radius']
            weights = self.params['weights']
            algo = self.params['algo']
            leaf_size = self.params['leaf_size']
            self.model = RadiusNeighborsRegressor(radius=radius, weights=weights,
                                                    algorithm=algo, leaf_size=leaf_size)

        # k-nearest neighbors regression
        elif self.method.lower() == 'knn': 
            n_neighbors = self.params['n_neighbors']
            weights = self.params['weights']
            algo = self.params['algo']
            leaf_size = self.params['leaf_size']
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights,
                                                algorithm=algo, leaf_size=leaf_size)

        # linear regression
        else: 
            self.model = linear_model.LinearRegression(fit_intercept=False)

        # fit model to data
        self.model.fit(featureMatrix, labels)
            
    def predict(self, feature):
        return self.model.predict(feature)

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

""" Create dictionary of points scored by team and against team """ 
def getTeamOppPTSDict(dfMaster, teams): 
    teamsDict = {team: pd.DataFrame() for team in teams} 
    for team in teams: 
        # get team info
        df = dfMaster.loc[dfMaster.teamAbbr == team]
        dates = df.gmDate.unique()

        # build dataframe dates
        dfTeam = pd.DataFrame(columns=['gmDate', 'PTS', 'OppTeam', 'OppPTS'])
        dfTeam.gmDate = dates

        for date in dates: 
            # input points scored by team on date
            dfTeam.loc[dfTeam.gmDate == date, 'PTS'] = getTeamPerf(df, date)

            # get opposing team info
            oppTeam = df.loc[df.gmDate == date, 'opptAbbr'].values[0]
            dfOpp = dfMaster.loc[dfMaster.teamAbbr == oppTeam]

            # input points scored by opposing team
            dfTeam.loc[dfTeam.gmDate == date, 'OppTeam'] = oppTeam
            dfTeam.loc[dfTeam.gmDate == date, 'OppPTS'] = getTeamPerf(dfOpp, date)

        teamsDict[team] = dfTeam
    return teamsDict

""" Get Team PTS on date """
def getTeamPerf(dfTeam, date, metric='PTS_G'): 
    df = dfTeam.loc[dfTeam.gmDate == date]
    return df[metric].sum() 

""" Get points per game scored by team's opposition """ 
def getTeamOppPerf(teamsDict, team, date, window=-1, metric='OppPTS'): 
    # get team data
    dfTeam = teamsDict[team]
    windowVals = dfTeam.loc[dfTeam.gmDate < date, metric].values
    # set to zero if empty windowVals = no games played
    if window == -1:
        # get entire team's opponents' histories 
        oppPerf = windowVals.mean() if windowVals.size else 0 
    else: 
        oppPerf = windowVals[-window:].mean() if windowVals.size else 0
    return oppPerf 

""" Get points per game scored by team's opposition averaged across all teams """ 
def getLeagueOppPerf(teamsDict, date, window, metric='OppPTS'):
    league = list(teamsDict.keys())
    leagueOppPerf = np.array([])
    for team in league: 
        # get team's opponents' average performance over window (don't append if team hasn't played yet)
        teamOppPerf = getTeamOppPerf(teamsDict, team, date, window, metric)
        leagueOppPerf = np.append(leagueOppPerf, teamOppPerf) if teamOppPerf else leagueOppPerf 
    # return 0 if empty leagueOppPerf = no games played
    return leagueOppPerf.mean() if leagueOppPerf.size else 0

""" Get Top n Players on Team (wrt Metric) by Date """
def getTopTeamPlayers(df, team, metric, date, n=2):
    # get team
    dfTeam = df[df.teamAbbr==team]
    
    # look at team performance up to (but not including) current game
    dfTeam = dfTeam[dfTeam.gmDate < date]
    dfTeam = dfTeam.groupby('Player', as_index=False)[metric].mean()
    
    # get top n players thus far
    topPlayers = dfTeam.nlargest(n, columns=[metric])['Player'].values
    return topPlayers

""" Checks if any of Player's top teammates are missing on game(date) """
def checkTeammates(df, player, date, metric='PTS_G', n=2):
    # get dataframe of player
    dfPlayer = df[df.Player == player]

    # get current team
    currTeam = dfPlayer.loc[dfPlayer.gmDate == date, 'teamAbbr'].values[0]
    
    # get teammates on current team
    teammates = df.loc[(df.teamAbbr == currTeam) & (df.gmDate == date), 'Player'].values.tolist()
    
    # get top n players thus far
    topTeammates = getTopTeamPlayers(df, currTeam, metric, date, n).tolist()
    
    # remove self if applicable
    if player in topTeammates: 
        topTeammates.remove(player)
        
    # check if top teammates are playing
    topPlayingTeammates = list(set(topTeammates) & set(teammates)) 
    return 1 if len(topPlayingTeammates) < len(topTeammates) else -1

""" get teammates on date """
def getTeammates(df, player, team, date):
    teammates = df.loc[(df.teamAbbr == team) & (df.gmDate == date), 'Player'].values.tolist()
    teammates.remove(player)
    return teammates

""" get game outcome (Win/Loss) """
def getGameOutcomes(df, dates): 
    results = df.loc[df.gmDate.isin(dates), 'teamRslt'].values
    return np.array([1 if result=='Win' else 0 for result in results])

""" get team location (Home/Away) """
def getTeamLoc(df, date): 
    teamLoc = df.loc[df.gmDate == date, 'teamLoc'].values[0]
    return 1 if teamLoc == 'Home' else -1

""" Get game performance according to metric """ 
def getGamePerf(df, date, metric='PTS_G'): 
    return df.loc[df.gmDate == date, metric].values[0]

""" Get Window Values based on metric """ 
def getWindowVals(df, dates, metric='PTS_G'): 
    return df.loc[df.gmDate.isin(dates), metric].values

""" get dates within window size """ 
def getDatesWindow(dates, i, window):
    return dates[i-window: i] if i >= window else dates[:i]

""" get team on date """
def getTeam(df, date):
    return df.loc[df.gmDate == date, 'teamAbbr'].values[0]

""" get opposing team on date """
def getOppTeam(df, date): 
    return df.loc[df.gmDate == date, 'opptAbbr'].values[0]

""" get ewm game perf """
def getEWMGamePerf(df, dates, i, window, com=0.3, metric='PTS_G'):
    # get performance over window
    datesWindow = getDatesWindow(dates, i, window)
    windowVals = getWindowVals(df, datesWindow, metric)

    # return ewm performance
    return predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values

""" get performance of top n teammates over window after applying ewm """ 
def topNTeammatesPerf(df, teammates, date, window, n=2, com=0.3, metric='PTS_G'):
    perfList = np.array([])
    for player in teammates: 
        # get teammate info
        dfPlayer = df[df.Player == player]
        
        # get valid dates
        dates = dfPlayer.loc[dfPlayer.gmDate < date, 'gmDate'].values
        
        # get performance over window
        datesWindow = dates[-window:]
        windowVals = getWindowVals(dfPlayer, datesWindow, metric) 
        
        # get ewm performance
        ewmPerf = predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values
        ewmPerf = ewmPerf[-1] if ewmPerf.size else 0
        
        # append to performance list
        perfList = np.append(perfList, ewmPerf)
    return heapq.nlargest(n, perfList)

""" Return Feature Vector """ 
def getFeature(dates, i, dfPlayer, dfLeague, teamsDict, featuresDict, metric='PTS_G'): 
    # initialize
    featureVec = np.array([])
    num_hsvt_features = 0

    for feature, featureParams in featuresDict.items(): 
        # compute standard deviation (std) of window values
        if feature == 'std':
            # get parameters 
            window = featureParams['window']

            # compute standard deviation of performance over relevant dates
            datesWindow = getDatesWindow(dates, i, window)
            windowVals = getWindowVals(dfPlayer, datesWindow, metric)
            featureVal = np.std(windowVals) if windowVals.size else 0 

            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            num_hsvt_features += 1

        # compute mean of window values
        if feature == 'mean': 
            # get parameters 
            window = featureParams['window']

            # compute average performance over relevant dates
            datesWindow = getDatesWindow(dates, i, window)
            windowVals = getWindowVals(dfPlayer, datesWindow, metric)
            featureVal = np.mean(windowVals) if windowVals.size else 0
            
            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            num_hsvt_features += 1

        # get performance of most recent game after applying ewm on window values
        if feature == 'ewm': 
            # get parameters 
            window = featureParams['window']
            com = featureParams['com']

            # compute performance of relevant dates & apply ewm
            ewm_windowVals = getEWMGamePerf(dfPlayer, dates, i, window, com=com, metric='PTS_G')
            
            # get most recent game performance 
            featureVal = ewm_windowVals[-1]

            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            num_hsvt_features += 1

        # get opponent information 
        if feature == 'oppt': 
            # get parameters
            window = featureParams['window']

            # get opponent on game date
            currDate = dates[i]
            oppTeam = getOppTeam(dfPlayer, currDate)

            # get average performance of opposing teams allowed by oppTeam & league
            oppTeamOppPerf = getTeamOppPerf(teamsDict, oppTeam, currDate, window) 
            leagueOppPerf = getLeagueOppPerf(teamsDict, currDate, window) 
            # if leagueOppPerf = 0 (no games played yet) then set featureVal = 1
            featureVal = oppTeamOppPerf / leagueOppPerf if leagueOppPerf else 1

            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            num_hsvt_features += 1

        # get outcome of most recent game after applying ewm on window values
        if feature == 'gmOutcome':
            # get parameters 
            window = featureParams['window']
            com = featureParams['com']

            # compute outcomes of games during dates window & apply ewma
            datesWindow = getDatesWindow(dates, i, window)
            gmOutcomes = getGameOutcomes(dfPlayer, datesWindow)
            gmOutcomes = predictionMethods.applyEWMA(pd.Series(gmOutcomes), param=com).values
            
            # get most recent game outcome
            featureVal = gmOutcomes[-1]

            # append to feature vector
            featureVec = np.append(featureVec, featureVal) 
            num_hsvt_features += 1

        # get teammate information 
        if feature == 'teammates': 
            # get parameters
            window = featureParams['window']
            com = featureParams['com']
            n = featureParams['n']

            # get current date and player
            currDate = dates[i]
            player = dfPlayer.Player.values[0]

            # get current team and teammates
            team = getTeam(dfPlayer, currDate)
            teammates = getTeammates(dfLeague, player, team, currDate)
            featureVal = topNTeammatesPerf(dfLeague, teammates, currDate, window, n, com, metric)

            # append to feature vector
            featureVec = np.append(featureVec, featureVal)
            num_hsvt_features += 1

        # get delta performance between previous game and prior performances
        if feature == 'delta': 
            # get parameters
            window = featureParams['window']
            com = featureParams['com']

            # get previous performance 
            prevDate = dates[i-1]
            prevGmPerf = getGamePerf(dfPlayer, prevDate)

            # get performance over window of games prior to previous date
            datesWindow = getDatesWindow(dates, i-1, window)
            windowVals = getWindowVals(dfPlayer, datesWindow, metric)

            # get difference in performances
            #featureVal = prevGmPerf - np.mean(windowVals)
            featureVal = prevGmPerf - windowVals[-1]

            # append to feature vector
            featureVec = np.append(featureVec, featureVal) 
            num_hsvt_features += 1

        # get team location on game date
        if feature == 'teamLoc':
            # get team location on game date
            currDate = dates[i]
            featureVal = getTeamLoc(dfPlayer, currDate)

            # append to feature vector
            featureVec = np.append(featureVec, featureVal) 

    return featureVec, num_hsvt_features

""" Update Label """ 
def updateLabel(label, dates, i, dfPlayer, labelsDict, metric='PTS_G', train=True): 
    # unpack label info
    labelType = list(labelsDict.keys())[0]
    labelParams = list(labelsDict.values())[0]

    # label = perf - mean(window)
    if labelType == 'mean': 
        # get parameters 
        window = labelParams['window']

        # get average performance over window
        datesWindow = getDatesWindow(dates, i, window)
        windowVals = getWindowVals(dfPlayer, datesWindow, metric)
        labelShift = np.mean(windowVals) 

    # label = perf - ewm(window)[-1]
    elif labelType == 'ewm':
        # get parameters
        window = labelParams['window']
        com = labelParams['com']

        # get most recent performance after applying ewm over window
        datesWindow = getDatesWindow(dates, i, window)
        windowVals = getWindowVals(dfPlayer, datesWindow, metric) 
        ewm_windowVals = predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values
        labelShift = ewm_windowVals[-1]

    # label = perf 
    else: 
        labelShift = 0
    return label - labelShift if train else label + labelShift

""" Construct matrix of features and vector of labels """ 
def getFeaturesLabels(infoDict, dataDict, featuresDict, labelsDict, modelDict): 
    # unpack info
    player = infoDict['player']
    metric = infoDict['metric']
    bufferWindow = infoDict['buffer']

    # unpack data
    dfLeague = dataDict['df']
    teamsDict = dataDict['teamsDict']

    # get player specific data (dataframe + dates played)
    dfPlayer = dfLeague[dfLeague.Player == player]
    dates = dfPlayer.gmDate.values 

    # initialize feature matrix and labels
    features = np.array([])
    labels = np.array([])

    # iterate through every game
    for i in range(bufferWindow, len(dates)): 
        # construct feature
        feature, n = getFeature(dates, i, dfPlayer, dfLeague, teamsDict, featuresDict, metric)

        # construct label 
        currDate = dates[i]
        label = getGamePerf(dfPlayer, currDate, metric)
        label = updateLabel(label, dates, i, dfPlayer, labelsDict, metric, train=True)

        # append to feature matrix and labels
        features = np.vstack([features, feature]) if features.size else feature
        labels = np.append(labels, label)

    # unpack model info
    rank = modelDict['rank']
    #features = hsvt(features, rank=rank)
    featuresProj = features.copy()
    featuresProj[:, :n] = hsvt(features[:, :n], rank=rank)
    return featuresProj, features, labels

""" update sla model """ 
def updateModel(updateCount, updateType, updatePeriod, sla, labels, label, 
    featuresUpdate, features, feature, featuresProj, featureProj, n, rank):
    # append to raw features and threshold when retraining model
    if updateType == 'raw': 
        # featuresUpdate = raw features
        featuresUpdate = np.vstack([featuresUpdate, feature])
        labels = np.append(labels, label)
        updateCount += 1

        if updateCount == updatePeriod: 
            # threshold raw features
            featuresUpdateProj = featuresUpdate.copy()
            featuresUpdateProj[:, :n] = hsvt(featuresUpdate[:, :n], rank=rank)

            # retrain model
            sla.fit(featuresUpdateProj, labels)

            # reinitialize
            featuresProj = featuresUpdateProj
            updateCount = 0

    # append to projected features (featuresUpdate = featuresProj)
    elif updateType == 'project':
        # featuresUpdate = projected features
        featuresUpdate = np.vstack([featuresUpdate, featureProj])
        labels = np.append(labels, label)
        updateCount += 1

        if updateCount == updatePeriod: 
            # retrain model
            sla.fit(featuresUpdate, labels)

            # reinitialize
            featuresProj = featuresUpdate 
            updateCount = 0

    # otherwise don't update
    return updateCount, featuresUpdate, featuresProj, labels

""" Train SLA Model """ 
# dataDict should be Train or TrainCV
def trainSLA(infoDict, dataDict, featuresDict, labelsDict, modelDict, slaDict):
    # construct feature matrix and label of vectors
    featuresProj, features, labels = getFeaturesLabels(infoDict, dataDict, featuresDict, labelsDict, modelDict)

    # unpack sla parameters 
    slaType = slaDict['type']
    slaParams = slaDict['params']

    # create and fit model to data
    sla = SLAForecast(slaType, slaParams)
    sla.fit(featuresProj, labels)

    # update sla dictionary
    slaDict.update({'model': sla, 'featuresProj': featuresProj, 'features': features, 'labels': labels}) 

""" Test SLA Model """ 
# dataDict should be CV or Test
def testSLA(infoDict, dataDict, featuresDict, labelsDict, modelDict, slaDict):  
    # unpack info
    player = infoDict['player']
    metric = infoDict['metric']
    bufferWindow = infoDict['buffer']

    # unpack data
    dfLeague = dataDict['df']
    teamsDict = dataDict['teamsDict']

    # unpack sla info
    sla = slaDict['model']
    featuresProj = slaDict['featuresProj']
    features = slaDict['features']
    labels = slaDict['labels']

    # unpack model info
    rank = modelDict['rank']
    project = modelDict['project']
    updateType = modelDict['updateType']
    updatePeriod = modelDict['updatePeriod']

    # get player specific data (dataframe + dates played)
    dfPlayer = dfLeague[dfLeague.Player == player]
    dates = dfPlayer.gmDate.values 

    # initialize
    preds = np.array([])
    true = np.array([])
    featuresUpdate = features.copy() if updateType == 'raw' else featuresProj.copy()
    updateCount = 0

    # iterate through every game
    for i in range(bufferWindow+1, len(dates)): 
    #for i in range(bufferWindow, len(dates)): 
        # get gameday feature (raw)
        feature, n = getFeature(dates, i, dfPlayer, dfLeague, teamsDict, featuresDict, metric)

        # project relevant features onto de-noised feature space if specified, else feature static
        featureProj = feature.copy()
        if project: 
            featureProj[:n] = projectFeatures(featuresProj[:, :n], feature[:n])

        # get gameday forecast
        if slaDict['type'] == 'lwr':
            #pred = sla.predict(feature)
            pred = sla.predict(featureProj)
        else:
            #pred = sla.predict(feature.reshape(1, feature.shape[0]))[0]
            pred = sla.predict(featureProj.reshape(1, featureProj.shape[0]))[0]
        pred = updateLabel(pred, dates, i, dfPlayer, labelsDict, metric, train=False)
        preds = np.append(preds, pred)

        # get gameday observation
        currDate = dates[i]
        label = getGamePerf(dfPlayer, currDate, metric)
        true = np.append(true, label)

        # update labels, features, model
        updateCount, featuresUpdate, featuresProj, labels = updateModel(updateCount, updateType, updatePeriod, 
            sla, labels, label, featuresUpdate, features, feature, featuresProj, featureProj, n, rank)
    return preds, true 





""" Get points per game scored by team's opposition """ 
""" def getTeamOppPerf(teamsDict, team, date, window=-1, metric='OppPTS'): 
    # get team data
    dfTeam = teamsDict[team]
    
    if window == -1:
        # get entire team's opponents' histories
        oppPerf = dfTeam.loc[dfTeam.gmDate < date, metric].mean()
    else: 
        oppPerf = dfTeam.loc[dfTeam.gmDate < date, metric].values
        oppPerf = oppPerf[-window:].mean()

        # get current date index
        idx = dfTeam.index[dfTeam.gmDate == date].values[0]
        print(idx)

        # get opponents' average performance over past 'window' games
        oppPerf = dfTeam.loc[idx-window: idx-1, metric].mean()
    return 0 if math.isnan(oppPerf) else oppPerf"""

"""def getTeamOppPerf(teamsDict, team, date, metric='OppPTS'): 
    # get team data
    dfTeam = teamsDict[team]
    
    # return team's opponents' average points per game
    oppPTS_G = dfTeam.loc[dfTeam.gmDate < date, metric].mean()
    return 0 if math.isnan(oppPTS_G) else oppPTS_G"""

""" Get points per game scored by team's opposition averaged across all teams """ 
"""def getLeagueOppPerf(teamsDict, date, metric='OppPTS'):
    league = list(teamsDict.keys())
    
    leagueOppPerf = np.array([])
    for team in league: 
        # get team's opponents' average points per game
        teamOppPerf = getTeamOppPerf(teamsDict, team, date, metric) 
        leagueOppPerf = np.append(leagueOppPerf, teamOppPerf)
    leagueOppPerf_G = leagueOppPerf.mean()
    return 0 if math.isnan(leagueOppPerf_G) else leagueOppPerf_G"""

"""
# dataDict should be CV or Test
def testSLA(infoDict, dataDict, featuresDict, labelsDict, modelDict, slaDict):  
    # unpack info
    player = infoDict['player']
    metric = infoDict['metric']
    bufferWindow = infoDict['buffer']

    # unpack data
    dfLeague = dataDict['df']
    teamsDict = dataDict['teamsDict']

    # unpack sla info
    sla = slaDict['model']
    featuresProj = slaDict['featuresProj']
    features = slaDict['features']
    labels = slaDict['labels']

    # unpack model info
    rank = modelDict['rank']
    project = modelDict['project']
    update = modelDict['update']
    updatePeriod = modelDict['updatePeriod']

    # get player specific data (dataframe + dates played)
    dfPlayer = dfLeague[dfLeague.Player == player]
    dates = dfPlayer.gmDate.values 

    # initialize
    preds = np.array([])
    true = np.array([])
    featuresUpdate = features.copy() ##if else 
    updateCount = 0

    # iterate through every game
    for i in range(bufferWindow, len(dates)): 
        # get gameday feature
        feature, n = getFeature(dates, i, dfPlayer, dfLeague, teamsDict, featuresDict, metric)

        # project relevant features onto de-noised feature space
        if project:
            featureProj = feature.copy()
            featureProj = projectFeatures(featuresProj, featureProj)
            #featureProj[:n] = projectFeatures(featuresProj[:, :n], featureProj[:n])

        # get gameday forecast
        pred = sla.predict(featureProj.reshape(1, featureProj.shape[0]))[0]
        pred = updateLabel(pred, dates, i, dfPlayer, labelsDict, metric, train=False)
        preds = np.append(preds, pred)

        # get gameday observation
        currDate = dates[i]
        label = getGamePerf(dfPlayer, currDate, metric)
        true = np.append(true, label)

        # update labels and features
        featuresUpdate = np.vstack([featuresUpdate, feature]) ## if else 
        labels = np.append(labels, label)
        updateCount += 1

        # update model
        if update and (updateCount == updatePeriod):
            # retrain model
            featuresUpdateTrain = hsvt(featuresUpdate, rank=rank)
            sla.fit(featuresUpdateTrain, labels) 

            # reinitialize so new features are projected onto appended feature space
            featuresProj = featuresUpdateTrain
            updateCount = 0

    return preds, true  """


"""def testSLA(infoDict, dataDict, featuresDict, labelsDict, modelDict, slaDict):  
    # unpack info
    player = infoDict['player']
    metric = infoDict['metric']
    bufferWindow = infoDict['buffer']

    # unpack data
    dfLeague = dataDict['df']
    teamsDict = dataDict['teamsDict']

    # unpack sla info
    sla = slaDict['model']
    featureMatrixProj = slaDict['featureMatrixProj']
    featureMatrix = slaDict['featureMatrix']
    labels = slaDict['labels']

    # unpack model info
    rank = modelDict['rank']
    project = modelDict['project']
    update = modelDict['update']
    updatePeriod = modelDict['updatePeriod']

    # get player specific data (dataframe + dates played)
    dfPlayer = dfLeague[dfLeague.Player == player]
    dates = dfPlayer.gmDate.values 

    # initialize
    preds = np.array([])
    true = np.array([])
    featureMatrixUpdate = featureMatrix.copy()
    updateCount = 0

    # iterate through every game
    for i in range(bufferWindow, len(dates)): 
        # get gameday feature
        feature, num_hsvt_features = getFeature(dates, i, dfPlayer, dfLeague, 
                                                teamsDict, featuresDict, metric)

        # project relevant features onto de-noised feature space
        if project:
            featureMatrixProj_temp = featureMatrixProj[:, :num_hsvt_features]
            featureProj = feature[:num_hsvt_features]
            feature[:num_hsvt_features] = projectFeatures(featureMatrixProj_temp, featureProj)

        # get gameday forecast
        pred = sla.predict(feature.reshape(1, feature.shape[0]))[0]
        pred = updateLabel(pred, dates, i, dfPlayer, labelsDict, metric, train=False)
        preds = np.append(preds, pred)

        # get gameday observation
        currDate = dates[i]
        label = getGamePerf(dfPlayer, currDate, metric)
        true = np.append(true, label)

        # append feature and labels 
        featureMatrixUpdate = np.vstack([featureMatrixUpdate, feature])
        labels = np.append(labels, label)
        updateCount += 1

        # update model
        if update and (updateCount == updatePeriod): 
            # retrain model 
            sla.fit(featureMatrixUpdate, labels)

            # reinitialize so new features are projected onto appended feature space
            featureMatrix = featureMatrixUpdate
            updateCount = 0

    return preds, true """

