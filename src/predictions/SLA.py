import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from itertools import combinations, product
import scipy.optimize

# personal libraries
from mrsc.src.predictions import predictionMethods
from mrsc.src.predictions import gamePredictions

""" Supervised Learning Forecasting """
class SLAForecast: 
    def __init__(self, method, params):
        self.method = method
        self.params = params
        self.sla = []
        
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

        # linear regression
        else: 
            self.model = linear_model.LinearRegression(fit_intercept=False)

        # fit model to data
        self.model.fit(featureMatrix, labels)
            
    def predict(self, feature):
        return self.model.predict(feature)
    
""" Extract numerical features from window """ 
def getWindowFeatures(windowVals, ewmParam=0.5):
    # compute feature values
    std_window = np.std(windowVals)
    mean_window = np.mean(windowVals)
    ewm_window = predictionMethods.applyEWMA(pd.Series(windowVals), param=ewmParam).values[-1]
    
    # create feature vector
    feature = np.array([std_window, mean_window, ewm_window])
    return feature

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

""" get game outcome (Win/Loss) """
def getGameOutcome(df, date): 
    result = df.loc[df.gmDate == date, 'teamRslt'].values[0]
    return 1 if result == 'Win' else -1


""" get team location (Home/Away) """
def getTeamLoc(df, date): 
    teamLoc = df.loc[df.gmDate == date, 'teamLoc'].values[0]
    return 1 if teamLoc == 'Home' else -1

""" get categorical features """ 
def getCategoricalFeatures(featuresDict, featureTypes): 
    categoricalFeature = np.array([])

    if 'teamLoc' in featureTypes: 
        teamDict = featuresDict['teamLoc']
        df = teamDict['df']
        currDate = teamDict['date']
        teamLoc = getTeamLoc(df, currDate)
        categoricalFeature = np.append(categoricalFeature, teamLoc)

    if 'gameOutcome' in featureTypes:
        gameOutcomeDict = featuresDict['gameOutcome']
        df = gameOutcomeDict['df']
        prevDate = gameOutcomeDict['date']
        prevGameOutcome = getGameOutcome(df, prevDate)
        categoricalFeature = np.append(categoricalFeature, prevGameOutcome)

    if 'teammates' in featureTypes:
        teammateDict = featuresDict['teammates']
        df = teammateDict['df']
        currDate = teammateDict['date']
        player = teammateDict['player']
        metric = teammateDict['metric']
        n = teammateDict['n']
        absentTeammate = checkTeammates(df, player, currDate, metric, n)
        categoricalFeature = np.append(categoricalFeature, absentTeammate)

    return categoricalFeature 

""" Construct features and labels for player """ 
def constructFeatureMatrixLabels(df, player, window, catFeatureTypes, metric='PTS_G', ewmParam=0.5, n=2): 
    labels = np.array([])
    featureMatrix = np.array([])

    # get player specific data
    dfPlayer = df[df.Player == player]
    dates = dfPlayer.gmDate.values 
        
    for i in range(window, len(dates)): 
        # get current date and value (use as label)
        currDate = dates[i]
        currVal = dfPlayer.loc[dfPlayer.gmDate == currDate, metric].values[0]

        # get numerical features
        datesWindow = dates[i-window: i]
        windowVals = dfPlayer.loc[dfPlayer.gmDate.isin(datesWindow), metric].values
        numFeatures = getWindowFeatures(windowVals, ewmParam)

        # get categorical feature
        prevDate = dates[i-1]
        catFeaturesDict = {'teamLoc': {'df': dfPlayer, 'date': currDate}, 
                'gameOutcome': {'df': dfPlayer, 'date': prevDate},
                'teammates': {'df': df, 'player': player, 'date': currDate, 'metric': metric, 'n': n}}
        catFeatures = getCategoricalFeatures(catFeaturesDict, catFeatureTypes)

        # concatenate features
        feature = np.append(numFeatures, catFeatures)

        # append to feature matrix and label vector
        featureMatrix = np.vstack([featureMatrix, feature]) if featureMatrix.size else feature
        labels = np.append(labels, currVal)
    return featureMatrix, labels

