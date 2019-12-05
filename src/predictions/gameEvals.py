from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from math import log, e
from scipy.stats import entropy

# dennis libraries
from mrsc.src.predictions import predictionMethods, SLA

""" convert performance sequence into binary string """
def getBinaryDelta(x, x_ref, deltaType='raw'):
	if deltaType == 'mean':
		deltas = np.array([1 if x[i] >= np.mean(x_ref[:i]) else 0 for i in range(1, len(x))])
	else:
		deltas = np.array([1 if x[i] >= x[i-1] else 0 for i in range(1, len(x))])
	return deltas

""" evaluate performance """
def binary_error(true, preds): 
	return 1 - accuracy_score(true, preds)

""" compute information entropy """ 
def computeEntropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def plotGame(seriesDict, trueDict, player, alpha=1, errorType='rmse', deltaType='raw', 
	folderName='', metric='PTS_G', saveFig=False): 
    # unpack 
    true = trueDict['data']
    slaType = seriesDict['type']
    series = seriesDict['data']

    # upper and lower bounds
    upper = np.mean(true) + alpha * np.std(true)
    lower = np.mean(true) - alpha * np.std(true)

    # compute errors restricted to games within bounds
    true_alpha, series_alpha = get_alpha_scores(true, series, alpha)

    # compute annual errors
    error_mean = np.abs(np.mean(true) - np.mean(series))
    error_mean = error_mean.round(1)

    # compute real-valued errors (restricted to games within bounds)
    if errorType == 'mae':
        error = mae(true_alpha, series_alpha)
    elif errorType == 'r2':
        error = r2_score(true_alpha, series_alpha)
    else: 
        error = rmse(true_alpha, series_alpha)
    error = error.round(1)

    # compute binary error
    true_binary = getBinaryDelta(true, true, deltaType)
    series_binary = getBinaryDelta(series, true, deltaType)
    b_error = binary_error(true_binary, series_binary).round(2)

    # compute correlation
    #corr = pd.Series(series_alpha).corr(pd.Series(true_alpha), method='spearman')
    #corr = corr.round(2)

    # plot true means    
    plt.figure()
    true_mean = np.mean(true)
    series_mean = np.mean(series)
    plt.axhline(true_mean, label='true mean', color='lightskyblue', linewidth=1.2)
    plt.axhline(series_mean, label='{} mean: err={}'.format(slaType, error_mean), color='sandybrown', linewidth=1.2)
                
    # plot bounds
    plt.axhline(upper, color='lightgrey', linestyle='--', linewidth=1.0)
    plt.axhline(lower, color='lightgrey', linestyle='--', linewidth=1.0)
    
    # plot game by game predictions
    plt.plot(true, label='true', marker='.', color='steelblue')
    plt.plot(series, label='{}: {}={}, 0/1={}'.format(slaType, errorType, error, b_error), marker='.', color='darkorange')
    #plt.plot(series, label='{}: {}={}, 0/1={}, corr={}'.format(slaType, errorType, error, b_error, corr), marker='.', color='darkorange')
                
    plt.ylabel(metric)
    plt.xlabel('Games')
    plt.title('{}'.format(player))
    plt.legend(loc='best', prop={'size': 8})
    plt.show() 

""" Plot moving average """ 
def plotMA(pred, true, windowSize, player, folderName='', metric='PTS_G', saveFig=False): 
    filterMA = np.ones((windowSize,)) / windowSize
    trueMA = np.convolve(true, filterMA, mode='valid')
    predMA = np.convolve(pred, filterMA, mode='valid')
    errorMA = rmse(predMA, trueMA)

    plt.figure()
    plt.plot(trueMA, label='true')
    plt.plot(predMA, label='pred')
    plt.ylabel(metric)
    plt.xlabel('Games')
    plt.title('{}: MA = {}, error = {}'.format(player, windowSize, errorMA.round(2)))
    plt.legend(loc='best')

    # save figure
    if saveFig: 
        fileName = 'plots/games/' + folderName + player + str(windowSize) + '.png'
        plt.savefig(fileName, bbox_inches='tight')
    plt.show()

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

def get_alpha_scores(true, series, alpha=2):
    # get annual information to compute lower and upper bounds
    true_mean = np.mean(true)
    true_std = np.std(true)
    upper = true_mean + alpha * true_std
    lower = true_mean - alpha * true_std

    # get indices where games lie within [lower, upper]
    idx = np.where(np.logical_and(true >= lower, true <= upper))

    # restrict attention to games within [lower, upper]
    true_alpha = true[idx]
    series_alpha = series[idx]

    return true_alpha, series_alpha

""" return average error over window """ 
def error(pred, true, errorType='rmse'):
    if errorType == 'mae':
        error = mae(true, pred)
    elif errorType == 'r2': 
        error = r2_score(true, pred)
    else:
        error = rmse(true, pred)
    return error

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

""" Plot window average (fixed) """ 
def plotWindowAvg(seriesDict, trueDict, windowSize, player, errorType='rmse', folderName='', metric='PTS_G', saveFig=False): 
    # unpack true data
    true = trueDict['data']

    # compute window
    trueWindow = getWindowAvg(true, windowSize)

    # plot true
    plt.figure()
    plt.plot(trueWindow, label='true', marker='.')

    # plot series
    slaType = seriesDict['type']
    series = seriesDict['data']
    seriesWindow = getWindowAvg(series, windowSize)
    if errorType == 'mae':
        error = mae(trueWindow, seriesWindow).round(2)
    elif errorType == 'r2': 
        error = r2_score(trueWindow, seriesWindow).round(2)
    else: 
        error = rmse(trueWindow, seriesWindow).round(2)
    corr = pd.Series(seriesWindow).corr(pd.Series(trueWindow)).round(2)
    plt.plot(seriesWindow, label='{}'.format(slaType), marker='.')
    plt.ylabel(metric)
    plt.xlabel('Games')
    plt.title('{}: window={}, {}={}, corr={}'.format(player, windowSize, errorType, error, corr))
    plt.legend(loc='best')
    plt.show() 


def getWindowAvg(series, window): 
    seriesWindow = np.array([])
    for i in range(0, len(series)-window, window): 
        windowAvg = np.mean(series[i: i+window])
        seriesWindow = np.append(seriesWindow, windowAvg)
    return seriesWindow 

""" Plot window average (fixed) """ 
def plotAnnual(seriesDict, trueDict, player, bufferWindow=1, folderName='', metric='PTS', saveFig=False): 
    # unpack true data
    true = trueDict['data']

    # plot true
    plt.figure()
    plt.plot(true, label='true', marker='o', markevery=[len(true)-1])

    # plot series
    for s, sdict in seriesDict.items(): 
        series = sdict['data']
        error = mae(true[-1], series[-1]).round(2)
        plt.plot(series, label='{}: {}'.format(s, error), marker='o', markevery=[len(true)-1])
    plt.xticks(range(len(true)), range(bufferWindow, len(true)+bufferWindow))
    plt.ylabel(metric)
    plt.xlabel('Years')
    plt.title('{}'.format(player))
    plt.legend(loc='best')
    plt.show() 


def stats(player, dfLeague, teamsDict, window, oppWindow, com): 
    dfPlayer = dfLeague[dfLeague.Player == player]
    dates = dfPlayer.gmDate.values
    currPerfs = np.array([])
    deltaPerfs = np.array([])
    prevPerfs = np.array([])
    prev_mean = np.array([])
    prev_prev = np.array([])
    prevGmResults = np.array([])
    stds = np.array([])
    teamLocs = np.array([])
    y_list = np.array([])
    y_mean_list = np.array([])
    mean_list = np.array([])
    opps = np.array([])
    x_means = np.array([])

    for i in range(2, len(dates)): 
        currDate = dates[i]
        prevDate = dates[i-1]
        prevprevDate = dates[i-2]
        datesWindow = getDatesWindow(dates, i, window)

        # get perfomance values
        currPerf = getGamePerf(dfPlayer, currDate)
        prevPerf = getGamePerf(dfPlayer, prevDate)
        
        
        deltaPerf = currPerf - prevPerf
        #windowVals = getWindowVals(dfPlayer, datesWindow)
        #deltaPerf = currPerf - predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values[-1]
        
        meanPerf = np.mean(getSeasonPerf(dfPlayer, prevprevDate))
        prevprevPerf = getGamePerf(dfPlayer, prevprevDate)

        # game outcome
        gmResults = getGameOutcomes(dfPlayer, datesWindow)
        gmResults = predictionMethods.applyEWMA(pd.Series(gmResults), param=com).values
        prevGmResult = gmResults[-1]

        # std
        windowVals = getWindowVals(dfPlayer, datesWindow)
        std = np.std(windowVals)
        
        # mean
        x_mean = np.mean(windowVals) if windowVals.size else 0

        # team location
        teamLoc = getTeamLoc(dfPlayer, currDate)

        # teammate effect
        team = getTeam(dfPlayer, currDate)
        teammates = getTeammates(dfLeague, player, team, currDate)
        y, y_mean, mean = teammateEffect(dfLeague, teammates, currDate, window, 1, com)
        
        # opponent effect (position invariant)
        """position = dfPlayer.loc[dfPlayer.gmDate == currDate, 'playPos'].values[0]
        oppTeam = getOppTeam(dfPlayer, currDate)
        oppTeamOppPerf = getTeamOppPosPerf(teamsDict, oppTeam, currDate, position)
        leagueOppPerf = getLeagueOppPosPerf(teamsDict, currDate, position, oppWindow)
        opp = oppTeamOppPerf / leagueOppPerf if leagueOppPerf else 1"""

        # append 
        x_means = np.append(x_means, x_mean)
        currPerfs = np.append(currPerfs, currPerf)
        prevPerfs = np.append(prevPerfs, prevPerf)
        deltaPerfs = np.append(deltaPerfs, deltaPerf)
        prev_mean = np.append(prev_mean, prevPerf - meanPerf)
        prev_prev = np.append(prev_prev, prevPerf - prevprevPerf)
        prevGmResults = np.append(prevGmResults, prevGmResult)
        stds = np.append(stds, std)
        teamLocs = np.append(teamLocs, teamLoc)
        y_list = np.append(y_list, y)
        y_mean_list = np.append(y_mean_list, y_mean)
        mean_list = np.append(mean_list, mean)

    df = pd.DataFrame({'gmDate': dates[2:], 'PTS': currPerfs, 'Delta': deltaPerfs, 
                       'mean(X(t-1))': x_means, 'Prev': prevPerfs, 'std': stds,
                       'Prev-Mean': prev_mean, 'Prev-Prev': prev_prev, 'PrevGmRslt': prevGmResults,
                       'teamLoc': teamLocs, 'Y(t-1)': y_list, 'Y(t-1)-mean(Y(t-2))': y_mean_list,
                       'mean(Y(t-1))': mean_list})
    return df

def teammateEffect(df, teammates, date, window, n=2, com=0.3, metric='PTS_G'):
    y_list = np.array([])
    y_mean_list = np.array([])
    mean_list = np.array([])
    
    for player in teammates: 
        # get teammate info
        dfPlayer = df[df.Player == player]
        
        # get valid dates
        dates = dfPlayer.loc[dfPlayer.gmDate < date, 'gmDate'].values
        
        # Y(t-1)
        #datesWindow = dates[-window:]
        #windowVals = getWindowVals(dfPlayer, datesWindow)
        #ewm_windowVals = predictionMethods.applyEWMA(pd.Series(windowVals), param=com).values
        #y = ewm_windowVals[-1] if ewm_windowVals.size else 0 
        y = getGamePerf(dfPlayer, dates[-1]) if dates.size else 0
        
        # mean(t-1)
        datesWindow = dates[-window:]
        windowVals = getWindowVals(dfPlayer, datesWindow)
        mean = np.mean(windowVals)
        
        # Y(t-1) - mean(t-2)
        prev_mean = np.mean(windowVals[:-1])
        y_mean = y - prev_mean
        
        # append to performance list
        y_list = np.append(y_list, y)
        y_mean_list = np.append(y_mean_list, y_mean)
        mean_list = np.append(mean_list, mean)
        
    # get index of best performing player
    idx = np.nanargmax(mean_list)
        
    return y_list[idx], y_mean_list[idx], mean_list[idx]
        
    #return heapq.nlargest(n, y_list), heapq.nlargest(n, y_mean_list), heapq.nlargest(n, mean_list)



"""players = ['James Harden', 'Russell Westbrook', 'Kevin Durant', 'LeBron James', 
          'Stephen Curry', 'Al Horford', 'Chris Paul', 'Jimmy Butler', 'Anthony Davis']
dfLeague = dfTest
teamsDict = teamsPosTestDict
window = 3
oppWindow = 10
com = 0.2

for player in players: 
    df = stats(player, dfLeague, teamsDict, window, oppWindow, com)
    print("{}".format(player))
    print("*** X(t) ***")
    print("X(t) & Prev: {}".format(df['PTS'].corr(df['Prev']).round(2)))
    print("X(t) & Mean: {}".format(df['PTS'].corr(df['mean(X(t-1))']).round(2)))
    print("X(t) & (Prev-Mean): {}".format(df['PTS'].corr(df['Prev-Mean']).round(2)))
    print("X(t) & (Prev-Prev): {}".format(df['PTS'].corr(df['Prev-Prev']).round(2)))
    print("X(t) & PrevGmRslt: {}".format(df['PTS'].corr(df['PrevGmRslt']).round(2)))
    print("X(t) & Std: {}".format(df['PTS'].corr(df['std']).round(2)))
    print("X(t) & TeamLoc: {}".format(df['PTS'].corr(df['teamLoc']).round(2)))
    print("X(t) & Y(t-1): {}".format(df['PTS'].corr(df['Y(t-1)']).round(2)))
    print("X(t) & Y(t-1)-mean(Y(t-2)): {}".format(df['PTS'].corr(df['Y(t-1)-mean(Y(t-2))']).round(2)))
    print("X(t) & mean(Y(t-1)): {}".format(df['PTS'].corr(df['mean(Y(t-1))']).round(2)))
    print()"""