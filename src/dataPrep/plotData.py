from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score

# dennis libraries
from mrsc.src.predictions import gamePredictions

def plotGame(seriesDict, trueDict, player, alpha=1, errorType='rmse', folderName='', metric='PTS_G', saveFig=False): 
    # unpack 
    true = trueDict['data']
    slaType = seriesDict['type']
    series = seriesDict['data']

    # upper and lower bounds
    upper = np.mean(true) + alpha * np.std(true)
    lower = np.mean(true) - alpha * np.std(true)

    # compute errors restricted to games within bounds
    true_alpha, series_alpha = gamePredictions.get_alpha_scores(true, series, alpha)

    # compute annual errors
    error_mean = np.abs(np.mean(true_alpha) - np.mean(series_alpha))
    error_mean = error_mean.round(1)

    # compute game errors
    if errorType == 'mae':
        error = mae(true_alpha, series_alpha)
    elif errorType == 'r2':
        error = r2_score(true_alpha, series_alpha)
    else: 
        error = rmse(true_alpha, series_alpha)
    error = error.round(1)

    # compute correlation
    corr = pd.Series(series_alpha).corr(pd.Series(true_alpha))
    corr = corr.round(2)

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
    plt.plot(series, label='{}: {}={}, corr={}'.format(slaType, errorType, error, corr), marker='.', color='darkorange')
                
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

