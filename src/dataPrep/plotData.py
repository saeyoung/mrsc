from matplotlib import pyplot as plt
import numpy as np 

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
    for s, sdict in seriesDict.items(): 
        series = sdict['data']
        seriesWindow = getWindowAvg(series, windowSize)
        if errorType == 'mae':
            error = mae(trueWindow, seriesWindow).round(2)
        else: 
            error = rmse(trueWindow, seriesWindow).round(2)
        plt.plot(seriesWindow, label='{}: {}'.format(s, error), marker='.')
    plt.ylabel(metric)
    plt.xlabel('Games')
    plt.title('{}: window = {}, errType = {}'.format(player, windowSize, errorType))
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
