from matplotlib import pyplot as plt
import numpy as np 

""" Plot moving average """ 
def plotMA(pred, true, windowSize, player, folderName, metric='PTS_G', saveFig=False): 
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

""" Compute root mean squared error """ 
def rmse(pred, true): 
    error = (pred - true) ** 2
    return np.sqrt(np.mean(error))

""" Plot window average (fixed) """ 
def plotWindowAvg(seriesDict, trueDict, windowSize, player, folderName, metric='PTS_G', saveFig=False): 
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
        error = rmse(trueWindow, seriesWindow).round(2)
        plt.plot(seriesWindow, label='{}: {}'.format(s, error), marker='.')
    plt.ylabel(metric)
    plt.xlabel('Games')
    plt.title('{}: window = {}'.format(player, windowSize))
    plt.legend(loc='best')
    plt.show() 

def getWindowAvg(series, window): 
    seriesWindow = np.array([])
    for i in range(0, len(series)-window, window): 
        windowAvg = np.mean(series[i: i+window])
        seriesWindow = np.append(seriesWindow, windowAvg)
    return seriesWindow 

