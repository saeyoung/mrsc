from matplotlib import pyplot as plt
import numpy as np 

"""
Plot moving average
"""
def plotMA(pred, true, windowSize, player, metric): 
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
    fileName = 'plots/games/' + player + str(windowSize) + '.png'
    plt.savefig(fileName, bbox_inches='tight')
    plt.show()

def rmse(pred, true): 
    error = (pred - true) ** 2
    return np.sqrt(np.mean(error))