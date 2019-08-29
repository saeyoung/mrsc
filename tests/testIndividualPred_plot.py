#############################################################
#
# NBA Individual Player Performance Prediction
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from mrsc.src.model.SVDmodel import SVDmodel
from mrsc.src.model.Target import Target
from mrsc.src.model.Donor import Donor
from mrsc.src.synthcontrol.mRSC import mRSC

from mrsc.src.importData import *
import mrsc.src.utils as utils

def getActivePlayers(stats, year, buffer):
    # list of name of the players who were active in this and last year
    thisYear = stats[stats.Year == year].copy()
    players = list(thisYear.Player.unique())
    for i in range(1, buffer+1):
        previousYear = stats[stats.Year == (year-i)].copy()
        players = list(set(players) & set(previousYear.Player.unique()))
    return players

def topPlayers(stats, year, metric, n):
    stats = stats[stats.Year == year]
    stats = stats.groupby('Player').mean().reset_index()
    stats_sorted = stats[stats.Year == year].sort_values(metric, ascending = False).reset_index(drop=True)
    return stats_sorted[["Player","player_id"]][:n]

def getWeitghts(target, donor, metrics_list, expSetup, method = "mean"):   
    # get mat_form_method
    mat_form_method = expSetup[0] # "fixed"
    
    # get weights for metrics
    weights_list = []
    for metrics in metrics_list:
        target_data = target.concat(metrics, 2016, pred_length=1)
        num_k = len(metrics)
        total_index = int(target_data.shape[1] / num_k)
        donor_data = donor.concat(metrics, 2016, total_index, method = mat_form_method)
    
        if (method == "mean"):
            weights = []
            for i in range(num_k):
                weights.append(1/(donor_data.iloc[:,i*total_index:(i+1)*total_index].mean().mean()))
            weights_list.append(weights)
        elif (method == "var"):
            weights = []
            for i in range(num_k):
                weights.append(1/(1+np.var(donor_data.iloc[:,i*total_index:(i+1)*total_index].to_numpy().flatten())))
            weights_list.append(weights)
        else:
            raise ValueError("invalid method")
    return weights_list

# test for a multiple time series imputation and forecasting
def test():
    """
    import data
    """
    print("importing data")
    players = pd.read_csv("../data/nba-players-stats/player_data.csv")
    players = players[players.year_start >= 1980] # only choose players who started after 1980
    players["player_id"] = range(0,len(players.name)) # assign id

    stats = pd.read_csv("../data/nba-players-stats/Seasons_Stats.csv")
    stats = stats[stats.Player.isin(players.name)]

    # only after 1980
    stats = stats[stats.Year >= 1980]

    # without duplicated names --> to do: how to distinguish multiple player with the same name
    stats = removeDuplicated(players, stats)
    stats.Year = stats.Year.astype(int)
    stats.year_count = stats.year_count.astype(int)

    print("preparing data")
    # transform stats to a dictionary composed of df's for each stat
    # the stats are re-calculated to get one stat for each year
    metricsPerGameColNames = ["PTS","AST","TOV","TRB","STL","BLK"]
    metricsPerGameDict = getMetricsPerGameDict(stats, metricsPerGameColNames)

    metricsPerCentColNames = ["FG","FT","3P"]
    metricsPerCentDict = getMetricsPerCentDict(stats, metricsPerCentColNames)

    metricsWeightedColNames = ["PER"]
    metricsWeightedDict = getMetricsWeightedDict(stats, metricsWeightedColNames)

    allMetricsDict = {**metricsPerGameDict, **metricsPerCentDict, **metricsWeightedDict}
    allPivotedTableDict = getPivotedTableDict(allMetricsDict)

    # this matrix will be used to mask the table
    df_year = pd.pivot_table(stats, values="Year", index="Player", columns = "year_count")


    """
    experiment setup
    """
    # overall setup
    expSetup = ["sliding", "SVD", "all", "pinv", False]
    threshold = 0.97

    metrics1 = ["PTS_G","PER_w","AST_G","TRB_G"]
    metrics2 = ["FG%","FT%","3P%"]
    metrics3 = ["BLK_G","TOV_G","STL_G"]

    metrics_list = [metrics1, metrics2, metrics3]

    ###################################################
    # offMetrics = ["PTS_G","AST_G","TOV_G","PER_w", "FG%","FT%","3P%"]
    # defMetrics = ["TRB_G","STL_G","BLK_G"]
    # metrics_list = [offMetrics, defMetrics]

    ###################################################
    playerName = "Ryan Anderson"

    target = Target(playerName, allPivotedTableDict, df_year)
    donor = Donor(allPivotedTableDict, df_year)

    weights_list = getWeitghts(target, donor, metrics_list, expSetup, method="mean")

    mrsc = mRSC(donor, target, probObservation=1)

    fig, axs = plt.subplots(3, 5)
    player_pred = pd.DataFrame()
    player_true = pd.DataFrame()
    for i in range(len(metrics_list)):
        mrsc.fit_threshold(metrics_list[i], weights_list[i], 2016, pred_length = 1, threshold = threshold, setup = expSetup)
        pred = mrsc.predict()
        true = mrsc.getTrue()
        pred.columns = [playerName]
        true.columns = [playerName]
        player_pred = pd.concat([player_pred, pred], axis=0)
        player_true = pd.concat([player_true, true], axis=0)

        # mrsc.plot()
        for j in range(len(metrics_list[i])):
            metric = metrics_list[i][j]
            true_trajectory = target.data[metric].dropna(axis='columns').iloc[:,:mrsc.total_index]

            pred_val = np.dot(mrsc.model.donor_data.iloc[:,j*mrsc.model.total_index:((j+1)*mrsc.model.total_index)].T, mrsc.model.beta).T
            pred_trajectory = pd.DataFrame(pred_val, columns =true_trajectory.columns, index=true_trajectory.index)

            markers_on = [true_trajectory.shape[1]-1]

            axs[i, j].plot(true_trajectory.T, marker = 'o', color='red', label='true')
            axs[i, j].plot(pred_trajectory.T, marker = 'o', markevery=markers_on, color='blue', label='prediction')
            axs[i, j].set_title(playerName + ": " +metric)
            # axs[i, j].legend()

    for ax in axs.flat:
        ax.set(xlabel='years played in NBA')
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.show()

    ###################
    mask = (player_true !=0 )
    mape = np.abs(player_pred - player_true) / player_true[mask]
    print("*** MAPE ***")
    print(mape.mean(axis=1))
    print("MAPE for all: ", mape.mean().mean())

    rmse = utils.rmse_2d(player_true, player_pred)
    print()
    print("*** RMSE ***")
    print(rmse)
    print("RMSE for all: ", rmse.mean())

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    test()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
