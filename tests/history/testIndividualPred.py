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
    activePlayers = getActivePlayers(stats, 2016, 5)
    activePlayers.sort()
    # offMetrics = ["PTS_G","AST_G","TOV_G","PER_w", "FG%","FT%","3P%"]
    # defMetrics = ["TRB_G","STL_G","BLK_G"]

    metrics_to_use = ["PTS_G","AST_G","TOV_G","PER_w", "FG%","FT%","3P%","TRB_G","STL_G","BLK_G"]

    weights = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    expSetup = ["fixed", "SVD", "all", "pinv", False]

    singvals_list = [1,2,4,8,16,32]

    print("start experiment")
    for singvals in singvals_list:
        pred_all = pd.DataFrame()
        true_all = pd.DataFrame()
        for playerName in activePlayers:
            target = Target(playerName, allPivotedTableDict, df_year)
            donor = Donor(allPivotedTableDict, df_year)

            mrsc = mRSC(donor, target, probObservation=1)
            mrsc.fit(metrics_to_use, weights, 2016, pred_length = 1, singvals = singvals, setup = expSetup)
            
            pred = mrsc.predict()
            true = mrsc.getTrue()
            pred.columns = [playerName]
            true.columns = [playerName]
            
            pred_all = pd.concat([pred_all, pred], axis=1)
            true_all = pd.concat([true_all, true], axis=1)

        mask = (true_all !=0 )
        mape = np.abs(pred_all - true_all) / true_all[mask]
        print(singvals)
        print(mape.mean(axis=1))

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
