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

def removeDuplicated(players, stats):
    """
    players: "../data/nba-players-stats/player_data.csv"
    stats: "../data/nba-players-stats/Seasons_Stats.csv"
    """
    # players with the same name
    names = players.name.unique()
    duplicated = np.array([])

    for name in names:
        numrows = len(players[players.name == name])
        if numrows != 1:
            duplicated = np.append(duplicated, name)

    duplicated = np.sort(duplicated)

    start_year = players.copy()
    start_year = start_year.rename(columns={"name":"Player"})

    # for non-duplicated players
    stats_not_duplicated = stats[~stats.Player.isin(duplicated)]
    stats_not_duplicated = pd.merge(stats_not_duplicated, start_year, on="Player", how="left")

    # only take the values that make sense
    stats_not_duplicated = stats_not_duplicated[(stats_not_duplicated.Year >= stats_not_duplicated.year_start) & (stats_not_duplicated.Year <= stats_not_duplicated.year_end )]
    stats_not_duplicated["year_count"] = stats_not_duplicated.Year - stats_not_duplicated.year_start

    return stats_not_duplicated

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
    print("* importing data")
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

    print("* preparing data")
    # transform stats to a dictionary composed of df's for each stat
    # the stats are re-calculated to get one stat for each year
    metricsPerGameColNames = ["PTS","AST","TOV","TRB","STL","BLK","3P"]
    metricsPerGameDict = getMetricsPerGameDict(stats, metricsPerGameColNames)

    metricsPerCentColNames = ["FG","FT"]
    metricsPerCentDict = getMetricsPerCentDict(stats, metricsPerCentColNames)

    metricsWeightedColNames = ["PER"]
    metricsWeightedDict = getMetricsWeightedDict(stats, metricsWeightedColNames)

    allMetricsDict = {**metricsPerGameDict, **metricsPerCentDict, **metricsWeightedDict}
    allPivotedTableDict = getPivotedTableDict(allMetricsDict)
    allMetrics = list(allMetricsDict.keys())

    # this matrix will be used to mask the table
    df_year = pd.pivot_table(stats, values="Year", index="Player", columns = "year_count")

    """
    experiment setup
    """
    activePlayers = getActivePlayers(stats, 2016, 4)
    activePlayers.sort()
    activePlayers.remove("Kevin Garnett")
    activePlayers.remove("Kobe Bryant")

    # offMetrics = ["PTS_G","AST_G","TOV_G","3P_G","PER_w", "FG%","FT%"]
    # defMetrics = ["TRB_G","STL_G","BLK_G"]
    expSetup = ["sliding", "SVD", "all", "pinv", False]
    threshold = 0.97
    metrics_to_use = ["PTS_G","AST_G","TOV_G","3P_G","PER_w", "FG%","FT%","TRB_G","STL_G","BLK_G"]

    print("* start experiment")

    print("*******************")
    print("uniform weights")
    weights = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    pred_all = pd.DataFrame()
    true_all = pd.DataFrame()
    for playerName in activePlayers:
        target = Target(playerName, allPivotedTableDict, df_year)
        donor = Donor(allPivotedTableDict, df_year)

        mrsc = mRSC(donor, target, probObservation=1)
        mrsc.fit_threshold(metrics_to_use, weights, 2016, pred_length = 1, threshold = threshold, setup = expSetup)
      
        pred = mrsc.predict()
        true = mrsc.getTrue()
        pred.columns = [playerName]
        true.columns = [playerName]
        
        pred_all = pd.concat([pred_all, pred], axis=1)
        true_all = pd.concat([true_all, true], axis=1)
    print()
    print("*** MAPE ***")
    mask = (true_all !=0 )
    mape = np.abs(pred_all - true_all) / true_all[mask]
    print(mape.mean(axis=1))
    print("MAPE for all: ", mape.mean().mean())
    rmse = utils.rmse_2d(true_all, pred_all)
    print()
    print("*** RMSE ***")
    print(rmse)
    print("RMSE for all: ", rmse.mean())


    print("*******************")
    print("mean - standardized weights")
    metrics_list = [metrics_to_use]
    weights = getWeitghts(target, donor, metrics_list, expSetup, method="mean")[0]

    pred_all = pd.DataFrame()
    true_all = pd.DataFrame()
    for playerName in activePlayers:
        target = Target(playerName, allPivotedTableDict, df_year)
        donor = Donor(allPivotedTableDict, df_year)

        mrsc = mRSC(donor, target, probObservation=1)
        mrsc.fit_threshold(metrics_to_use, weights, 2016, pred_length = 1, threshold = threshold, setup = expSetup)
      
        pred = mrsc.predict()
        true = mrsc.getTrue()
        pred.columns = [playerName]
        true.columns = [playerName]
        
        pred_all = pd.concat([pred_all, pred], axis=1)
        true_all = pd.concat([true_all, true], axis=1)
    print()
    print("*** MAPE ***")
    mask = (true_all !=0 )
    mape = np.abs(pred_all - true_all) / true_all[mask]
    print(mape.mean(axis=1))
    print("MAPE for all: ", mape.mean().mean())
    rmse = utils.rmse_2d(true_all, pred_all)
    print()
    print("*** RMSE ***")
    print(rmse)
    print("RMSE for all: ", rmse.mean())

    print("*******************")
    print("var - standardized weights")
    metrics_list = [metrics_to_use]
    weights = getWeitghts(target, donor, metrics_list, expSetup, method="var")[0]

    pred_all = pd.DataFrame()
    true_all = pd.DataFrame()
    for playerName in activePlayers:
        target = Target(playerName, allPivotedTableDict, df_year)
        donor = Donor(allPivotedTableDict, df_year)

        mrsc = mRSC(donor, target, probObservation=1)
        mrsc.fit_threshold(metrics_to_use, weights, 2016, pred_length = 1, threshold = threshold, setup = expSetup)
      
        pred = mrsc.predict()
        true = mrsc.getTrue()
        pred.columns = [playerName]
        true.columns = [playerName]
        
        pred_all = pd.concat([pred_all, pred], axis=1)
        true_all = pd.concat([true_all, true], axis=1)
    print()
    print("*** MAPE ***")
    mask = (true_all !=0 )
    mape = np.abs(pred_all - true_all) / true_all[mask]
    print(mape.mean(axis=1))
    print("MAPE for all: ", mape.mean().mean())
    rmse = utils.rmse_2d(true_all, pred_all)
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
