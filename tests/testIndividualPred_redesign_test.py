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

def prepareData(stats):
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
    return allPivotedTableDict, allMetrics

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

# def getMetrics(target, donor, pred_year, allMetrics, threshold, expSetup, boundary = "threshold"):
#     target_data = target.concat(allMetrics)

#     num_k = len(allMetrics)
#     total_index = int(target_data.shape[1] / num_k)
#     mat_form_method = expSetup[0]
    
#     df_result = pd.DataFrame(0, columns = allMetrics, index = allMetrics)
#     for metric_of_interest in allMetrics:
#         df = donor.concat([metric_of_interest], 2016, total_index, method = mat_form_method)
#         apprx_rank = utils.approximate_rank(df, t = threshold)
#         energy_captured = utils.svdAanlysis(df, title=metric_of_interest, k=apprx_rank, verbose = False)[apprx_rank-1]

#         if (boundary == "threshold"):
#             b = threshold
#         elif (boundary == "energy"):
#             b = energy_captured
#         else:
#             raise Exception("wrong parameter")

#         metrics = [metric_of_interest]
#         candidates = copy.deepcopy(allMetrics)
#         candidates.remove(metric_of_interest)

#         while True:
#             energy_diff_df = pd.DataFrame()
#             for metric in candidates:
#                 comb = metrics+[metric]
#                 df = donor.concat(comb, 2016, total_index, method = mat_form_method)
#                 energy_at_apprx_rank = utils.svdAanlysis(df, k=apprx_rank, verbose=False)[-1]
                
#                 if(energy_at_apprx_rank > b):
#                     energy_diff = np.abs(energy_at_apprx_rank - energy_captured)
#                     energy_diff_df = pd.concat([energy_diff_df, pd.DataFrame([[energy_diff]], index=[metric])], axis=0)
#         #             print(energy_diff)
#             if (energy_diff_df.shape[0] == 0):
#                 break
#             new_metric = energy_diff_df.sort_values(0).index[0]
#             metrics = metrics + [new_metric]
#             candidates.remove(new_metric)
#         df_result.loc[metric_of_interest,metrics] = 1
        
#     metrics_list =[]
#     for i in range(num_k):
#         a = ((df_result.iloc[i,:]==1) & (df_result.iloc[:,i]==1))
#         metrics_list.append(a.index[a].values.tolist())

#     return metrics_list

# test for a multiple time series imputation and forecasting
def test():
    """
    import data
    """
    pred_year = 2015 # the year that we are living in
    pred_interval = 1 # we are making predictions for pred_year+1 and +2

    print("*** importing data ***")
    players = pd.read_csv("../data/nba-players-stats/player_data.csv")
    players = players[players.year_start >= 1980] # only choose players who started after 1980
    # players["player_id"] = range(0,len(players.name)) # assign id

    stats = pd.read_csv("../data/nba-players-stats/Seasons_Stats.csv")
    stats = stats[stats.Player.isin(players.name)]

    # only after 1980
    stats = stats[stats.Year >= 1980]

    # without duplicated names --> to do: how to distinguish multiple player with the same name
    stats = removeDuplicated(players, stats)
    stats.Year = stats.Year.astype(int)
    stats.year_count = stats.year_count.astype(int)

    print("*** preparing data ***")

    ########### Donor ##########
    # filter stats by the year
    stats_donor = stats[stats.Year <= pred_year]
    allPivotedTableDict, allMetrics = prepareData(stats_donor)
    donor = Donor(allPivotedTableDict)

    ########### Target ##########
    # filter stats by the year
    stats_target = stats[stats.Year <= pred_year+pred_interval]
    allPivotedTableDict, allMetrics = prepareData(stats_target)
    
    # just to debug
    df_year = pd.pivot_table(stats, values="Year", index="Player", columns = "year_count")

    """
    experiment setup
    """
    # overall setup
    donorSetup= [None,"fixed", True]
    # weighting = donorSetup[0] # None / "normalize"
    # mat_form_method = donorSetup[1] # "fixed"
    # skipNan = donorSetup[2] # (Boolean)
    denoiseSetup = ["SVD", "all"]
    # denoise_method = denoiseSetup[0] # "SVD"
    # denoise_mat_method = denoiseSetup[1] # "all"
    regression_method = "pinv"

    threshold = 0.97
    verbose = False

    ###################################################
    offMetrics = ["PTS_G","AST_G","TOV_G","PER_w", "FG%","FT%","3P_G"]
    defMetrics = ["TRB_G","STL_G","BLK_G"]
    # metrics_list = [offMetrics, defMetrics]

    metrics_list = [allMetrics]

    ###################################################

    ##############################################################
    # test 1
    ##############################################################
    playerNames = getActivePlayers(stats, pred_year, buffer=4)
    playerNames.remove("Kevin Garnett")
    playerNames.remove("Kobe Bryant")
    # playerNames.remove("Jason Kidd")

    all_pred = pd.DataFrame()
    all_true = pd.DataFrame()
    for playerName in playerNames:
        # print(playerName)
        # print("*** year - year_count matching for this player")
        # a = df_year[df_year.index == playerName]
        # print(a.dropna(axis=1))

        target = Target(playerName, allPivotedTableDict)
        # print("*** target - total index: ", target.total_index)
        # print(target.concat(metrics_list[1]))

        mrsc = mRSC(donor, target, pred_interval, probObservation=1)
        
        player_pred = pd.DataFrame()
        player_true = pd.DataFrame()
        for i in range(len(metrics_list)):
            mrsc.fit_threshold(metrics_list[i], pred_interval, threshold, donorSetup, denoiseSetup,regression_method, verbose)
            pred = mrsc.predict()
            true = mrsc.getTrue()
            pred.columns = [playerName]
            true.columns = [playerName]
            player_pred = pd.concat([player_pred, pred], axis=0)
            player_true = pd.concat([player_true, true], axis=0)
        all_pred = pd.concat([all_pred, player_pred], axis=1)
        all_true = pd.concat([all_true, player_true], axis=1)

    ###################
    print(all_pred)
    print(all_pred.shape)
    mask = (all_true !=0 )
    mape = np.abs(all_pred - all_true) / all_true[mask]
    print("*** MAPE ***")
    print(mape.mean(axis=1))
    print("MAPE for all: ", mape.mean().mean())

    rmse = utils.rmse_2d(all_true, all_pred)
    print()
    print("*** RMSE ***")
    print(rmse)
    print("RMSE for all: ", rmse.mean())
    ##############################################################


    # ##############################################################
    # # test 2
    # ##############################################################
    # playerName = "LeBron James"
    # target = Target(playerName, allPivotedTableDict)
    # mrsc = mRSC(donor, target, pred_interval, probObservation=1)
    # mrsc._assignData(metrics_list[0], pred_interval, weighting="normalize", mat_form_method = "fixed", skipNan = True)
    # # print(mrsc.metrics)
    # # print(mrsc.num_k)
    # # print(mrsc.target_data.shape)
    # # print(mrsc.target_data)
    # # print(mrsc.donor_data.shape)
    # # print(mrsc.donor_data)

    # print("*** year - year_count matching for this player")
    # a = df_year[df_year.index == playerName]
    # print(a.dropna(axis=1))
    # print("*** target - total index: ", target.total_index)

    # mrsc.fit_threshold(allMetrics, pred_interval, threshold, donorSetup, denoiseSetup,regression_method, verbose)
    # pred = mrsc.predict()
    # true = mrsc.getTrue()
    # print("CHECK")
    # print(mrsc.target_data)
    # print()
    # print("PRED")
    # print(pred)
    # print()
    # print("TRUE")
    # print(true)


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
