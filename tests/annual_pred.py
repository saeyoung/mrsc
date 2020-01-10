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
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import copy
import pickle

from mrsc.src.model.SVDmodel import SVDmodel
from mrsc.src.model.Target import Target
from mrsc.src.model.Donor import Donor
from mrsc.src.synthcontrol.mRSC import mRSC
from mrsc.src.importData import *
import mrsc.src.utils as utils

from statsmodels.tsa.arima_model import ARMA

def prepareData(stats):
    # transform stats to a dictionary composed of df's for each stat
    # the stats are re-calculated to get one stat for each year
    metricsPerGameColNames = ["PTS","AST","TOV","TRB","STL","BLK","3P","MP"]
    metricsPerGameDict = getMetricsPerGameDict(stats, metricsPerGameColNames)

    metricsPerCentColNames = ["FG","FT"]
    metricsPerCentDict = getMetricsPerCentDict(stats, metricsPerCentColNames)

    metricsWeightedColNames = ["PER"]
    metricsWeightedDict = getMetricsWeightedDict(stats, metricsWeightedColNames)

    allMetricsDict = {**metricsPerGameDict, **metricsPerCentDict, **metricsWeightedDict}
    allPivotedTableDict = getPivotedTableDict(allMetricsDict)
    allMetrics = list(allMetricsDict.keys())
    return allPivotedTableDict, allMetrics

def getTopPlayers(stats, year, metric, n):
    stats = stats[stats.Year == year]
    stats = stats.groupby('Player').mean().reset_index()
    stats_sorted = stats[stats.Year == year].sort_values(metric, ascending = False).reset_index(drop=True)
    return stats_sorted[["Player"]][:n]

# bug in playerName
def getBenchmark(target, metrics_to_use, pred_interval):    
    target_data, nanIndex = target.concat(metrics_to_use)
    num_k = len(metrics_to_use)
    interv_index = int(target_data.shape[1]/num_k - pred_interval)
    total_index = int(interv_index + 1)
    
    # true
    true = utils.get_postint_data(target_data, interv_index, total_index, num_k).T
    true.index = metrics_to_use
    
    # predictions
    history = utils.get_preint_data(target_data, interv_index, total_index, num_k)
    pred = []
    for i in range(num_k):
        pred.append(history.iloc[:,i*interv_index:(i+1)*interv_index].mean(axis=1).to_list())

    #pred = pd.DataFrame(pred, index=metrics_to_use, columns = [playerName]) #bug
    pred = pd.DataFrame(pred, index=metrics_to_use, columns = [target.key])
    return true, pred

def getR2(true, pred, bench):
    ss_res = pd.DataFrame((true.values - pred.values)**2, index=true.index).sum(axis=1)
    ss_tot = pd.DataFrame((true.values - bench.values)**2, index=true.index).sum(axis=1)
    return (1-ss_res/ss_tot).to_frame(name = pred.columns.values[0])

def edit_year_count(df):
    for player in df.Player.unique():
        df.loc[df.Player == player, 'year_count'] = np.arange(len(df[df.Player == player]))
    return df

def fix_duplicates(stats, duplicate_names):
    for player in duplicate_names:
        y_true = stats[stats.Player==player]['Year'].values
        y_old = np.arange(np.min(y_true), np.min(y_true) + len(y_true))
        idx = np.nonzero(y_true - y_old)[0]
        years = y_true[idx]
        stats.loc[(stats['Player'] == player) & (stats['Year'].isin(years)), 'Player'] = player + ' Jr'
    return stats

""" 
retrive players who played in 'pred_year' who satisfy the following:
1. played at least 'min_games' in 'pred_year' 
2. played at least 'buffer' seasons, where each admissible season has > 'min_games'
"""
def get_active_players(stats_target, pred_year, buffer, min_games):
    df = stats_target.copy()    
    
    # minimum number of years a player must have played in NBA
    year_min = pred_year - buffer

    # only consider players that played at least 'min_games' in 'pred_year' 
    active = df.loc[(df.Year == pred_year) & (df.G >= min_games), 'Player'].tolist()

    # consider remaining players who played >= 'buffer' seasons (prior to 'pred_year') with > 'min_games'
    df = df.loc[df.Player.isin(active)]
    counts = df['Player'].value_counts()
    df = df[df['Player'].isin(counts[counts > buffer].index)].sort_values(by=['Player'])
    active = df['Player'].unique()
    
    return active.tolist()


def get_active_players_old(stats_target, pred_year, buffer, min_games):
    df = stats_target.copy()    
    
    # minimum number of years a player must have played in NBA
    year_min = pred_year - buffer

    # only consider players that played at least 'min_games' in 'pred_year' 
    active = df.loc[(df.Year == pred_year) & (df.G >= min_games), 'Player'].tolist()

    # consider remaining players who played >= 'buffer' seasons (prior to 'pred_year') with > 'min_games'
    active_final = []
    df = df[df.Year < pred_year]
    
    for player in active: 
        if len(df[df.Player==player]) >= buffer:
            active_final.append(player)
    
    return active_final

def annual_predictions(playerNames, allPivotedTableDict, donor, pred_interval, metrics, pred_metrics,
                      threshold, donorSetup, denoiseSetup, regression_method, verbose, dir_name, top_players):
    all_pred = pd.DataFrame()
    all_true = pd.DataFrame()
    all_bench = pd.DataFrame()
    all_R2 = pd.DataFrame()
    
    for playerName in playerNames:            
        target = Target(playerName, allPivotedTableDict)
        mrsc = mRSC(donor, target, pred_interval, probObservation=1)        
        player_pred = pd.DataFrame()
        player_true = pd.DataFrame()
        
        # benchmark
        true, benchmark = getBenchmark(target, pred_metrics, pred_interval)
        
        for metric in metrics:
            mrsc.fit_threshold(metric, threshold, donorSetup, denoiseSetup,regression_method, verbose)
            pred = mrsc.predict()
            pred = pred[pred.index.isin(pred_metrics)]
            true = mrsc.getTrue()
            
#             # ARMA
#             data = mrsc.target_data.T.ewm(com=0.5).mean().T.values.flatten()
#             data = data[:-1]
#             ewm = data[-1]
#     #         if (np.sum(data != 0)==0):
#     #             pred_arima = 0
#     #         else:
#     #             model = ARMA(data, order=(1, 1))
#     #             model_fit = model.fit(disp=False)
#     #             pred_arma = model_fit.predict(len(data), len(data))

#             pred = 0.5*pred + 0.5*ewm

            pred.columns = [playerName+" "+ str(a) for a in range(pred_interval)]
            true.columns = [playerName+" "+ str(a) for a in range(pred_interval)]
            player_pred = pd.concat([player_pred, pred], axis=0)
            player_true = pd.concat([player_true, true], axis=0)

        all_pred = pd.concat([all_pred, player_pred], axis=1)
        all_true = pd.concat([all_true, player_true], axis=1)
        all_bench = pd.concat([all_bench, benchmark], axis=1)

        R2 = getR2(player_true, player_pred, benchmark)
        all_R2 = pd.concat([all_R2, R2], axis=1)

    ###################
    print("Number of metrics: {}".format(all_pred.shape[0]))
    print("Number of players: {}".format(all_pred.shape[1]))
    print()
    mask = (all_true !=0 )
    mape = np.abs(all_pred[mask] - all_true[mask]) / all_true[mask]
    print("*** MAPE ***")
    print(mape.mean(axis=1))
    print("MAPE for all: ", mape.mean().mean())
    
    rmse = utils.rmse_2d(all_true, all_pred)
    print()
    print("*** RMSE ***")
    print(rmse)
    print("RMSE for all: ", rmse.mean())

    print()
    print("*** R2 ***")
    print(all_R2.mean(axis=1))
    print("R2 for all: ", all_R2.mean(axis=1).mean(axis=0))

    edited_R2 = copy.deepcopy(all_R2)
    edited_R2[edited_R2 <0] = 0
    print()
    print("*** edited R2 ***")
    print(edited_R2.mean(axis=1))
    print("R2 for all: ", edited_R2.mean().mean())
        
    return all_pred, all_true, all_R2, all_bench
    ##############################################################

# test for a multiple time series imputation and forecasting
def test():
    """
    import data
    """

    """ USER PARAMETERS """
    starting_year = 1970
    #min_games_donor = 40
    #min_games_target = 40
    #min_games = np.min((min_games_donor, min_games_target))
    min_games= 30
    train_year = 2015
    pred_interval = 1
    pred_year = train_year + pred_interval
    buffer = 4
    num_top_players = 300

    parameters = {'starting_year': starting_year,
                 #'min_games_donor': min_games_donor,
                 #'min_games_target': min_games_target,
                 'min_games': min_games,
                 'pred_year': pred_year,
                 'pred_interval': pred_interval,
                 'min_num_years_played': buffer}

    """ players dataframe """
    players = pd.read_csv("../data/nba-players-stats/player_data.csv")

    # sort players by (name, year_start)
    players = players.sort_values(by=['name', 'year_start'])

    # filter players by years considered
    players = players[players.year_start >= starting_year] 

    """ stats dataframe """
    stats = pd.read_csv("../data/nba-players-stats/Seasons_Stats.csv")

    # fix the name* issue
    stats = stats.replace('\*','',regex=True)

    # sort players by (name, year)
    stats = stats.sort_values(by=['Player', 'Year'])

    # remove multiple rows for the same [Year, Player]
    totals = stats[stats.Tm == "TOT"]
    duplicates_removed = stats.drop_duplicates(subset=["Year", "Player"], keep=False)
    stats = pd.concat([duplicates_removed, totals], axis=0).sort_values("Unnamed: 0")

    # filter players by years considered
    stats = stats[stats.Year >= starting_year]

    """ players + stats dataframes """
    valid_players = list(set(stats.Player) & set(players.name))
    stats = stats[stats['Player'].isin(valid_players)]
    players = players[players['name'].isin(valid_players)]

    # correct names in "players" dataframe
    duplicate_names = []
    for name in players.name:
        numrows = len(players[players['name']==name])
        if numrows == 2:
            duplicate_names.append(name)
            i = 0
            for birth_date in players.loc[players['name']==name, 'birth_date']:
                if i == 1:
                    players.loc[(players['name']==name) & (players['birth_date']==birth_date) , 'name'] = name + ' Jr'
                i += 1
        elif numrows == 3:
            players = players[players.name != name]
             
    # correct names in "stats" dataframe
    stats = fix_duplicates(stats, duplicate_names)

    # merge
    players = players.rename(columns={"name": "Player"})
    stats = pd.merge(stats, players, on='Player', how='left')
    # sanity check 
    stats = stats[(stats.Year >= stats.year_start) & (stats.Year <= stats.year_end)]

    stats.Year = stats.Year.astype(int)
    stats.year_start = stats.year_start.astype(int)
    stats['year_count'] = stats.Year - stats.year_start

    """ donor setup """
    # only consider years prior to 'pred_year'
    stats_donor = stats[stats.Year < pred_year]
    stats_donor = stats_donor.sort_values(by=['Player', 'Year'])

    # only consider years in which at least "min_games" number of games were played
    stats_donor = stats_donor[stats_donor.G >= min_games]

    # edit 'year_count'
    stats_donor = edit_year_count(stats_donor)

    # create donor object
    allPivotedTableDict_d, allMetrics = prepareData(stats_donor)
    donor = Donor(allPivotedTableDict_d)
    """ target setup """
    # consider all years up to (and including) 'pred_year'
    stats_target = stats[stats.Year <= pred_year]

    # exclude years prior to 'pred_year' in which < 'min_games' were played
    idx = stats_target.loc[(stats_target.G < min_games) & (stats_target.Year < pred_year)].index
    stats_target.drop(idx, inplace=True)

    # edit 'year_count'
    stats_target = edit_year_count(stats_target)

    # create target dictionary of values
    allPivotedTableDict, allMetrics = prepareData(stats_target)

    # get target player names
    targetNames = get_active_players(stats_target, pred_year, buffer, min_games) 
    targetNames.sort()

    if 'Kevin Garnett' in targetNames: 
        targetNames.remove("Kevin Garnett")
    if 'Kobe Bryant' in targetNames:
        targetNames.remove("Kobe Bryant")

    print("*** DATA PREP DONE! ***")

    predMetrics = ["PTS_G","AST_G","TOV_G","FG%","FT%","3P_G","TRB_G","STL_G","BLK_G"]
    

    """
    setup
    """
    # user input
    donor_window_type = 'sliding'
    normalize_metric = 'variance' 
    threshold = 0.97
    #helper_metrics = ['MP']
    helper_metrics = []
    num_top = 200
    top_players = getTopPlayers(stats, pred_year, 'PTS', num_top).values.flatten().tolist()

    # setup 
    donorSetup= [normalize_metric, donor_window_type, True]
    denoiseSetup = ["SVD", "all"]
    regression_method = "pinv"
    verbose = False
    metrics_list = [[metric] + helper_metrics for metric in predMetrics]
    print(metrics_list)

    selected_targetNames = list(set(targetNames) & set(top_players))
    selected_targetNames = targetNames

    print("the number of targets tested: ", len(selected_targetNames))


    """ 
    directory path
    """
    # donor window type
    donor_window_label = donor_window_type + '/'

    # prediction method
    helper_metrics_label = ''
    if helper_metrics:
        pred_method = 'mrsc/'
        for helper_metric in helper_metrics: 
            helper_metrics_label = helper_metrics_label + helper_metric + '_'
        helper_metrics_label += '/'
    else:
        pred_method = 'rsc/'
        
    # prediction year
    pred_year_label = str(pred_year) + '/'

    # metric normalizing type
    if normalize_metric == None:
        normalize_metric_label = 'no_normalization/'
    else:
        normalize_metric_label = normalize_metric + '/'
        
    # singular value threshold energy level 
    threshold_label = str(threshold*100)[:2] + '/'

    # prediction length
    pred_length_label = str(pred_interval) + 'year/'
        
    dir_name = 'plots/' + pred_method + pred_year_label + donor_window_label + normalize_metric_label + helper_metrics_label + pred_length_label
        
    print(dir_name)
    """
    computation
    """
    print("Computing...")
    print()

    print("*** SETUP ***")
    for key, value in parameters.items():
        print("{}: {}".format(key, value))

    print("donor window type: ", donor_window_type)
    print("normalization metric: ", normalize_metric)
    print("threshold: ", threshold)
    print("helper metrics: ", helper_metrics)
    print("denoise setup: ", denoiseSetup)
    print("regression method: ", regression_method)
    print("metrics list: ", metrics_list)
    print()
    print("Experiment: {}".format(dir_name))
    print()

    all_pred, all_true, all_R2, all_bench = annual_predictions(selected_targetNames, allPivotedTableDict, donor, pred_interval, metrics_list, predMetrics,
                       threshold, donorSetup, denoiseSetup, regression_method, verbose, dir_name, top_players) 

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
