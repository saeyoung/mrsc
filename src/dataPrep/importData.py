######################################################
#
# to manage stats.csv
#
######################################################
import numpy as np
import pandas as pd
import random
import copy

# for team predictoin

def getData(pre1, pre2, metrics, game_ids):
    """
        pre1 = (string) target or donor
        pre2 = (string) home or away
        metrics = (list) list of metrics
    """
    prefix = pre1+ "_" + pre2 + "_"
    df = pd.DataFrame()
    for i in range(len(metrics)):
        bucket = pd.read_pickle("../data/nba-hosoi/"+ prefix +metrics[i]+".pkl")
        df = pd.concat([df, bucket], axis = 1)
    df = df[df.index.isin(game_ids)]
    print("DataFrame size ", df.shape, "was created.")
    return df

######################################################
   
# for individual player prediction 
def getMetricPerGame(stats, metric, newColName):
    """
    stats: (df) stats dataframe
    metric: (string) column name of stats df
    newColName: (string) new column name for the processed data
    
    output: (df) for each player, for each year, the per-game metric is computed.
    """
    columnsOfInterest = ["year_count", "Player", "G", metric]
    df = stats.loc[:,columnsOfInterest].groupby(["Player","year_count"]).sum()
    df[newColName] = df[metric]/df["G"]
    return df.iloc[:,-1:]

def getMetricsPerGameDict(stats, metrics):
    """
    stats: (df) stats dataframe
    metrics: (list) column names (strings) of stats df
    
    output: (dict) dict of df's.
    """
    metricsPerGameDict = {}
    for metric in metrics:
        newColName = metric+"_G"
        metricsPerGameDict.update({newColName : getMetricPerGame(stats, metric, newColName)})
    return metricsPerGameDict

def getMetricPerCent(stats, metric, newColName):
    """
    recalculating per cent using the metric and metric attempted.
    """
    attempts = metric+"A"
    columnsOfInterest = ["year_count", "Player", metric, attempts]

    df = stats.loc[:,columnsOfInterest].groupby(["Player","year_count"]).sum()
    df[newColName] = df[metric]/df[attempts]
    df.loc[df[attempts] == 0, newColName] = 0    # 0 if no attempt
    return df.iloc[:,-1:]

def getMetricsPerCentDict(stats, metrics):
    metricsPerCentDict = {}
    for metric in metrics:
        newColName = metric+"%"
        metricsPerCentDict.update({newColName : getMetricPerCent(stats, metric, newColName)})
    return metricsPerCentDict

def getMetricsWeighted(stats, metric, newColName):
    """
    weighted average based on the number of games played
    """
    columnsOfInterest = ["year_count", "Player", "G", metric]
    df = stats.loc[:,columnsOfInterest]
    g_sum = stats.loc[:,["Player","year_count","G"]].groupby(["Player","year_count"]).sum().reset_index()
    df = df.merge(g_sum, on=["year_count","Player"], how="left")
    df[newColName] = df[metric]*df["G_x"]/df["G_y"]
    df = df.groupby(["Player","year_count"]).sum().iloc[:,-1:]
    return df

def getMetricsWeightedDict(stats, metrics):
    metricsWeightedDict = {}
    for metric in metrics:
        newColName = metric+"_w"
        metricsWeightedDict.update({newColName : getMetricsWeighted(stats, metric, newColName)})
    return metricsWeightedDict

def getPivotedTable(metric, metricsDict):
    """
    metric: (string) column name of df's in metricsDict
    metricsDict: (dict) df's - in the form of the output of getMetrics~~(stats, metrics)
    
    output: (df) pivoted table with players in rows and year count in columns.
    """
    df = metricsDict[metric].reset_index()
    return pd.pivot_table(df, values=metric, index="Player", columns = "year_count")

def getPivotedTableDict(metricsDict):
    """
    metricsDict: (dict) dict of df's
    """
    pivotedTableDict = {}
    for metric in metricsDict.keys():
        pivotedTableDict.update({metric: getPivotedTable(metric, metricsDict)})
    return pivotedTableDict

##############################################

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

def dict_to_list(my_dict, get_keys = False):
    """
    my_dict: (dict) your dictionary
    
    output: (list) values
            (list, list) keys and values if get_keys == True
    """
    if get_keys:
        return list(my_dict.values()), list(my_dict.keys())
    return list(my_dict.values())

def getSlidingWindowList(pivotedTableDict, window):
    """
    outputs sliding window style donor pool
    """
    slidingWindowList = []
    values, keys = dict_to_list(pivotedTableDict, get_keys = True)
    for i in range(len(keys)):
        slidingWindowList.append(slidingWindowDF(values[i], window, p=0, save=False))
    return slidingWindowList

