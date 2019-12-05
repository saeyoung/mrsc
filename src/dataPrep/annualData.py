# third party libraries

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy
import pickle

# personal libraries
from mrsc.src.model.SVDmodel import SVDmodel
from mrsc.src.model.Target import Target
from mrsc.src.model.Donor import Donor
from mrsc.src.synthcontrol.mRSC import mRSC
from mrsc.src.dataPrep.importData import *
import mrsc.src.utils as utils

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

def edit_year_count(df):
    for player in df.Player.unique():
        df.loc[df.Player == player, 'year_count'] = np.arange(len(df[df.Player == player]))
    return df

"""
only fixes for names that appear TWICE
"""
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



def createAnnualData(params, df_recent): 
    starting_year = params[0]
    min_games = params[1]
    min_years = params[2]
    pred_year = params[3]
    pred_interval = params[4]

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


    # merge 'stats' and 'df_recent'
    stats = pd.merge(left=stats, right=df_recent, how='outer')

    # merge 'players' and 'stats'
    players = players.rename(columns={"name": "Player"})
    stats = pd.merge(stats, players, on='Player', how='left')


    # sanity check 
    stats = stats[(stats.Year >= stats.year_start) & (stats.Year <= stats.year_end)]

    stats.Year = stats.Year.astype(int)
    stats.year_start = stats.year_start.astype(int)
    stats['year_count'] = stats.Year - stats.year_start
    
    return stats

def createTargetDonors(params, stats):
    starting_year = params[0]
    min_games = params[1]
    min_years = params[2]
    pred_year = params[3]
    pred_interval = params[4]

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


    ## exclude players who played < 'min_games' in 'pred_year'
    #stats_target = stats_target[stats_target.G >= min_games]


    # create target dictionary of values
    allPivotedTableDict, allMetrics = prepareData(stats_target)

    # get target player names
    targetNames = get_active_players(stats_target, pred_year, min_years, min_games) 
    targetNames.sort()


    return donor, allPivotedTableDict, targetNames, stats




def getTopPlayers(stats, year, metric, n):
    stats = stats[stats.Year == year]
    stats = stats.groupby('Player').mean().reset_index()
    stats_sorted = stats[stats.Year == year].sort_values(metric, ascending = False).reset_index(drop=True)
    return stats_sorted[["Player"]][:n]

def getDictionaryGameByGame(data, metrics):
    my_dict = {}
    for i in range(len(metrics)):
        data_pivot = pd.pivot_table(data, values=metrics[i], index="Player", columns = "gmDate")
        shifted_df = data_pivot.apply(lambda x: pd.Series(x.dropna().values), axis=1).fillna(np.nan)
        my_dict.update({metrics[i]: shifted_df})

    return my_dict

