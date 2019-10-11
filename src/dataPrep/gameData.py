from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy
import pickle

""" Read in Games Data and create DataFrame """
def createGameData():
    df = pd.read_csv("../data/nba-enhanced-stats/2012-18_playerBoxScore.csv")
    
    game_metrics = ['playPTS', 'playAST', 'playTO','playFG%','playFT%','play3PM','playTRB','playSTL', 'playBLK']
    year_metrics = ['PTS_G','AST_G','TOV_G','TRB_G','STL_G','BLK_G','3P_G','FG%','FT%']
    colname_dict = {'playPTS': 'PTS_G', 'playAST': 'AST_G', 'playTO':'TOV_G',
                    'playFG%': 'FG%','playFT%':'FT%','play3PM':'3P_G',
                    'playTRB':'TRB_G','playSTL':'STL_G','playBLK':'BLK_G'}

    # edit column names to fit with the yearly data
    df = df.rename(columns=colname_dict)

    # add date column
    date_col = pd.to_datetime(df.gmDate + " " + df.gmTime, format='%Y-%m-%d %H:%M').rename("date")
    df = pd.concat([date_col,df], axis=1)

    # relevant columns
    cols = ['date', 'gmDate', 'playDispNm', 'teamAbbr', 'teamLoc', 'teamRslt', 'teamDayOff', 'playStat', 'playMin', 'opptAbbr', 'opptDayOff']
    df_games = df[cols + year_metrics]
    df_games = df_games.rename(columns={"playDispNm": "Player"})

    # fix special characters
    df_games.Player = df_games.Player.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    return df_games


""" Split Game DataFrame into Train/CV/Test Dataframes """
def splitDFs(df, trainDates, cvDates, testDates): 
    # Training Dataframe
    d1 = trainDates[0]
    d2 = trainDates[1]
    dfTrain = df[(df.gmDate >= d1) & (df.gmDate <= d2)]

    # Validation Dataframe
    d1 = cvDates[0]
    d2 = cvDates[1]
    dfCV = df[(df.gmDate >= d1) & (df.gmDate <= d2)]
    
    # Validation Dataframe
    d1 = testDates[0]
    d2 = testDates[1]
    dfTest = df[(df.gmDate >= d1) & (df.gmDate <= d2)]
    
    return dfTrain, dfCV, dfTest

