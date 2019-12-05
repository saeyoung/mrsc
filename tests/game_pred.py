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



def test():
    """
    import data
    """
    print("*** importing data ***")

    annual_pred = pd.read_pickle("annual_pred_2016.pkl")
    target_players = list(annual_pred.columns)

    data = pd.read_csv("../data/nba-enhanced-stats/2012-18_playerBoxScore.csv")

    game_metrics = ['playPTS', 'playAST', 'playTO','playFG%','playFT%','play3PM','playTRB','playSTL', 'playBLK']
    year_metrics = ['PTS_G','AST_G','TOV_G','TRB_G','STL_G','BLK_G','3P_G','FG%','FT%']
    colname_dict = {'playPTS': 'PTS_G', 'playAST': 'AST_G', 'playTO':'TOV_G',
                    'playFG%': 'FG%','playFT%':'FT%','play3PM':'3P_G',
                    'playTRB':'TRB_G','playSTL':'STL_G','playBLK':'BLK_G'}

    # edit column names to fit with the yearly data
    data = data.rename(columns=colname_dict)

    date_col = pd.to_datetime(data.gmDate + " " + data.gmTime, format='%Y-%m-%d %H:%M').rename("date")
    data = pd.concat([date_col,data], axis=1)

    stats_game = data[["date","gmDate","playDispNm"]+year_metrics]
    stats_game = stats_game.rename(columns={"playDispNm": "Player"})

    print(data.columns)


    # for playerName in playerNames:
    #     # this player
    #     playerGames = stats_game[stats_game.Player == playerName]

    #     # 2015-2016 season
    #     playerGames = playerGames[(playerGames.gmDate <= '2016-04-30') & (playerGames.gmDate >= '2015-10-01')]

    #     mask = playerGames[year_metrics] != 0
    #     mae = np.abs(playerGames[year_metrics].sub(year_pred[playerName].values, axis='columns'))/playerGames[year_metrics][mask]
    #     mape = mae.mean().to_frame(name=playerName)
    #     gamely_variation_mape = pd.concat([gamely_variation_mape, mape], axis=1)

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
