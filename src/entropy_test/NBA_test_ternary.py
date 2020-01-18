#############################################################
#
# Test 2. regression model for a fixed input string size N, Ternary
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import copy
import pickle
from math import log, e, pi
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from utils import *
from ternary import *

def import_data():
    print("*** importing data ***")

    annual_pred = pd.read_pickle("annual_pred_2016.pkl")
    target_players = list(annual_pred.columns)

    data = pd.read_csv("../../data/nba-enhanced-stats/2012-18_playerBoxScore.csv")

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

    df = pd.read_pickle("../../data/nba-hosoi/nba_scores_2103-2018.pkl")
    df = df[["nbaId","path","game_date","home","away","season"]].drop_duplicates().reset_index(drop=True)

    a = pd.concat([df,(df["game_date"] + str(" ") + df["home"]).rename("key")], axis=1)
    b = pd.concat([df,(df["game_date"] + str(" ") + df["away"]).rename("key")], axis=1)
    appended = pd.concat([a,b], axis=0)

    new_data = pd.concat([data, (data["gmDate"] + str(" ") + data["teamAbbr"]).rename("key")], axis=1)
    data_fin = new_data.merge(appended, how='left', left_on='key', right_on='key')

    stats_game = data_fin[["date","gmDate","gmTime","nbaId","playDispNm"]+year_metrics]
    stats_game = stats_game.rename(columns={"playDispNm": "Player"})

    return stats_game

def test():
    players = ['LeBron James', 'Kevin Durant', 'Stephen Curry', 'Russell Westbrook', 'James Harden', 
              'Giannis Antetokounmpo', 'Anthony Davis', 'Jimmy Butler', 'Draymond Green', 'Chris Paul',
              'Klay Thompson', 'John Wall', 'Paul George', 'DeMarcus Cousins', 'Rudy Gobert', 'Kyle Lowry',
              'Paul Millsap', 'Blake Griffin', 'Damian Lillard', 'DeAndre Jordan', 'Kyrie Irving',
              'Al Horford', 'DeMar DeRozan', 'Kevin Love', 'Andre Drummond', 'Carmelo Anthony', 'LaMarcus Aldridge',
              'Kemba Walker', 'Eric Bledsoe', 'Dwight Howard', 'Eric Gordon', 'George Hill', 'Jeff Teague', 
              'Andrew Wiggins', 'Serge Ibaka', 'Avery Bradley', 'Trevor Ariza', 'Devin Booker', 'Bradley Beal',
              'Karl-Anthony Towns', 'Marc Gasol', 'Khris Middleton']
    metric = "PTS_G"
    alpha = 1
    window = 50
    plot = False
    #############
    stats_game = import_data()

    entropy_list=[]
    lbs=[]
    # new_lbs =[]
    for player_name in players[:2]:

        for alpha in [0.25, 0.5, 0.75, 1, 1.25]:
            print(player_name)
            player_data = stats_game[stats_game.Player == player_name][metric].reset_index(drop=True)
            player_gamma, upper, lower = get_gamma(player_data, player_data, alpha, window)
            size = len(player_gamma)
            p0 = np.sum(np.array(player_gamma) == -1)/size
            p1 = np.sum(np.array(player_gamma) == 0)/size
            p2 = np.sum(np.array(player_gamma) == 1)/size
            p = [p0, p1, p2]
            ub = p0+p2
            
            # show the data
            if plot:
                plt.title("{}, size={} \n window={}, alpha={}".format(player_name, size, window, alpha))
                plt.plot(upper, color="grey")
                plt.plot(lower, color="grey")
                plt.plot(player_data, color="orange")
                plt.show()
                
            # convert [-1, 0, 1] to [0, 1, 2]
            player_gamma_uncomp = list(np.array(player_gamma) + 1)
            player_gamma_string = list_to_string(player_gamma_uncomp)
            # compress
            compressed = compress(player_gamma_string)
            compression_ratio = len(compressed)/len(player_gamma_string)    
            # entropy
            ent = get_entropy_ternary(size, compression_ratio, player_name, plot)
            entropy_list.append(ent)

            # lower bound
            lb = g_inverse(ent, a=0.005)
            lbs.append(lb)
            # another lower bound
            # new_lb = 1/(2*pi*e) * (e ** (2*ent))
            # new_lbs.append(new_lb)

            # a = g_inverse(ent-0.04, a=0.005)
            # b = g_inverse(ent+0.04, a=0.005)

            print("alpha = ", alpha)
            print("window = ", window)
            print("observed p = ", np.round(p,3))
            print("length = ", size)
            print("compression ratio: ", compression_ratio)
            print("estimated entropy: ", ent)
            print("P(e) lower bound : ", lb)
            print("1 - largest p    : ", 1-np.max(p))
            # print("P(e) range       : ", a, b)
            # print("P(e) upper bound : ", ub)
            print("--------------------------")
            print()
    
    # plt.title("Entropy distribution \n window={}, alpha={}".format(window, alpha))
    # plt.hist(entropy_list)
    # plt.axvline(np.mean(entropy_list), color="red", label="mean={}".format(np.mean(entropy_list).round(3)))
    # plt.legend()
    # plt.show()

    # plt.title("Error lower bounds distribution \n window={}, alpha={}".format(window, alpha))
    # plt.hist(lbs)
    # plt.axvline(np.mean(lbs), color="red", label="mean={}".format(np.mean(lbs).round(3)))
    # plt.legend()
    # plt.show()


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
