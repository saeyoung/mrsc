######################################################
#
# Utility functions
#
######################################################
import numpy as np
import pandas as pd
import random
import copy

from sklearn.metrics import mean_squared_error
from numpy.linalg import svd, matrix_rank, norm

def getIndivFromDict(keys, data):
    """
    keys: (list) list of keys - the key(index) for each individual. Different from the dictionary keys.
    data: (dict) dictionary of {metric : df}
    """
    metrics = data.keys()
    newDict = {}
    for metric in metrics: 
        newDict.update({metric: data[metric].iloc[data[metric].index.isin(keys),:]})
    return newDict


def get_preint_data(combinedDF, intervIndex, totalIndex, nbrMetrics, reindex = True):
    """
    Input:
        combinedDF: (dataframe) concatenated df of size (N, nbrMetrics*totalIndex)
        intervIndex: pre-int period
        totalIndex: total period
        nbrMetrics: number of metrics
    
    Output:
        pre intervention of all metrics, concatenated
    """
    if reindex:
        combinedDF.columns = range(combinedDF.shape[1])
    indexToChoose = []
    for k in range(nbrMetrics):
        indexToChoose = indexToChoose + list(range(k*totalIndex,k*totalIndex + intervIndex))
    return combinedDF.loc[:,indexToChoose]

def get_postint_data(combinedDF, intervIndex, totalIndex, nbrMetrics, reindex = True):
    
    """
    Input:
        combinedDF: (dataframe) concatenated df of size (N, nbrMetrics*totalIndex)
        intervIndex: pre-int period
        totalIndex: total period
        nbrMetrics: number of metrics

    Output:
        pre intervention of all metrics, concatenated
    """
    if reindex:
        combinedDF.columns = range(combinedDF.shape[1])
    indexToChoose = []
    for k in range(nbrMetrics):
        indexToChoose = indexToChoose + list(range(k*totalIndex + intervIndex, (k+1)*totalIndex))
    return combinedDF.loc[:,indexToChoose]

def approximate_rank(X, t=99):
    """
    Input:
        X: matrix of interest
        t: an energy threshold. Default (99%)
        
    Output:
        r: approximate rank of Z
    """
    u, s, v = np.linalg.svd(X, full_matrices=False)
    total_energy = (100*(s**2).cumsum()/(s**2).sum())
    r = list((total_energy>t)).index(True)+1
    return r


#############################################

# def hsvt(df, rank): 
#     """
#     Input:
#         df: matrix of interest
#         rank: rank of output matrix
#     Output:
#         thresholded matrix
#     """
#     u, s, v = np.linalg.svd(df, full_matrices=False)
#     s[rank:].fill(0)
#     vals = (np.dot(u*s, v))
#     return pd.DataFrame(vals, index = df.index, columns = df.columns)



# def combined_df(df_list):
#     keys = df_list[0].index
#     combined_df = pd.DataFrame()
#     for i in range(len(df_list)):
#         combined_df = pd.concat([combined_df,df_list[i].reset_index(drop=True)], axis=1)
#     return keys, combined_df

# def dict_to_list(my_dict, get_keys = False):
#     """
#     my_dict: (dict) your dictionary
    
#     output: (list) values
#             (list, list) keys and values if get_keys == True
#     """
#     if get_keys:
#         return list(my_dict.values()), list(my_dict.keys())
#     return list(my_dict.values())

# def getSlidingWindowList(pivotedTableDict, window):
#     """
#     outputs sliding window style donor pool
#     """
#     slidingWindowList = []
#     values, keys = dict_to_list(pivotedTableDict, get_keys = True)
#     for i in range(len(keys)):
#         slidingWindowList.append(slidingWindowDF(values[i], window, p=0, save=False))
#     return slidingWindowList

# def mse_2d(y, y_pred):
#     # y, y_pred are df (2d)
#     return ((y - y_pred) ** 2).mean(axis=1)

# def rmse_2d(y, y_pred):
#     # y, y_pred are df (2d)
#     return np.sqrt(((y - y_pred) ** 2).mean(axis=1))

# def mse(y, y_pred):
#     # y, ypred are 1d array
#     return np.mean((y - y_pred) ** 2)

# def rmse(y, y_pred):
#     # y, ypred are 1d array
#     return np.sqrt(mse(y,y_pred))

# def mape(y, y_pred):
#     mask = (y != 0)
#     return np.mean(np.abs((y - y_pred)[mask] / y[mask]))

# # data prep



# ########################
# def getDonorTargetDf(stats_not_duplicated, metric, pred_year, target_id):
#     """
#     stats_not_duplicated = (dataframe) stats df
#     metric = (string) metric of interest (column name of stats_not_duplicated)
#     window = (int) sliding window size
#     """
    
#     # data up to pred_year
#     stats_this_year = stats_not_duplicated[stats_not_duplicated.Year <= pred_year]
    
#     #target
#     stats_target = stats_this_year[stats_this_year.player_id == target_id]
#     num_years = stats_target.iloc[-1, -1]
#     target_pivot = pd.pivot_table(stats_target, values=metric, columns=['year_count'],index=['Player'])
#     if(np.isnan(target_pivot).sum().sum() != 0):
#         raise("NaN value in target")

#     # donor
    
#     stats_donor = stats_this_year[stats_this_year.year_count <= num_years] # only who played more than num_years
#     stats_donor = stats_donor[stats_donor.player_id != target_id] 
#     donor_pivot = pd.pivot_table(stats_donor, values=metric, columns=['year_count'],index=['Player'])
#     donor_pivot = donor_pivot[~donor_pivot[num_years].isnull()]
#     donor_pivot = donor_pivot.T.fillna(donor_pivot.mean(axis=1)).T
    
#     return donor_pivot, target_pivot

# def topPlayers(stats, year, metric, n):
#     stats = stats[stats.Year == year]
#     stats = stats.groupby('Player').mean().reset_index()
#     stats_sorted = stats[stats.Year == year].sort_values(metric, ascending = False).reset_index(drop=True)
#     return stats_sorted[["Player","player_id"]][:n]

# # plots
# def plotPrediction(df_true, df_pred, metric, name, num_sv = 0):
#     """
#     Plot the groundtruth and prediction of each year.
#     (Only the last dot is a test datapoint, all the datapoints before the marker are train datapoints)
#     """
#     title = "Target Player: "+ name
#     if num_sv != 0:
#         title = title + "; HSVT: "+ str(num_sv)
#     else:
#         title = title + "; no HSVT"
#     markers_on = [df_true.shape[0]-1]
#     plt.plot(df_pred, "blue", marker = 'o', markevery=markers_on)
#     plt.plot(df_true, "red", marker = 'o', markevery=markers_on)
#     plt.xlabel("year")
#     plt.ylabel(metric)
#     plt.legend(["Prediction","Truth"])
#     plt.title(title)
#     plt.show()
    
# def plotPredictionEachYear(df_true, df_pred, metric, name, num_sv):
#     """
#     Plot the groundtruth and prediction of each year. (Each dot means new prediction)
#     """
#     title = "Target Player: "+ name
#     if num_sv != 0:
#         title = title + "; HSVT: "+ str(num_sv)
#     else:
#         title = title + "; no HSVT"
#     plt.plot(df_pred, "blue", marker = 'o')
#     plt.plot(df_true, "red", marker = 'o')
#     # plt.plot(df_numdoner, "grey", linestyle = "dashed")
#     plt.xlabel("year")
#     plt.ylabel(metric)
#     plt.legend(["Prediction","Truth"])
#     plt.title(title)
#     plt.show()

# def plotMape(df_true, df_pred, name, num_sv):
#     df_mape = pd.DataFrame(mape(df_true.values, df_pred.values), index = df_true.index)
    
#     title = "Target Player: "+ name
#     if num_sv != 0:
#         title = title + "; HSVT: "+ str(num_sv)
#     else:
#         title = title + "; no HSVT"
    
#     plt.plot(df_mape, "gold", marker = 'o')
#     plt.xlabel("year")
#     plt.ylabel("MAPE")
#     plt.legend(["MAPE", "Doner Pool Size"])
#     plt.title(title)
#     plt.show()

# def plotDonorSize(df_numdonor, name, num_sv):
#     df_mape = pd.DataFrame(mape(df_true.values, df_pred.values), index = df_true.index)
    
#     title = "Target Player: "+ name
#     if num_sv != 0:
#         title = title + "; HSVT: "+ str(num_sv)
#     else:
#         title = title + "; no HSVT"
        
#     plt.plot(df_numdonor, "grey", linestyle = "dashed", marker = 'o')
#     plt.xlabel("year")
#     plt.ylabel("size")
#     plt.legend(["Doner Pool Size"])
#     plt.title(title)
#     plt.show()
