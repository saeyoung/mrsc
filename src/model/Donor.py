######################################################
#
# Donor
#
######################################################
import numpy as np
import pandas as pd
import random
import copy

class Donor:
    def __init__(self, data):
        """
        data: (dict) dictionary of {metric : df}
        keys: (list) player name
        """
        self.data = data        
        self.keys = list(data.values())[0].index
        
    def slidingWindow(self, df, window, p=0, sort_index = False, save=False):
        """
        df = (dataframe) pivoted df with players in rows, year_count in columns.
        window = (int) sliding window size
        p = (float, 0<=p<=1) fraction of NaN's allowed in each row
        """
        df_final = pd.DataFrame(columns = range(window))
        for i in range(df.shape[1]-window+1):
            df_window = df.iloc[:,i:(i+window)]
            df_window = df_window[np.isnan(df_window).sum(axis=1)/window <= p]
            df_window.columns = range(window)
            df_final = df_final.append(df_window)
        if (sort_index):
            df_final = df_final.sort_index()
        if (save):
            df_final.to_pickle("../data/nba-players-stats/sliding_window_{}_{}.pkl".format(window,metric))
        return df_final
    
    def fixedWindow(self, df, window, p=0, sort_index = False, save=False):
        """
        df = (dataframe) pivoted df with players in rows, year_count in columns.
        window = (int) sliding window size
        p = (float, 0<=p<=1) fraction of NaN's allowed in each row
        """

        df_final = df.iloc[:,:window]
        df_final = df_final[np.isnan(df_final).sum(axis=1)/window <= p]
        if (sort_index):
            df_final = df_final.sort_index()
        if (save):
            df_final.to_pickle("../data/nba-players-stats/fixed_window_{}_{}.pkl".format(window,metric))
        return df_final
    
    def concat(self, metrics, total_index, method = "fixed", skipNan = True, nan_index = []):
        """
        metrics: (list) metrics you want to use
        total_index: (int) window size
        method: (string) "sliding" or "fixed"
        
        output: concatenated donor matrix
        """
        df_concat = pd.DataFrame()
        window = total_index
        if (skipNan == False) & (len(nan_index) !=0):
            window = window + len(nan_index)
            index_to_keep = [n for n in range(window) if n not in nan_index]
            # print("@%&)!@#$*)@#*$)!@#", index_to_keep)
        for metric in metrics:
            if (method == "sliding"):
                df_metric = self.slidingWindow(self.data[metric], window)
            elif (method == "fixed"):
                df_metric = self.fixedWindow(self.data[metric], window)
            else:
                raise Exception("Invalid method - donor construction method sould be 'sliding' or 'fixed'.")
            
            if (skipNan == False) & (len(nan_index) !=0):
                df_metric = df_metric.iloc[:,index_to_keep]

            df_concat = pd.concat([df_concat, df_metric], axis=1)
        df_concat.columns = range(df_concat.shape[1]) #reindex

        return df_concat

    def dict(self, metrics, pred_year, window, method = "sliding"):
        """
        metrics: (list) metrics you want to use
        pred_year: (int) the year you want to make a prediction
        window: (int) window size
        
        output: (dict) donor matrix for each metric, {metric: df}
        """
        mask = self.years < pred_year
        dict_return = {}
        for metric in metrics:
            df_valid = self.data[metric][mask]
            if (method == "sliding"):
                df = self.slidingWindow(df_valid, window)
            elif (method == "fixed"):
                df = self.fixedWindow(df_valid, window)
            dict_return.update({metric:df})
        return dict_return

