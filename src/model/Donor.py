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
    def __init__(self, data, years):
        """
        data: (dict) dictionary of {metric : df}
        years: (df) cols = year_count, index = keys, values = calendar year
        """
        self.data = data
        self.years = years
        
        self.kyes = years.index
        
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
    
    def concat(self, metrics, pred_year, window, method = "sliding"):
        """
        metrics: (list) metrics you want to use
        pred_year: (int) the year you want to make a prediction
        window: (int) window size
        
        output: (dict) maked dict containing all values before pred_year
        """
        mask = self.years < pred_year
        df_concat = pd.DataFrame()
        for metric in metrics:
            df_valid = self.data[metric][mask]
            if (method == "sliding"):
                df_concat = pd.concat([df_concat, self.slidingWindow(df_valid, window)], axis=1)
            elif (method == "fixed"):
                df_concat = pd.concat([df_concat, self.fixedWindow(df_valid, window)], axis=1)
        return df_concat

