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

# def getMetrics(target, donor, pred_year, allMetrics, threshold, expSetup):
#     target_data = target.concat(allMetrics, pred_year, pred_length=1)

#     num_k = len(allMetrics)
#     total_index = int(target_data.shape[1] / num_k)
#     mat_form_method = expSetup[0]

#     metrics_list = []
#     for metric_of_interest in allMetrics:
#         df = donor.concat([metric_of_interest], 2016, total_index, method = mat_form_method)
#         apprx_rank = utils.approximate_rank(df, t = threshold)
#         energy_captured = utils.svdAanlysis(df, title=metric_of_interest, verbose = False, k=apprx_rank)[apprx_rank-1]

#         metrics = [metric_of_interest]
#         candidates = copy.deepcopy(allMetrics)
#         candidates.remove(metric_of_interest)

#         while True:
#             energy_diff_df = pd.DataFrame()
#             for metric in candidates:
#                 comb = metrics+[metric]
#                 df = donor.concat(comb, 2016, total_index, method = mat_form_method)
#                 energy_at_apprx_rank = utils.svdAanlysis(df, k=apprx_rank, verbose=False)[-1]
				
#                 # if(energy_at_apprx_rank > threshold):
#                 if(energy_at_apprx_rank > energy_captured):
#                     energy_diff = np.abs(energy_at_apprx_rank - energy_captured)
#                     energy_diff_df = pd.concat([energy_diff_df, pd.DataFrame([[energy_diff]], index=[metric])], axis=0)
#         #             print(energy_diff)
#             if (energy_diff_df.shape[0] == 0):
#                 break
#             new_metric = energy_diff_df.sort_values(0).index[0]
#             metrics = metrics + [new_metric]
#             candidates.remove(new_metric)
#         metrics_list.append(metrics)
#         print(metrics)

#     return metrics_list

def getMetrics(target, donor, pred_year, allMetrics, threshold, expSetup, boundary = "threshold"):
	target_data = target.concat(allMetrics, pred_year, pred_length=1)

	num_k = len(allMetrics)
	total_index = int(target_data.shape[1] / num_k)
	mat_form_method = expSetup[0]
	
	df_result = pd.DataFrame(0, columns = allMetrics, index = allMetrics)
	for metric_of_interest in allMetrics:
		df = donor.concat([metric_of_interest], 2016, total_index, method = mat_form_method)
		apprx_rank = utils.approximate_rank(df, t = threshold)
		energy_captured = utils.svdAanlysis(df, title=metric_of_interest, verbose = False)[apprx_rank-1]

		if (boundary == "threshold"):
			b = threshold
		elif (boundary == "energy"):
			b = energy_captured
		else:
			raise Exception("wrong parameter")

		metrics = [metric_of_interest]
		candidates = copy.deepcopy(allMetrics)
		candidates.remove(metric_of_interest)

		while True:
			energy_diff_df = pd.DataFrame()
			for metric in candidates:
				comb = metrics+[metric]
				df = donor.concat(comb, 2016, total_index, method = mat_form_method)
				energy_at_apprx_rank = utils.svdAanlysis(df, k=apprx_rank, verbose=False)[-1]
				
				if(energy_at_apprx_rank > b):
					energy_diff = np.abs(energy_at_apprx_rank - energy_captured)
					energy_diff_df = pd.concat([energy_diff_df, pd.DataFrame([[energy_diff]], index=[metric])], axis=0)
		#             print(energy_diff)
			if (energy_diff_df.shape[0] == 0):
				break
			new_metric = energy_diff_df.sort_values(0).index[0]
			metrics = metrics + [new_metric]
			candidates.remove(new_metric)
		df_result.loc[metric_of_interest,metrics] = 1
		
	metrics_list =[]
	for i in range(num_k):
		a = ((df_result.iloc[i,:]==1) & (df_result.iloc[:,i]==1))
		metrics_list.append(a.index[a].values.tolist())

	return metrics_list

def getWeitghts(target, donor, metrics_list, expSetup, method = "mean"):   
	# get mat_form_method
	mat_form_method = expSetup[0] # "fixed"
	
	# get weights for metrics
	weights_list = []
	for metrics in metrics_list:
		target_data = target.concat(metrics, 2016, pred_length=1)
		num_k = len(metrics)
		total_index = int(target_data.shape[1] / num_k)
		donor_data = donor.concat(metrics, 2016, total_index, method = mat_form_method)
	
		if (method == "mean"):
			weights = []
			for i in range(num_k):
				weights.append(1/(donor_data.iloc[:,i*total_index:(i+1)*total_index].mean().mean()))
			weights_list.append(weights)
		elif (method == "var"):
			weights = []
			for i in range(num_k):
				weights.append(1/(1+np.var(donor_data.iloc[:,i*total_index:(i+1)*total_index].to_numpy().flatten())))
			weights_list.append(weights)
		else:
			raise ValueError("invalid method")
	return weights_list

def test():
	"""
	import data
	"""
	print("* importing data")
	players = pd.read_csv("../data/nba-players-stats/player_data.csv")
	players = players[players.year_start >= 1980] # only choose players who started after 1980
	players["player_id"] = range(0,len(players.name)) # assign id

	stats = pd.read_csv("../data/nba-players-stats/Seasons_Stats.csv")
	stats = stats[stats.Player.isin(players.name)]

	# only after 1980
	stats = stats[stats.Year >= 1980]

	# without duplicated names --> to do: how to distinguish multiple player with the same name
	stats = removeDuplicated(players, stats)
	stats.Year = stats.Year.astype(int)
	stats.year_count = stats.year_count.astype(int)

	print("* preparing data")
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

	# this matrix will be used to mask the table
	df_year = pd.pivot_table(stats, values="Year", index="Player", columns = "year_count")

	"""
	experiment setup
	"""
	pred_year = 2016
	# targets
	activePlayers = getActivePlayers(stats, pred_year, 4)
	activePlayers.sort()
	activePlayers.remove("Kevin Garnett")
	activePlayers.remove("Kobe Bryant")
	
	# overall setup
	expSetup = ["sliding", "SVD", "all", "pinv", False]
	threshold = 0.97

	###################################################
	# offMetrics = ["PTS_G","AST_G","TOV_G","PER_w", "FG%","FT%","3P_G"]
	# defMetrics = ["TRB_G","STL_G","BLK_G"]
	# metrics_list = [offMetrics, defMetrics]
	###################################################

	print("* start experiment")
	pred_all = pd.DataFrame()
	true_all = pd.DataFrame()
	for playerName in activePlayers:
		print()
		target = Target(playerName, allPivotedTableDict, df_year)
		donor = Donor(allPivotedTableDict, df_year)

		metrics_list = getMetrics(target, donor, pred_year, allMetrics, threshold, expSetup, boundary="threshold")
		weights_list = getWeitghts(target, donor, metrics_list, expSetup, method="var")

		print(metrics_list)
		
		mrsc = mRSC(donor, target, probObservation=1)

		player_pred = pd.DataFrame()
		player_true = pd.DataFrame()
		for i in range(len(metrics_list)):
		    mrsc.fit_threshold(metrics_list[i], weights_list[i], pred_year, pred_length = 1, threshold = threshold, setup = expSetup)

		    pred = mrsc.predict()
		    true = mrsc.getTrue()
		    pred.columns = [playerName]
		    true.columns = [playerName]

		    print("pred")
		    print(pred)
		    print("true")
		    print(true)

		    player_pred = pd.concat([player_pred, pred.loc[allMetrics[i],:]], axis=0)
		    player_true = pd.concat([player_true, pred.loc[allMetrics[i],:]], axis=0)

		pred_all = pd.concat([pred_all, player_pred], axis=1)
		true_all = pd.concat([true_all, player_true], axis=1)

	###################
	mask = (true_all !=0 )
	mape = np.abs(pred_all - true_all) / true_all[mask]

	print(mape)
	print()
	print("*** MAPE ***")
	print(mape.mean(axis=1))
	print("MAPE for all: ", mape.mean().mean())

	rmse = utils.rmse_2d(pred_all, true_all)
	print()
	print("*** RMSE ***")
	print(rmse)
	print("RMSE for all: ", rmse.mean())

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
