################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import os
import re
import json
import gdal
import datetime
import itertools
import shapefile
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import functions_ground_motion as fgm

import sklearn.metrics as metrics
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["ground_motion_failure"]))

ground_motion_dir = FILE_PATHS["ground_motion_csv"]
ground_motion_file = ground_motion_dir + ""
# for later: this is how the ground motion data have been saved
#out_dir_csv + 'Timeseries_GroundMotion_pixel'+str(i)+'_'+str(j)+'_failure.csv'


precip_dir = FILE_PATHS["rain_intensity_caliv_valid"]
out_dir = FILE_PATHS["time_series_ml"]

# Here comes the rain again
rain_dir = FILE_PATHS["rain_dir"]
rain_file = rain_dir + "2014-01-01_to_2019-12-31_Intensity.csv"

topo_dir = FILE_PATHS["topo_dir"]
slopefile = topo_dir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road file
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp" # this is in EPSG:32633

merged_ground_motion_distance_dir = FILE_PATHS["ground_motion_csv"]
merged_ground_motion_distance_file = merged_ground_motion_distance_dir + "merged_result.csv"


# load the data
# Probably load the data on each of the csv files as we process it
# so that the data doesn't need to be stored in memory all the time.






########### read pixel position data ##################

ground_motion_distance_pxl = pd.read_csv(merged_ground_motion_distance_file)
ground_motion_distance_pxl = np.array(ground_motion_distance_pxl)

# the first column only has indices - we don't need that.
ground_motion_distance_pxl = ground_motion_distance_pxl[:,1:]

ground_motion_distance_df = pd.DataFrame({'ground_motion': ground_motion_distance_pxl[:, 0], 'time': ground_motion_distance_pxl[:, 1], 'slope': ground_motion_distance_pxl[:,2], 'rows': ground_motion_distance_pxl[:,3],'cols': ground_motion_distance_pxl[:,4], 'distance_to_road': ground_motion_distance_pxl[:,5]})


ground_motion_distance_df['time'] =  pd.to_datetime(ground_motion_distance_df['time'])
####### need to add the rainfall data to the datafram as well. forgot to do this ###########
rain_data = pd.read_csv(rain_file)
rain_data = np.array(rain_data)
rain_data_df = pd.DataFrame({'rain_intensity_mm_sec':rain_data[:,1]})
date_list = pd.date_range(start="2014-01-01", end="2018-12-31")

rain_data_df['time'] = date_list

#print(rain_data_df.head())

# note that the ground motion data goes all the way to end of 2019 but the rainfall data is only until end of 2018
result_with_rain = pd.merge(ground_motion_distance_df, rain_data_df, how='left', on=['time'])
#print(result_with_rain.head())
result_with_rain = result_with_rain.set_index('time')
result_with_rain = result_with_rain.sort_values('time', ascending=True)
#result_with_rain.to_csv(ground_motion_dir+'merged_result_with_rain.csv')
print(result_with_rain.info(verbose=False))
# remove rows which contain missing values - these will only correspond to 2019 values, we dont have rain data for those
result_with_rain = result_with_rain.dropna()
# will bring any non numeric values to nan - easy fix- need to refine
print(result_with_rain.info(verbose=False))
#result_with_rain['ground_motion'] = result_with_rain.to_numeric(result_with_rain['ground_motion'])
result_with_rain['ground_motion'] = result_with_rain['ground_motion'].apply(pd.to_numeric, errors='coerce')
result_with_rain = result_with_rain.dropna()

print(result_with_rain.info(verbose=False))
########### MACHINE LEARNING STUFF ########################


def regression_results(y_true, y_pred):# Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


########## define training and testing data #################
X_train = result_with_rain[:'2017'].drop(['ground_motion'], axis = 1)
y_train = result_with_rain.loc[:'2017', 'ground_motion']
X_train.info(verbose=False)

X_test = result_with_rain['2018':].drop(['ground_motion'], axis = 1)
y_test = result_with_rain.loc['2018':, 'ground_motion']
X_test.info(verbose=False)

#from sklearn.model_selection import TimeSeriesSplit


# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
#models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
models.append(('KNN', KNeighborsRegressor()))
models.append(('RF', RandomForestRegressor(n_estimators = 3))) # Ensemble method - collection of many decision trees
#models.append(('SVR', SVR(gamma='auto'))) # kernel = linear

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # TimeSeries Cross validation
    tscv = TimeSeriesSplit(n_splits=3)

    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

'''
con=ground_motion_distance_pxl['time_of_motion']
ground_motion_distance_pxl['time_of_motion']=pd.to_datetime(ground_motion_distance_pxl['time_of_motion'])
ground_motion_distance_pxl.set_index('time_of_motion', inplace=True)
#check datatype of index
print(ground_motion_distance_pxl.index)

#convert to time series:
ground_motion_ts = ground_motion_distance_pxl['#Passengers']
ground_motion_ts.head(10)
'''
