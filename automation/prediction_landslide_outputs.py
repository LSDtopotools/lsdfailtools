# Script to find whether the test points fail, given the set of calibration
# parameters that we have found (from the closest calibration point)

# Author: Marina Ruiz Sanchez-Oro
# Date: 15/11/2021


######################################################
######################################################
# Importing modules
######################################################
######################################################
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import rasterio
import val_functions as fn
from shapely import wkt
import geopandas as gpd
from run_json import *

FILE_PATHS = read_paths_file()
bool_lat_lon = FILE_PATHS["bool_lat_lon"]

# select from the full DEMs the pixel corresponding to the test pixels
def select_topo_data(topo_file, lons_list, lats_list):
    with rasterio.open(topo_file) as src:
        vals = [x for x in src.sample(zip(lons_list, lats_list))]
    return vals

############################################################
def comparison_with_anomalous_failure(lat_failures, lon_failures, anomalies_csv):
    anomalies = pd.read_csv(anomalies_csv)
    lat_anomalies = anomalies['Y']
    lon_anomalies = anomalies['X']
    anomalies_list = []


    for i in range(len(lat_failures)):
        lat_failure = lat_failures[i]
        lon_failure = lon_failures[i]
        # find the lat in the full anomalies lat, lon list
        is_lat_anomalous = lat_anomalies.isin([lat_failure]).any()

        if is_lat_anomalous == False:
            print('Your failure is not anomalous')
            failure = False
            anomalies_list.append(failure)
        else:
            # find the index
            lat_anomalies_df = pd.DataFrame(anomalies['Y'])
            lon_anomalies_df = pd.DataFrame(anomalies['X'])
            index_lat_anomaly = lat_anomalies.index[lat_anomalies_df['Y'] == lat_failure]
            # is the lon also the same? Let's check
            check_lon_anomaly = lon_anomalies.iloc[index_lat_anomaly]

            if check_lon_anomaly[0] == lon_failure:
                print(' Your point is anomalous and it will always fail. DONT TRUST THE PREDICTED TIME.')
                failure = True
                anomalies_list.append(failure)
            else:
                print('Your failure is not anomalous')
                failure = False
                anomalies_list.append(failure)
    return anomalies_list

############################################################
print('I am now finding the time of failure for your test points. Hold on tight.')
def find_lon_lat_failures(lats, lons, rain, depths,calibrated_multiple_point_params,demval_point,slopeval_point,failval_point,rundir ):
    lat_failures = []
    lon_failures = []
    for i in range(len(lats)):
        lat, lon = fn.get_fos_point_of_interest(rain, depths, calibrated_multiple_point_params.loc[i], lats[i], lons[i], demval_point[i][0], slopeval_point[i][0], failval_point[i][0], rundir)
        lat_failures.append(int(lat))
        lon_failures.append(int(lon))


    lat_failures = np.asarray(lat_failures)
    lon_failures = np.asarray(lon_failures)
    return lat_failures, lon_failures

# lat_failures.append(int(4558367))
# lon_failures.append(int(511315))
# print(lat_failures, lon_failures)



### the boolean file is just to load the coordinates in the right coordinate frame
# this will be the same as the one of the input epsg:4326

def get_output_csv(lat_failures, lon_failures, distance_between_points,anomalous_failures_bool, rundir):
    test_points = pd.read_csv(bool_lat_lon)
    test_points['geometry'] = test_points['geometry'].apply(wkt.loads)
    test_points_gdf = gpd.GeoDataFrame(test_points, crs='epsg:4326')


    # test some of the graphs and output variables from the validation
    for i in range(len(lat_failures)):
        lat_failure = lat_failures[i]
        lon_failure = lon_failures[i]


        FoS = np.load(f'FoS_{lat_failure}_{lon_failure}.npy')
        FoS_temp = np.load(f'FoS_temp_{lat_failure}_{lon_failure}.npy')
        min_depth = np.load(f'min_depth_{lat_failure}_{lon_failure}.npy')

        FoS_df = pd.DataFrame(FoS_temp[0,:])
        FoS_df.columns = ['FoS']


        # what is the earliest time where we see the FoS go below zero?

        day_of_failure = (FoS_df['FoS'] < 1.0).idxmax()
        print(f'The first failure is predicted on day {day_of_failure} after the start of the rainfall timeseries.')

        x_values = np.arange(0,np.shape(FoS_temp)[1])
        y_values = np.arange(0,np.shape(FoS_temp)[0])

        FoS_df['is_it_failure'] = np.where((FoS_df['FoS']>=1),0,1)
        #seaborn.scatterplot(data=FoS_df['FoS'], x=x_values, y=FoS_df['FoS'], hue=FoS_df['is_it_failure'], s=1)
        FoS_df_to_save = pd.DataFrame(FoS_df['FoS'])
        FoS_df_to_save['days'] = x_values.tolist()
        FoS_df_to_save['distance_m_from_calib_to_test_point'] = pd.Series(distance_between_points['distance_m_from_calib_to_test_point'].loc[i], index=FoS_df_to_save.index[[0]])
        FoS_df_to_save['distance_m_from_calib_to_test_point'] = FoS_df_to_save['distance_m_from_calib_to_test_point'].fillna('')
        FoS_df_to_save['day_of_failure'] = pd.Series(day_of_failure, index=FoS_df_to_save.index[[0]])
        FoS_df_to_save['day_of_failure'] = FoS_df_to_save['day_of_failure'].fillna('')
        FoS_df_to_save['anomalous_failure'] = pd.Series(anomalous_failures_bool[i], index=FoS_df_to_save.index[[0]])
        FoS_df_to_save['anomalous_failure'] = FoS_df_to_save['anomalous_failure'].fillna('')

        full_point = test_points_gdf['geometry'][i]
        full_point_x = full_point.x
        full_point_y = full_point.y
        FoS_df_to_save.to_csv(f'{rundir}_fos_timeseries_{full_point_y}_{full_point_x}.csv', index=False)
        #print(FoS_df_to_save.head(5))
        #plt.title(f'First failure day: {day_of_failure}')
        #plt.show()
        #plt.savefig(f'precip_fos_{lat_failure}_{lon_failure}.png')
        #plt.clf()
