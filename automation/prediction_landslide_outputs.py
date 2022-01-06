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
    """
    select_topo_data selects the value of a lat, lon location in a raster file
    :param topo_file: raster of the area of interest. Can be DEM, slope, etc
    :param lons_list: list of longitudes of the calibrated points closest to the test points
    :param lats_list: list of latitudes of the calibrated points closest to the test points
    :returns:
        - vals - value of the raster given the lat,lon point
    """
    with rasterio.open(topo_file) as src:
        vals = [x for x in src.sample(zip(lons_list, lats_list))]
    return vals


############################################################
#print('I am now finding the time of failure for your test points. Hold on tight.')
def find_lon_lat_failures(lats, lons, rain, depths, calibrated_multiple_point_params, demval_point, slopeval_point, failval_point, rundir):
    """
    find_lon_lat_failures creates arrays of lat, lon points from the given test points
    :param lats: list of latitudes of the calibrated points closest to the test points
    :param lons: list of longitudes of the calibrated points closest to the test points
    :param rain: dataframe with a timeseries of rainfall data in mm/day
    :param depths: array of soil depths for the iverson model soil stability module
    :param calibrated_multiple_point_params: dataframe with the parameters of the calibrated points closest to test points
    :param demval_point: list of values of the DEM at the lat, lon coordinates of the calibrated points
    :param slopeval_point: list of values of the slope raster at the lat, lon coordinates of the calibrated points
    :param failval_point: list of values of the failures raster at the lat, lon coordinates of the calibrated points
    :param rundir: directory where some temporary analysis files will be created
    :returns:
        - lat_failures - array of latitudes of failure points (out of the given test points in the area of interest)
        - lon_failures - array of longitudes of failure points (out of the given test points in the area of interest)
    """
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

############################################################
def comparison_with_anomalous_failure(lat_failures, lon_failures, anomalies_csv):
    """
    comparison_with_anomalous_failure compares the lat, lon of the failure points with the anomalous failures to check if they are anomalous.
    :param lat_failures: array of latitudes of failure points (out of the given test points in the area of interest)
    :param lon_failures: array of longitudes of failure points (out of the given test points in the area of interest)
    :param anomalies_csv: csv file with the detected anomalous failure points that always fail
    :returns:
        - anomalies_list - Boolean list. Elements are True if failure is anomalous, False otherwise.
    """
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


### the boolean file is just to load the coordinates in the right coordinate frame
# this will be the same as the one of the input epsg:4326

def get_output_csv(lat_failures, lon_failures, distance_between_points, anomalous_failures_bool, rundir):
    """
    get_output_csv creates a csv file with a timeseries of factor of safety for each test point, as well as the distance
    from the chosen calibrated point, the day of failure and whether the failure is anomalous or not.
    :param lat_failures: array of latitudes of failure points (out of the given test points in the area of interest)
    :param lon_failures: array of longitudes of failure points (out of the given test points in the area of interest)
    :param distance_between_points: dataframe with the distance (m) between then test points and the calibrated points
    :param anomalous_failures_bool: Boolean list. Elements are True if failure is anomalous, False otherwise.
    :param rundir: directory where the output files will be created
    """
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
        FoS_df_to_save.to_csv(f'{rundir}fos_timeseries_{full_point_y}_{full_point_x}.csv', index=False)
        #print(FoS_df_to_save.head(5))
        #plt.title(f'First failure day: {day_of_failure}')
        #plt.show()
        #plt.savefig(f'precip_fos_{lat_failure}_{lon_failure}.png')
        #plt.clf()
