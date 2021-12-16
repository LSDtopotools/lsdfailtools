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
from shapely import wkt
import geopandas as gpd
from prediction_landslide_outputs import *
from run_json import *

############################################
# Script with function that will run functions needed to get the outputs from
# the simulation. It will use the files created in get_closest_calib_point.py
# The output is the timeseries with the Factor of safety for each point
# Marina Ruiz Sanchez-Oro
# 10/12/2021
############################################
FILE_PATHS = read_paths_file()
bool_lat_lon = FILE_PATHS["bool_lat_lon"]
#rundir = FILE_PATHS["rundir"]
# parameter files
# the params used to define the physical soil properties in the iverson MC runs
Iverson_MC_params_file = FILE_PATHS["iverson_param"]

# observed failure data files
# don't need this anymore!
failfile = FILE_PATHS["ground_motion_failure"]

# topography files
demfile = FILE_PATHS["dem_file"]
slopefile = FILE_PATHS["slope_file"]
closest_cal_points = FILE_PATHS["closest_cal_points"]
points_in_buffer = FILE_PATHS["points_in_buffer"]
anomaly_failures = FILE_PATHS["anomaly_failures"]

def landslide_output_from_rain(rainfall_file, rundir):
    ##########################################################################
    # 0. Load rasters into arrays for DEM, slope, failtimes and prefailtimes for a given failure threshold. Let's use 80mm/yr for now.
    demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
    slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
    failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)

    # select the point of interest from the raster files.
    #'./test_closest_calibration_points.csv' - this is the new file instead of the one with the single point
    calibrated_multiple_points_path = closest_cal_points
    calibrated_multiple_point_params = pd.read_csv(calibrated_multiple_points_path, index_col=None)

    lons = calibrated_multiple_point_params['lon_test'].to_list()
    lats = calibrated_multiple_point_params['lat_test'].to_list()
    ##########################################################################
    # values of the corresponding points
    demval_point = select_topo_data(demfile, lons, lats)
    slopeval_point = select_topo_data(slopefile, lons, lats)
    failval_point = select_topo_data(failfile, lons, lats)


    print('Now we have all the points we need for our analysis.')
    ##########################################################################
    ############################################################

    # Read the Iverson params
    Iverson_MC_params = pd.read_csv(Iverson_MC_params_file)
    depths  = np.arange(Iverson_MC_params.at[0,'depth'], Iverson_MC_params.at[1,'depth'], 0.2)


    ###################### RAINFALL DATA #######################
    # We are assuming that the rainfall data is the same for all the points
    # the area of interest hasa rough length of 30km which is the resolution of the
    # precipitaiton data we have.
    rainfile = rainfall_file
    #"/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/2014-01-01_to_2019-12-31_Intensity.csv"

    rain = pd.read_csv(rainfile)

    rainlist = [0]
    for i in range(1,len(rain)):
    	rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
    rain['time_s'] = rainlist
    rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']
    ############################################################
    #anomalies_list = comparison_with_anomalous_failure( anomalies_csv)
    lat_failures, lon_failures = find_lon_lat_failures(lats, lons, rain, depths,calibrated_multiple_point_params,demval_point,slopeval_point,failval_point,rundir)
    ###########################################################
    distance_between_points_file = points_in_buffer
    distance_between_points = pd.read_csv(distance_between_points_file, index_col=None)

    anomalous_failures_bool = comparison_with_anomalous_failure(lat_failures, lon_failures, anomaly_failures)
    ###########################################################
    get_output_csv(lat_failures, lon_failures, distance_between_points,anomalous_failures_bool, rundir)
