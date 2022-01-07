import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import rasterio
from shapely import wkt
import geopandas as gpd

from .find_closest_calibrated_point import *

############################################
# Script with function that will run functions needed to get the closest calibrated points from the
# list of lat, lon points that are in the area of interest (output from get_lat_lon_check.py)
# Marina Ruiz Sanchez-Oro
# 10/12/2021
############################################

def run_find_closest_calibrated_point(rundir, file_paths):
    """
    run_find_closest_calibrated_point finds the calibrated points closest to the test points.
    """

    dem_file = file_paths["dem_file"]
    calibration_file = file_paths["calibration_file"]
    transform_bil = f'{rundir}/test_transform.bil'
    transform_csv = f'{rundir}/test_transform.csv'
    bool_lat_lon = f'{rundir}/bool_lat_lon.csv'
    closest_cal_points = f'{rundir}/test_closest_calibration_points_add_coords.csv'
    points_in_buffer = f'{rundir}/test_points_within_buffer_distance.csv'

    # find_closest_calibrated_point
    convert_calib_to_lat_lon(dem_file,\
    calibration_file,\
    transform_bil)

    convert_calib_raster_to_csv_shp(transform_bil)

    multipoint, selected_rows = create_calib_multipoint(transform_csv)

    calib_params_closest_point(multipoint,selected_rows,bool_lat_lon, calibration_file,\
    closest_cal_points)

    get_points_in_buffer_distance(closest_cal_points, points_in_buffer, bool_lat_lon)
