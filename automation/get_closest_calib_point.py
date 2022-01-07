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

def run_find_closest_calibrated_point(file_paths):
    """
    run_find_closest_calibrated_point finds the calibrated points closest to the test points.
    """

    dem_file = file_paths["dem_file"]
    calibration_file = file_paths["calibration_file"]
    transform_bil = file_paths["transform_bil"]
    transform_csv = file_paths["transform_csv"]
    bool_lat_lon = file_paths["bool_lat_lon"]
    closest_cal_points = file_paths["closest_cal_points"]
    points_in_buffer = file_paths["points_in_buffer"]

    # find_closest_calibrated_point
    convert_calib_to_lat_lon(dem_file,\
    calibration_file,\
    transform_bil)

    convert_calib_raster_to_csv_shp(transform_bil)

    multipoint, selected_rows = create_calib_multipoint(transform_csv)

    calib_params_closest_point(multipoint,selected_rows,bool_lat_lon, calibration_file,\
    closest_cal_points)

    get_points_in_buffer_distance(closest_cal_points, points_in_buffer, bool_lat_lon)
