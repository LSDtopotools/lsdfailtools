import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import rasterio
from shapely import wkt
import geopandas as gpd

from run_json import *

from find_closest_calibrated_point import *

############################################
# Script with function that will run functions needed to get the closest calibrated points from the
# list of lat, lon points that are in the area of interest (output from get_lat_lon_check.py)
# Marina Ruiz Sanchez-Oro
# 10/12/2021
############################################
FILE_PATHS = read_paths_file()
dem_file = FILE_PATHS["dem_file"]
calibration_file = FILE_PATHS["calibration_file"]
transform_bil = FILE_PATHS["transform_bil"]
transform_csv = FILE_PATHS["transform_csv"]
bool_lat_lon = FILE_PATHS["bool_lat_lon"]
closest_cal_points = FILE_PATHS["closest_cal_points"]
points_in_buffer = FILE_PATHS["points_in_buffer"]

def run_find_closest_calibrated_point():
    """
    run_find_closest_calibrated_point finds the calibrated points closest to the test points.
    """
    # find_closest_calibrated_point
    convert_calib_to_lat_lon(dem_file,\
    calibration_file,\
    transform_bil)

    convert_calib_raster_to_csv_shp(transform_bil)

    multipoint, selected_rows = create_calib_multipoint(transform_csv)

    calib_params_closest_point(multipoint,selected_rows,bool_lat_lon, calibration_file,\
    closest_cal_points)

    get_points_in_buffer_distance(closest_cal_points, points_in_buffer, bool_lat_lon)
