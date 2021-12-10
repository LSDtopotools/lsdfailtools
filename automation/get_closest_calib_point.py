import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import rasterio
import seaborn
from shapely import wkt
import geopandas as gpd

from find_closest_calibrated_point import *

############################################
# Script with function that will run functions needed to get the closest calibrated points from the
# list of lat, lon points that are in the area of interest (output from get_lat_lon_check.py)
# Marina Ruiz Sanchez-Oro
# 10/12/2021
############################################


def run_find_closest_calibrated_point():
    # find_closest_calibrated_point
    convert_calib_to_lat_lon('/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_AoI_epsg32633.bil',\
    '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv',\
    './test_transform.bil')

    convert_calib_raster_to_csv_shp('./test_transform.bil')

    multipoint, selected_rows = create_calib_multipoint('./test_transform.csv')

    calib_params_closest_point(multipoint,selected_rows,'./bool_lat_lon.csv', '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv',\
    './test_closest_calibration_points_add_coords.csv')

    get_points_in_buffer_distance('./test_closest_calibration_points_add_coords.csv', './test_points_within_buffer_distance.csv', './bool_lat_lon.csv')
