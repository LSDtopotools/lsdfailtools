import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import rasterio
from shapely import wkt
import geopandas as gpd

from lat_lon_area_check import *
from run_json import *

############################################
# Script with function that will run functions needed to calculate the points
# that are in the area of interest. The inputs are the path and the file give by the user.
# Marina Ruiz Sanchez-Oro
# 10/12/2021
############################################
FILE_PATHS = read_paths_file()
bool_lat_lon = FILE_PATHS["bool_lat_lon"]
aoi_shapefile = FILE_PATHS["aoi_shapefile"]


def run_lat_lon_area_check(input_csv_path, input_csv_name):
    """
    run_lat_lon_area_check checks if the test points are in area of interest and creates a file with the points in the area of interest.
    :param input_csv_path: path to the location of the csv file with the test points
    :param input_csv_name: name of csv file with the test points
    """
    full_input_csv = input_csv_path+input_csv_name

    # lat_lon_area_check
    print('Starting lat_lon_area_check')
    dataframe_aoi = is_it_in_aoi(full_input_csv, bool_lat_lon, aoi_shapefile)
