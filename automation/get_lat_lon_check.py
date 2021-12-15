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

############################################
# Script with function that will run functions needed to calculate the points
# that are in the area of interest. The inputs are the path and the file give by the user.
# Marina Ruiz Sanchez-Oro
# 10/12/2021
############################################


def run_lat_lon_area_check(input_csv_path, input_csv_name):
    full_input_csv = input_csv_path+input_csv_name

    # lat_lon_area_check
    print('Starting lat_lon_area_check')
    dataframe_aoi = is_it_in_aoi(full_input_csv, './bool_lat_lon.csv', '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/AoI.shp')
