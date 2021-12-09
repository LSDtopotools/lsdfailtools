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

from lat_lon_area_check import *

def run_lat_lon_area_check(input_csv_path, input_csv_name):
    # input_csv_path = sys.argv[1]
    # input_csv_name = sys.argv[2]
    full_input_csv = input_csv_path+input_csv_name

    # lat_lon_area_check
    print('Starting lat_lon_area_check')
    dataframe_aoi = is_it_in_aoi(full_input_csv, './bool_lat_lon.csv', '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/AoI.shp')
