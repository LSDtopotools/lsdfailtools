#####
# Script to check whether the list of given coordinates are in the area of
# interest.

# Author: Marina Ruiz Sanchez-Oro
# Date 4/11/2021

#####

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import pandas as pd
import json
import os
import rasterio
import sys
from run_json import *

# input the data file
## must have a list of lat lon points
def is_it_in_aoi(lat_lon_file, output_file, AoI_file):
    """
    is_it_in_aoi checks that the given lat, lon test points are in the area of
    interest.

    :lat_lon_file: lat, lon coords of the points to test. Must be in EPSG:4326.
    :output_file: name of .csv output file
    :AoI_file: shapefile of the area of interest
    :return: file with geometry point objects of the points in the area of interest
    """
    # shapefile with the area of interest
    AoI_file = gpd.read_file(AoI_file, index=False)
    AoI_file = gpd.GeoDataFrame(AoI_file)
    AoI_file = AoI_file.drop(columns=['id'])

    # coordinate system of the AoI file
    AoI_crs = AoI_file.crs
    print(AoI_crs)

    # csv file with the points of interest - to check if they are in area of interest.
    # THE COORDINATES MUST BE IN THE SAME CRS AS THE AREA OF INTEREST
    csv_file_path = lat_lon_file
    csv_df = pd.read_csv(csv_file_path)

    # convert test points to shapely point geometry
    geometry = [Point(xy) for xy in zip(csv_df.lon, csv_df.lat)]

    # assume the projection is the same as the one in the AoI shapefile.
    geo_df = gpd.GeoDataFrame(csv_df, geometry=geometry)
    geo_df = geo_df.set_crs(AoI_crs)

    geo_df = geo_df.drop(columns=['lat', 'lon'])

    for i in range(len(geo_df)):
        test_point = geo_df['geometry'].iloc[i]
        is_in_area = AoI_file.contains(test_point)[0]
        geo_df.at[i,'is_in_area'] = is_in_area

    # keep only the columns with the points in the AoI
    geo_df = geo_df[geo_df.is_in_area]
    geo_df['geometry'].to_csv(output_file, index=None)

    return geo_df


################################################################################
################################################################################
################################################################################
################################################################################
