import numpy as np
#import system
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import product
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import pandas as pd
import numpy as np
#import shapefile
import itertools
import json
import os
import rasterio
from rasterio.features import shapes

import sys
import fiona
import image_functions as fn
from shapely.ops import nearest_points


#####
# File to take in a point and find where the closest calibrated point is.
# Find the parameters of the closest calibrated point.
#####


### IMPORTANT NOTE ### REMEBER THAT SOME POINTS HAVE TWO VALUES, THIS IS WHY
### THE NUMBER OF ROWS IN THE CALIBRATION FILE DOESN'T MATCH THE NUMBER OF
### COORDINATE POINTS

dem_file = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_AoI_epsg32633.bil"

# 0. Load rasters into arrays
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(dem_file)

calibration_file = '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv'

# need to convert the csv into a geodataframe
input_df = pd.read_csv(calibration_file)
input_array = 0*demarr
for i in range (len(input_df)):
    x = int(input_df['row'].iloc[i])
    y = int(input_df['col'].iloc[i])
    input_array[x,y] = input_df.index[i]

input_array[input_array == 0] = 'nan'

new_geotransform,new_projection,file_out = fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), "./_test_transform.bil", pixelWidth, input_array)
print('I am done yayyy')


# The following is useful code. KEEP IT!! - just need to uncomment it
filename='test_transform'
inDs = gdal.Open('./_test_transform.bil'.format(filename))
outDs = gdal.Translate('{}.xyz'.format(filename), inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
outDs = None
try:
    os.remove('{}.csv'.format(filename))
except OSError:
    pass
os.rename('{}.xyz'.format(filename), '{}.csv'.format(filename))
os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=NO {0}.shp {0}.csv'.format(filename))

# This has converted the raster to a csv with the xy coordinates and the Z values at the calibrated points.
## We want to split the data now so that we have a lat, lon, Z column and it's easier to sort the data that is not NaN.

# Load the file
input_df = pd.read_csv('./test_transform.csv', ' ')
selected_rows = input_df[~input_df['Z'].isnull()]
selected_rows_indices = selected_rows.index
print(selected_rows.index)

geometry = [Point(xy) for xy in zip(selected_rows.X, selected_rows.Y)]
geo_df = gpd.GeoDataFrame(selected_rows, geometry=geometry)
print(geo_df.head)

geo_df = geo_df.drop(columns=['X', 'Y', 'Z'])
multipoint = geo_df.geometry.unary_union

# test point - need to change this to be the output from the lat_lon_area_check.py script.
# need to also be able to take a list of points instead of just one.
point = Point(515854, 4.551284e+06)

# this prints out the point closest to our list of points
## need to figure ou
closest_cal_point = nearest_points(multipoint, point)[0]
print(closest_cal_point, nearest_points(multipoint, point)[0])
#print(closest_cal_point.x)

# note that Z is not the altitude but the row number in the initial dataframe... need to figure out a better way to factor this in.
calibration_params_index = list(selected_rows[selected_rows['X']==closest_cal_point.x].index.values)
print(calibration_params_index)
calibration_parameters_selection = int(selected_rows['Z'][calibration_params_index])
print(calibration_parameters_selection)

calibration_file = '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv'

full_calibration_df = pd.read_csv(calibration_file)

print(full_calibration_df.head())
calibration_parameters = full_calibration_df.iloc[[calibration_parameters_selection]]
print(calibration_parameters)

# save the parameters to a new file with only the point we want
#calibration_parameters = calibration_parameters.drop[0]
calibration_parameters = calibration_parameters.drop(calibration_parameters.columns[[0]], axis=1)
calibration_parameters.to_csv('./test_one_point_calibration.csv',index=False)
