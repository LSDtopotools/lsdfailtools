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
import shapefile
import itertools
import json
import os
import rasterio
from rasterio.features import shapes

import sys

import functions as fn

import Figure_functions as ff


######################################################
######################################################
# set up directories
######################################################
######################################################

with open("file_paths_visualisation.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["figures_dir"]))


# Model directory
rundir = FILE_PATHS["input_data_dir"]

#failure threshold
threshold = FILE_PATHS["failure_threshold"] # mm/yr




# failure data files
faildir = FILE_PATHS["interferometry_out_dir"]
failfile = faildir + "All_1st_failtime__threshold"+str(threshold)+"mmyr.bil"
prefailfile = faildir + "All_1st_prefailtime__threshold"+str(threshold)+"mmyr.bil"

# topography files
demfile = FILE_PATHS["dem_file"]
slopefile = FILE_PATHS["slope_file"]

# road file
roadfile = FILE_PATHS["road_file"]

# calibrated points files
calibfile = FILE_PATHS["calibration_file"]

fig_out_dir = FILE_PATHS["figures_dir"]

validfile = FILE_PATHS["validation_file"]


######################################################
######################################################
# See which points were calibrated
######################################################
######################################################

# 0. Load rasters into arrays for DEM, slope, failtimes for a given failure threshold. Let's use 80mm/yr for now.
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)

# need to convert the csv into a geodataframe
valid_df = pd.read_csv(validfile)
valid_array = 0*demarr
for i in range (len(valid_df)):
    x = int(valid_df.iloc[i,-3])
    y = int(valid_df.iloc[i,-2])
    valid_array[x,y] = valid_df.iloc[i,-6]

valid_array[valid_array == 0] = 'nan'



# convert the csv files with the x,y pixel coordinates into lat, long coordinates in a binary -raster- file

new_geotransform,new_projection,file_out = fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), fig_out_dir+"validation_csv_to_raster.bil", pixelWidth, valid_array)

# convert the raster file into a point shapefile
filename = fig_out_dir+'validation_csv_to_raster'
inDs = gdal.Open('{}.bil'.format(filename))
outDs = gdal.Translate('{}.xyz'.format(filename), inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"], noData = np.nan)
outDs = None
try:
    os.remove('{}.csv'.format(filename))
except OSError:
    pass
os.rename('{}.xyz'.format(filename), '{}.csv'.format(filename))

os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=YES {0}.shp {0}.csv'.format(filename))


# deleta Nan values and change column names in the point shapefile.
shp_file = gpd.read_file(fig_out_dir+"validation_csv_to_raster.shp")
shp_file['Z'] = shp_file['Z'].astype('float64')
shp_file = shp_file[shp_file.Z.notnull()]
shp_file = shp_file.dropna()
#shp_file = shp_file.rename({'Z': 'time_of_failure'}, axis=1)
shp_file.to_file(driver = 'ESRI Shapefile', filename= fig_out_dir+"validation_csv_to_shapefile.shp")
