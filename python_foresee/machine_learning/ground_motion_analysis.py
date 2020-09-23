################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import os
import re
import json
import gdal
import datetime
import itertools
import shapefile
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
import functions_ground_motion as fgm

################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["ground_motion_failure"]))

ground_motion_dir = FILE_PATHS["ground_motion_csv"]
ground_motion_file = ground_motion_dir + ""
# for later: this is how the ground motion data have been saved
#out_dir_csv + 'Timeseries_GroundMotion_pixel'+str(i)+'_'+str(j)+'_failure.csv'


precip_dir = FILE_PATHS["rain_intensity_caliv_valid"]
out_dir = FILE_PATHS["time_series_ml"]

# Here comes the rain again
rain_dir = FILE_PATHS["rain_dir"]
rain_file = rain_dir + "2014-01-01_to_2019-12-31_Intensity.csv"

topo_dir = FILE_PATHS["topo_dir"]
slopefile = topo_dir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road file
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp" # this is in EPSG:32633

# load the data
# Probably load the data on each of the csv files as we process it
# so that the data doesn't need to be stored in memory all the time.

# create the csv file with all the pixel locations
# fgm.make_pxl_csv(ground_motion_dir)



# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)


slopearr, pixelWidth, (geotransform, inDs) = fgm.ENVI_raster_binary_to_2d_array(slopefile)

ground_motion_pxl = pd.read_csv(ground_motion_dir + "pixel_values.csv",sep=',')
ground_motion_pxl = np.array(ground_motion_pxl)

# the first column only has indices - we don't need that.
ground_motion_pxl = ground_motion_pxl[:,1:]


# create a copy of the array to make the mask from
slope_array_to_mask = np.copy(slopearr)

# if the pixels have a ground motion time series, keep them.
for i in (range(ground_motion_pxl.shape[0])):
    slope_array_to_mask[ground_motion_pxl[i,0], ground_motion_pxl[i,1]] = 1.0

# this creates a binary mask
mask = (slope_array_to_mask == 1.0).astype(int)

# mask the slope array with the ground motion pixels
masked_slope_array = mask*slopearr


#np.savetxt(ground_motion_dir +"slopearr.csv", masked_slope_array, delimiter=",")



# calculate distances to main road.
distarr = fgm.calc_dist2road(slopearr.shape, roadline, geotransform)

print(np.shape(distarr))


# DATA(for each pixel):
# 1.Ground motion data (DONE)
# 2.Slope (DONE)
# 3.Elevation (TO DO)
# 4.Distance to road (DONE)
# 5.Drainage Area (TO DO)
# 6.Curvature (TO DO)
#
# METHOD
# 1. Separate the data into training and testing data.
#
#
#
# OUTPUT:
#
# We want to calculate a ground motion timeseries for each of the pixels. In order to do this, we will input the time series
# for each of the pixels, which will involve having a ground motion timeseries for each pixel. We also should calculate the
# distance to the road and will also include the slope and elevation of each pixel.
