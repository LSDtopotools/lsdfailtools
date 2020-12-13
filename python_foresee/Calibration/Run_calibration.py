
######################################################
######################################################
# Importing modules
######################################################
######################################################

# Importing the model
import sys
sys.path.insert(0,'../../lsdfailtools')
import iverson2000 as iverson

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import shapefile
import itertools

# importing custom functions
import Calibration_functions as fn




################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

with open("file_paths_calibration.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["calibration_dir"]))


# Model directory
calibration_dir = FILE_PATHS["calibration_dir"]

# parameter files
# the params that fix the iterations and the number of calibrated points in our "sortaMarkovChainMonteCarlo"
Cal_params_file = FILE_PATHS["calibration_param"]
# the params used to define the physical soil properties in the iverson MC runs
Iverson_MC_params_file = FILE_PATHS["iverson_param"]

# failure data files
failfile = FILE_PATHS["ground_motion_failure"]

# topography files

demfile = FILE_PATHS["dem_file"]
slopefile = FILE_PATHS["slope_file"]

# road file
roadfile = FILE_PATHS["road_file"] # this is in EPSG:32633

# piezometry files
piezo_data_file = FILE_PATHS["piezo_file"]

######################################################
######################################################
# Read the files to create the necessary variables
######################################################
######################################################
Nodata_value = -9999.

# Read the calibration params
Cal_params = pd.read_csv(Cal_params_file)

# Define the dates to generate the rainfall file
StartDate = Cal_params.at[0,'StartDate']
EndDate = Cal_params.at[0,'EndDate']

# Read the Iverson params
Iverson_MC_params = pd.read_csv(Iverson_MC_params_file)

# Load rasters into arrays for slope, failtimes and prefailtimes for a given failure threshold.
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)


######################################################
######################################################
# Make/read the inputs
######################################################
######################################################

# Read the rainfall data
rainfile = calibration_dir + StartDate + "_to_" + EndDate + "_Intensity.csv"
rain = pd.read_csv(rainfile)

# Read the piezometer data - that's the other model input
Piezo_data = pd.read_csv(piezo_data_file)
GW_d_ini = fn.GW_depth_ini(Piezo_data, StartDate)


######################################################
######################################################
# Select pixels
######################################################
######################################################

# calculate distances to main road.
# We wanted to calibrate the points closest to the road, as this is where the landslides are likely to have a greater impact on the road
distarr = fn.calc_dist2road(slopearr.shape, roadline, geotransform)

# Now select pixels based on the distance array and the number of pixels we want.
final_selectarr = fn.select_pixels(distarr, failarr, Cal_params.at[0,'Num_cal'])


######################################################
######################################################
# Run the calibration
######################################################
######################################################


# Now calibrate the points
fn.calibrate_points_MC(final_selectarr, demarr, slopearr, failarr, rain, GW_d_ini, Cal_params, Iverson_MC_params, calibration_dir)
