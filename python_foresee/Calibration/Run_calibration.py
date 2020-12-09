
######################################################
######################################################
# Importing modules
######################################################
######################################################

# Importing the model
import sys
sys.path.insert(0,'../../lsdfailtools')# Importing the model
import iverson2000 as iverson

# I'll need that to process the outputs
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

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


# Model directory
rundir = FILE_PATHS["rain_intensity_caliv_valid"]

# parameter files
# the params that fix the iterations and the number of calibrated points in our "sortaMarkovChainMonteCarlo"
Cal_params_file = rundir + "Calibration_parameters.csv"
# the params used to define the physical soil properties in the iverson MC runs
Iverson_MC_params_file = rundir + "Iverson_MC_parameters.csv"

# failure data files
faildir = FILE_PATHS["ground_motion_failure"]

failfile = faildir + "Failtime_1_since_20141020.bil"

# topography files
topo_dir = FILE_PATHS["topo_dir"]

demfile = topo_dir + "eu_dem_AoI_epsg32633.bil"
slopefile = topo_dir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road file
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp" # this is in EPSG:32633

# piezometry files
piezo_path = FILE_PATHS["piezo_dir"]
piezo_data_file = piezo_path + "data_piezometer.csv"

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
rainfile = rundir + StartDate + "_to_" + EndDate + "_Intensity.csv"
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
fn.calibrate_points_MC(final_selectarr, demarr, slopearr, failarr, rain, GW_d_ini, Cal_params, Iverson_MC_params, rundir)




"""fig_height = 7; fig_width = 7
fig=plt.figure(6, facecolor='White',figsize=[fig_width, fig_height])

ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)


dem_mask = np.ma.masked_where(demarr <= -10, demarr)
Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

ax1.add_line(l)

Map2 = ax1.imshow(distarr, interpolation='None', cmap=plt.cm.jet, vmin = np.amin(distarr), vmax =  np.amax(distarr), alpha = 0.3)

select_mask = np.ma.masked_where(final_selectarr <= 0, final_selectarr)
Map2 = ax1.imshow(select_mask, interpolation='None', cmap=plt.cm.jet, vmin = 0, vmax =  1, alpha = 1.)

print ('figure saved')
plt.savefig(rundir+ 'selected_points_for_calibration.jpg')"""
