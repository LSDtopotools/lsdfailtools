# Importing the model
import lsdfailtools.iverson2000 as iverson

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import product
from datetime import datetime
import pandas as pd
import numpy as np
import shapefile
import itertools
import json
import os

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

# Setting the depth resolution vector
depths = np.arange(0.1,3.0,0.1)

# failure threshold
threshold = 80 # mm/yr




# failure data files
faildir = FILE_PATHS["interferometry_out_dir"]
failfile = faildir + "All_1st_failtime__threshold"+str(threshold)+"mmyr.bil"
prefailfile = faildir + "All_1st_prefailtime__threshold"+str(threshold)+"mmyr.bil"

# topography files
demfile = FILE_PATHS["dem_file"]
slopefile = FILE_PATHS["slope_file"]

# road files
roadfile = FILE_PATHS["road_file"]

# calibrated points files
calibfile = FILE_PATHS["calibration_file"]

fig_out_dir = FILE_PATHS["figures_dir"]

validfile = FILE_PATHS["validation_file"]

Cal_params_file = FILE_PATHS["calibration_params"]

######################################################
######################################################
# See which points were calibrated
######################################################
######################################################

# 0. Load rasters into arrays for DEM, slope, failtimes and prefailtimes for a given failure threshold. Let's use 80mm/yr for now.
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)
prefailarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(prefailfile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)

# now convert it to pixel coordinates
roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
roadline = roadline.astype('int')
line = mlines.Line2D(roadline[:,0], roadline[:,1], linewidth = 1., color='black')



# read calibrated points
calibrated = pd.read_csv(calibfile)
Cal_params = pd.read_csv(Cal_params_file)
validated = pd.read_csv(validfile)

failinterval = Cal_params.at[0,'failinterval'] * 24 * 3600

# start  and end dates of rain data as defined on the Calibration_parameters file.
StartDate = Cal_params.at[0,'StartDate']
EndDate = Cal_params.at[0,'EndDate']


rainfile = rundir + StartDate + "_to_" + EndDate + "_Intensity.csv"
rain = pd.read_csv(rainfile)

rainlist = [0]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
rain['time_s'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

#######################
# Map calibrated points
ff.map_calibrated(demarr, calibrated, line, 10, 15, fig_out_dir + 'Map_calibrated_pixels_test.png')

######################
# Map the distribution of failtimes (calibrated and validated points) along with precipitation data
ff.plot_failtime_calib_valid(calibrated, validated, rain, 10, 10, fig_out_dir + 'Failtime_distribution_test.png')

######################

# Map the distribution of parameters wrt with slope and elevation

ff.plot_parameters(calibrated, 7, 18, fig_out_dir + 'Failure_params_test.png')

######################
# Map the validation

depths = np.arange(0.1,3.0,0.1)
ff.map_validation_arrays(rain, depths, calibrated, validated, line, demarr, slopearr, failarr, failinterval, 10, 10, fig_out_dir + 'Map_validation_with_scale_test.png')
ff.map_validation_arrays_zoom(rain, depths, calibrated, validated, line, demarr, slopearr, failarr, failinterval, 10, 10, fig_out_dir + 'Map_validation_zoom_with_scale_test.png')
ff.map_validation_colorbar(rain, depths, calibrated, validated, line, demarr, slopearr, failarr, failinterval, 10, 15, fig_out_dir + 'Map_validation_with_colorbar_test.png')
######################
# Plot rain data
rain = pd.read_csv(rundir+"2014-01-01_to_2019-12-31_Intensity.csv")
ff.plot_rain(rain, 15, 15, fig_out_dir + 'Rain_test.png')
#  Put the dates on the json file
######################
# Look at some rain data and failures
rain = pd.read_csv(rundir+"2014-01-01_to_2019-12-31_Intensity.csv")
ff.plot_rain_failures(rain, calibrated, 15, 15, fig_out_dir + 'Rain_failures_test.png')
######################
# need to fix this density plot
ff.density_plot(validated,10,10, fig_out_dir + "observed_vs_modelled_pdf_update_test.png")
ff.time_split_violin_plot(validated, 10,10, fig_out_dir + "observed_vs_modelled_violin_plot_test.png")
