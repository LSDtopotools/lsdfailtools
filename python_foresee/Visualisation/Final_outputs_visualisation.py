# Importing the model
import lsdfailtools.iverson2000 as iverson

# I'll need that to process the outputs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import product
from datetime import datetime
import pandas as pd
import numpy as np
import shapefile
import itertools
import json

#MR: i´m assuming for now this is InSAR Insar_functions
import sys
sys.path.insert(0,'../Alldata_processing/InSAR')
import Insar_functions as fn
 #import functions as fn

import Figure_functions as ff


######################################################
######################################################
# set up stuff
######################################################
######################################################

Nodata_value = -9999.

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


# Model directory
rundir = FILE_PATHS["rain_intensity_caliv_valid"]

# Number of MC runs
Nruns = 25

# Number of while loopies
itermax = 50

# Setting the depth resolution vector
depths = np.arange(0.2,3.1,0.1)

#failure threshold
threshold = 80 # mm/yr

# Number of points to be calibrated
Num_cal = 200


# failure data files
faildir = FILE_PATHS["interferometry_out_dir"]
failfile = faildir + "All_1st_failtime__threshold"+str(threshold)+"mmyr.bil"
prefailfile = faildir + "All_1st_prefailtime__threshold"+str(threshold)+"mmyr.bil"

# topography files
topodir = FILE_PATHS["topo_dir"]
demfile = topodir + "eu_dem_AoI_epsg32633.bil"
slopefile = topodir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road files
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp"

# calibrated points files
# need to check if this is the right directory
calibdir = FILE_PATHS["rain_intensity_caliv_valid"]
calibfile = calibdir + "Calibrated_all.csv"

rainfile = FILE_PATHS["rain_dir"]
fig_out_dir = FILE_PATHS["figures_dir"]

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
l = mlines.Line2D(roadline[:,0], roadline[:,1])

# read calibrated points
calibrated = pd.read_csv(calibfile)

'''
#######################
# Map calibrated points
ff.map_calibrated (demarr, calibrated, l, 12, 12, fig_out_dir + 'Map_calibrated_pixels.png')

######################
# Map the distribution in terms of failtimes
ff.plot_failtime (calibrated, 12, 12, fig_out_dir + 'Failtime_distribution.png')
######################
# Map the distribution of parameters
ff.plot_parameters (calibrated, 7, 18, fig_out_dir + 'Failure_params.png')
'''
######################
# Map the validation
rain = pd.read_csv(rainfile+"2014-01-01_to_2019-12-31_Intensity.csv")

depths = np.arange(0.2,3.1,0.1)
ff.map_validation(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, roadfile, 15, 15, fig_out_dir + 'Map_validation_test.png')
'''
######################
# Look at some rain data
rain = pd.read_csv(rainfile+"2014-01-01_to_2019-12-31_Intensity.csv")
ff.plot_rain(rain, 15, 15, fig_out_dir + 'Rain.png')

######################
# Look at some rain data and failures
rain = pd.read_csv(rainfile+"2014-01-01_to_2019-12-31_Intensity.csv")
ff.plot_rain_failures(rain, calibrated, 15, 15, fig_out_dir + 'Rain_failures.png')

######################
# Look at some rain data and failures
rain = pd.read_csv(rainfile+"2014-01-01_to_2019-12-31_Intensity.csv")
depths = np.arange(0.2,3.1,0.1)
ff.plot_rain_failures_valid(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, 15, 15, fig_out_dir + 'Rain_failures_validation.png')
######################

# Try a PCA on calibratd points
rain = pd.read_csv(rainfile+"2014-01-01_to_2019-12-31_Intensity.csv")
depths = np.arange(0.2,3.1,0.1)
ff.plot_rain_parameters_correlation(rain, calibrated, 10, 10, rundir + 'pca_test.png')
'''

#########
# What to do next?

# Isolate the function to find the necessary rainfall to cause failure

# Make a function to map limit rainfall for failure.
# Get an idea of return period for that rainfall.
# associate to risk of failure. !!!! the groundwater level should be a condition.

# make documentation-ish like thing
# make a list of the available data and the processed data.


#########
# Groundwork for Marina's takeover

# Essentially: make useful tools and functions

#########


quit()
