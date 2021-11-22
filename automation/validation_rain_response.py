import time

print ('Run_validation.py')
print ('This is the file that performs the validation of the iverson model.')
print('Authors: GchGoodwin, MRuizSanchez-Oro')
print ('pausing for')
print ('3 ... ')
print ('2 ... ')
print ('1 ... ')
print ('GO!')

######################################################
######################################################
# Importing modules
######################################################
######################################################
import sys

sys.path.insert(0,'../../lsdfailtools/lsdfailtools')
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

import val_functions as fn




################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

# Model directory

with open("file_paths_validation.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["output_validation_dir"]))


# Model directory
rundir = './'

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



# calibrated points files
calibfile = '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv'


######################################################
######################################################
# See which points were calibrated
######################################################
######################################################

# 0. Load rasters into arrays for DEM, slope, failtimes and prefailtimes for a given failure threshold. Let's use 80mm/yr for now.
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)

# now convert it to pixel coordinates
roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
roadline = roadline.astype('int')
line = mlines.Line2D(roadline[:,0], roadline[:,1])

#distarr = fn.calc_dist2road(slopearr.shape, roadline, geotransform)

# read calibrated points
calibrated = pd.read_csv(calibfile)


# Read the calibration params
Cal_params = pd.read_csv(Cal_params_file)

StartDate = Cal_params.at[0,'StartDate']
EndDate = Cal_params.at[0,'EndDate']

# Read the Iverson params
Iverson_MC_params = pd.read_csv(Iverson_MC_params_file)
depths  = np.arange(Iverson_MC_params.at[0,'depth'], Iverson_MC_params.at[1,'depth'], 0.2)


#################################
# perform the validation

early_late = 'early'
rainfile = "./early_precip_100_0.csv"

rain = pd.read_csv(rainfile)

rainlist = [0]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
rain['time_s'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']


fn.run_validation(rain, depths, calibrated, demarr, slopearr, failarr,rundir, early_late)

# validation will output a file with the location and the time of failure
# We want to output as well whether we can trust the point, a graph with the
# output and the point in time where it will fail.
