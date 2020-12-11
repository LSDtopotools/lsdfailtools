


import time

print ('Run_validation.py')
print ('This is the file that performs the validation of the iverson model.')
print('Author: GchGoodwin')
print('Last update: 26/06/2020')
print ('pausing for')
print ('3 ... ')
#time.sleep(1.0)
print ('2 ... ')
#time.sleep(1.0)
print ('1 ... ')
#time.sleep(1.0)
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
import os

import Validation_functions as fn
#import sys
#sys.path.insert(0,'../Visualisation')

import Figure_functions as ff




################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

# Model directory

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


# Model directory
rundir = FILE_PATHS["rain_intensity_caliv_valid"]



# parameter files
# the params that fix the iterations and the number of calibrated points in our "sortaMarkovChainMonteCarlo"
Cal_params_file = rundir+"Calibration_parameters.csv"
# the params used to define the physical soil properties in the iverson MC runs
Iverson_MC_params_file = rundir+"Iverson_MC_parameters.csv"

# failure data files
faildir = FILE_PATHS["ground_motion_failure"]
failfile = faildir + "Failtime_1_since_20141020.bil"

# topography files
topo_dir = FILE_PATHS["topo_dir"]
demfile = topo_dir+"eu_dem_AoI_epsg32633.bil"
slopefile = topo_dir+"eu_dem_AoI_epsg32633_SLOPE.bil"
cutfile = topo_dir + "eu_dem_v11_E40N20_AoI.bil" # this one needs to be in WGS84 to interface with NASA

# road file
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp" # this is in EPSG:32633

# Rain files - This is the path to creat the rainfall input

# Rain file
raindir = FILE_PATHS["rain_dir"]

PPT_dir = FILE_PATHS["PPT_dir"]
PPT_cmd_run = PPT_dir + "PPT_CMD_RUN.py"

# piezometry files
piezo_path = FILE_PATHS["piezo_dir"]
piezo_data_file = piezo_path + "data_piezometer.csv"

# calibrated points files
calibdir = FILE_PATHS["rain_intensity_caliv_valid"]
calibfile = calibdir + "Calibrated_single_test.csv"


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
l = mlines.Line2D(roadline[:,0], roadline[:,1])

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

rainfile = rundir + StartDate + "_to_" + EndDate + "_Intensity.csv"
rain = pd.read_csv(rainfile)

rainlist = [0]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
rain['time_s'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

if os.path.isfile(rundir+'Validated_updated.csv') is False:
    fn.run_validation_single_output(rain, depths, calibrated, demarr, slopearr, failarr,rundir)
	# fn.run_validation(rain, depths, calibrated, demarr, slopearr, failarr,rundir)
validated = pd.read_csv(rundir+'Validated_updated.csv')

#######################
#Plot the FoS vs depth
# Load the data
factor_of_safety = np.load(rundir+"fos.npy")
psi = np.load(rundir+'Psi.npy')
time_index = np.load(rundir+'time_index.npy')
points = [[368,607],[14,770]]
# points= [[14,770]]
ff.psi_fos_depth_plot_failure(validated, rain, psi,factor_of_safety,time_index, depths, points, 1, 8, 8, rundir + 'Figures/psi_fos_depth_at failure')

# ff.psi_fos_depth_plot_random_timestep(validated, rain, psi,factor_of_safety,time_index, depths, points, 384, 8, 8, rundir + 'Figures/psi_fos_depth_at')

#######################
# Map calibrated points
#ff.map_calibrated (demarr, calibrated, l, 12, 12, rundir + 'Figures/Map_calibrated_pixels.png')


######################
# Map the distribution in terms of failtimes
#ff.plot_failtime (calibrated, 12, 12, rundir + 'Figures/Failtime_distribution.png')

######################
# Map the distribution of parameters
#ff.plot_parameters (calibrated, 7, 18, rundir + 'Figures/Failure_params.png')


######################
# Map the validation
failinterval = Cal_params.at[0,'failinterval'] * 24 * 3600
ff.map_validation(rain, depths, calibrated, validated, l, demarr, slopearr, failarr, failinterval, 15, 15, rundir + 'Figures/Map_validation.png')
ff.plot_failtime_calib_valid (calibrated, validated, rain, 8, 8, rundir + 'Figures/Failtime_distribution_valid.png')
######################
# Look at some rain data
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#ff.plot_rain(rain, 15, 15, rundir + 'Rain.png')

######################
# Look at some rain data and failures
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#ff.plot_rain_failures(rain, calibrated, 15, 15, rundir + 'Rain_failures.png')


######################
# Look at some rain data and failures
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#depths = np.arange(0.2,3.1,0.1)
#ff.plot_rain_failures_valid(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, 15, 15, rundir + 'Rain_failures_validation.png')

######################
# Try a PCA on calibratd points
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#depths = np.arange(0.2,3.1,0.1)
#ff.plot_sensitivity(rain, calibrated, 10, 10, rundir + 'pca_test.png')


quit()

######################################################
######################################################
# Prepare the selection of pixels to calibrate
######################################################
######################################################

# is this section needed for running the validation?? Is this not needed only
# for calibration and not for validation?
'''
# set up the conditions for calibration to happen
distarr = np.zeros((demarr.shape), dtype = np.float)
selectarr = np.zeros((demarr.shape), dtype = np.float)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)

# now convert it to pixel coordinates
roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
roadline = roadline.astype('int')

# then calculate a matrix of distances to it!
print ('calculating distances')
for i,j in itertools.product(range(demarr.shape[0]), range(demarr.shape[1])):
	distarr[i,j] = min(j-roadline[:,0]) **2 + min(i-roadline[:,1])**2
distarr[distarr <= 0.] = 1




# now figure out a way to select based on distance.
selectarr = 0* distarr
selectarr = (100 - distarr**(1/2.5) ) / 5000
selectarr[selectarr <=0.] = 0.0

final_selectarr = 0 * selectarr

npoints = 0
iterations = 0
while npoints < Num_cal:
	print ('iteration:', iterations)
	for i,j in product (range(demarr.shape[0]), range(demarr.shape[1])):
		die_roll = np.random.rand()

		if selectarr[i,j] > die_roll and final_selectarr[i,j] != 1  and failarr[i,j]-prefailarr[i,j] > 2*24*3600 and failarr[i,j]-prefailarr[i,j] < 100*24*3600 and npoints < Num_cal:
			final_selectarr[i,j] = 1
			npoints += 1

	iterations +=1

print (np.sum(final_selectarr))






'''
