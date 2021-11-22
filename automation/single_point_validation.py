######################################################
######################################################
# Importing modules
######################################################
######################################################
import sys
import time
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

import rasterio
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

# select the point of interest from the raster files.

calibrated_single_point_path = './test_one_point_calibration.csv'
calibrated_single_point_params = pd.read_csv(calibrated_single_point_path, index_col=None)

# this is for testing
## need to double check that the coordinates are correct

# this is the test point in the study area.
lons = [516402]
lats = [4548245]

# choose from the full DEMs the pixel corresponding to the test pixel

with rasterio.open(demfile) as src:
    for demval in src.sample(zip(lons, lats)):
        demval_point = demval

with rasterio.open(slopefile) as src:
    for slopeval in src.sample(zip(lons, lats)):
        slopeval_point = slopeval

with rasterio.open(failfile) as src:
    for failval in src.sample(zip(lons, lats)):
        failval_point = failval


print('Now we have all the points we need for our analysis.')

############################################################
# set the start and end date and take the iverson parameters.

Cal_params = pd.read_csv(Cal_params_file)

StartDate = Cal_params.at[0,'StartDate']
EndDate = Cal_params.at[0,'EndDate']

# Read the Iverson params
Iverson_MC_params = pd.read_csv(Iverson_MC_params_file)
depths  = np.arange(Iverson_MC_params.at[0,'depth'], Iverson_MC_params.at[1,'depth'], 0.2)


###################### RAINFALL DATA #######################

#rainfile = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/2014-01-01_to_2019-12-31_Intensity.csv"
early_late = 'early'
rainfile = f"./{early_late}_precip.csv"

rain = pd.read_csv(rainfile)

rainlist = [0]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
rain['time_s'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']




############################################################
print('I am now starting the validation. Hold on tight.')
fn.get_fos_point_of_interest(rain, depths, calibrated_single_point_params, lats[0], lons[0], demval[0], slopeval[0], failval[0], rundir)

###########################################################
# test some of the graphs and output variables from the validation

FoS = np.load('FoS.npy')
print(np.shape(FoS))
print(FoS)
#depth = np.load('data.npy')
FoS_temp = np.load('FoS_temp.npy')
print(np.shape(FoS_temp))
print(FoS_temp[0,:])
print(FoS_temp[-1,:])


min_depth = np.load('min_depth.npy')
print(np.shape(min_depth))
print(min_depth)
import seaborn

FoS_df = pd.DataFrame(FoS_temp[0,:])
FoS_df.columns = ['FoS']
print(FoS_df)

x_values = np.arange(0,np.shape(FoS_temp)[1])
y_values = np.arange(0,np.shape(FoS_temp)[0])

print(x_values, y_values)
FoS_df['is_it_failure'] = np.where((FoS_df['FoS']>=1),0,1)
print(FoS_df)
seaborn.lineplot(data=FoS_df['FoS'], x=x_values, y=FoS_df['FoS'], hue=FoS_df['is_it_failure'], palette="vlag")
plt.title(f'{early_late} precip')
#plt.show()
plt.savefig(f'{early_late}_precip_fos.png')

#plt.plot(FoS_temp[-1,:], label = 'Deep')
# plt.plot()
# #plt.legend()
# plt.xlabel('Time (days)')
# plt.ylabel('FoS')
# plt.show()
#fn.oS_vs_failure_depth(factor_of_safety, depth, points, fig_height, fig_width, fig_name)
