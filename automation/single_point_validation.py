# Script to find whether the test points fail, given the set of calibration
# parameters that we have found (from the closest calibration point)

# Author: Marina Ruiz Sanchez-Oro
# Date: 15/11/2021


######################################################
######################################################
# Importing modules
######################################################
######################################################
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import rasterio
import val_functions as fn
import seaborn



################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################


with open("file_paths_validation.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["output_validation_dir"]))


# Model directory
rundir = './'

# parameter files
# the params used to define the physical soil properties in the iverson MC runs
Iverson_MC_params_file = FILE_PATHS["iverson_param"]

# observed failure data files
failfile = FILE_PATHS["ground_motion_failure"]

# topography files
demfile = FILE_PATHS["dem_file"]
slopefile = FILE_PATHS["slope_file"]


# calibrated points files
#calibfile = '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv'
#calibfile = './test_closest_calibration_points.csv'

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
#'./test_closest_calibration_points.csv' - this is the new file instead of the one with the single point
calibrated_multiple_points_path = './test_closest_calibration_points.csv'
calibrated_multiple_point_params = pd.read_csv(calibrated_multiple_points_path, index_col=None)

lons = calibrated_multiple_point_params['lon_test'].to_list()
lats = calibrated_multiple_point_params['lat_test'].to_list()

# test point: the point below should fail.
# let's add it to the list so that we know if things work properly.

#lons.append(516402)
#lats.append(4548245)

#print(lons,lats)
#lons = [516402]
#lats = [4548245]


# select from the full DEMs the pixel corresponding to the test pixels
def select_topo_data(topo_file, lons_list, lats_list):
    with rasterio.open(topo_file) as src:
        vals = [x for x in src.sample(zip(lons, lats))]
    return vals

# values of the corresponding points
demval_point = select_topo_data(demfile, lons, lats)
slopeval_point = select_topo_data(slopefile, lons, lats)
failval_point = select_topo_data(failfile, lons, lats)


print('Now we have all the points we need for our analysis.')

############################################################

# Read the Iverson params
Iverson_MC_params = pd.read_csv(Iverson_MC_params_file)
depths  = np.arange(Iverson_MC_params.at[0,'depth'], Iverson_MC_params.at[1,'depth'], 0.2)


###################### RAINFALL DATA #######################
# We are assuming that the rainfall data is the same for all the points
# the area of interest hasa rough length of 30km which is the resolution of the
# precipitaiton data we have.
rainfile = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/2014-01-01_to_2019-12-31_Intensity.csv"
#early_late = 'early'
#rainfile = f"./{early_late}_precip.csv"

rain = pd.read_csv(rainfile)

rainlist = [0]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
rain['time_s'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']


############################################################
print('I am now finding the time of failure for your test points. Hold on tight.')
lat_failures = []
lon_failures = []
for i in range(len(lats)):
    print(lats[i], lons[i], demval_point[i][0], slopeval_point[i][0], failval_point[i][0])
    #quit()

    lat, lon = fn.get_fos_point_of_interest(rain, depths, calibrated_multiple_point_params.loc[i], lats[i], lons[i], demval_point[i][0], slopeval_point[i][0], failval_point[i][0], rundir)
    lat_failures.append(int(lat))
    lon_failures.append(int(lon))

print(lat_failures, lon_failures)

np.asarray(lat_failures)
np.asarray(lon_failures)

###########################################################
# test some of the graphs and output variables from the validation
for i in range(len(lat_failures)):
    print(i)
    lat_failure = lat_failures[i]
    lon_failure = lon_failures[i]


    FoS = np.load(f'FoS_{lat_failure}_{lon_failure}.npy')
    FoS_temp = np.load(f'FoS_temp_{lat_failure}_{lon_failure}.npy')
    min_depth = np.load(f'min_depth_{lat_failure}_{lon_failure}.npy')

    FoS_df = pd.DataFrame(FoS_temp[0,:])
    FoS_df.columns = ['FoS']
    print(FoS_df)
    #quit()

    # what is the earliest time where we see the FoS go below zero?
    #FoS_below_zero = FoS['FoS']<1
    #print(FoS_df[165:170])
    day_of_failure = (FoS_df['FoS'] < 1.0).idxmax()
    print(f'The first failure is predicted on day {day_of_failure} after the start of the rainfall timeseries.')

    x_values = np.arange(0,np.shape(FoS_temp)[1])
    y_values = np.arange(0,np.shape(FoS_temp)[0])

    FoS_df['is_it_failure'] = np.where((FoS_df['FoS']>=1),0,1)
    seaborn.scatterplot(data=FoS_df['FoS'], x=x_values, y=FoS_df['FoS'], hue=FoS_df['is_it_failure'], s=1)
    plt.title(f'All precip')
    #plt.show()
    plt.savefig(f'precip_fos_{lat_failure}_{lon_failure}.png')
    plt.clf()
quit()
#plt.plot(FoS_temp[-1,:], label = 'Deep')
# plt.plot()
# #plt.legend()
# plt.xlabel('Time (days)')
# plt.ylabel('FoS')
# plt.show()
#fn.oS_vs_failure_depth(factor_of_safety, depth, points, fig_height, fig_width, fig_name)
