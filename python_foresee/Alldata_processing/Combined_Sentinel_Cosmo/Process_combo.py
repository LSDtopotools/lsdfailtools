
"""
Process_combo.py

This is a file to process Sentinel and Cosmo-SkyMed data.
Here, processing means finding which pixels on our DEM have one or more failures and when.

"""



################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import os
import json
import datetime
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt


import Combo_functions as fn

################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

with open("file_paths_combined_sentinel_cosmo.json") as file_with_paths :
	FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["ground_motion_output"]))



out_dir = FILE_PATHS["ground_motion_output"]
out_dir_csv = FILE_PATHS["ground_motion_csv_output"]


# The coordinates in thes files are in UTMZone33N: EPSG32633.
ew_file = FILE_PATHS["interferometry_EW"]
v_file = FILE_PATHS["interferometry_VERT"]

# For some unknown reason the coordinates in this file are in UTMZone32N: EPSG32632
sentinel_file_badproj = FILE_PATHS["sentinel_file_original"]
sentinel_file = FILE_PATHS["sentinel_file_reprojected"]

# Let's put it back in EPSG32633 to make it easy for ourselves, of it's not done already
if not os.path.isfile(sentinel_file):
	fn.reproject_shp(sentinel_file_badproj, FILE_PATHS["reprojection"])

# The coordinates in thes files are in UTMZone33N: EPSG32633.
slopefile = FILE_PATHS["slopefile"]


# Other axes in data
ew_axis = np.array([1,0,0])
vert_axis = np.array([0,0,1])
ns_axis = np.array([0,1,0])

# import precipitation data
rain_file = FILE_PATHS["rain_file"]

N_bands = 3


################################################################################
################################################################################
# SENTINEL data
################################################################################
################################################################################

# Open the Sentinel files.
print ('Opening Sentinel file')
Sentinel = gpd.read_file(sentinel_file)
Sentinel_coords = fn.get_coordinates(Sentinel)


# Define measurement dates and interval
S_cols = Sentinel.columns.values
S_datecols = np.array([item for item in S_cols if item.startswith('D20')])
S_dates = np.array([ datetime.datetime.strptime(item, 'D%Y%m%d') for item in S_datecols ])
S_intervals = S_dates[1:] - S_dates[:-1]
S_intervals_yr = [ item.days/365 for item in S_intervals ]

################################################################################
################################################################################
# COSMO-SKYMED data
################################################################################
################################################################################


# Open the Cosmo-SkyMed files.
print ('Opening Cosmo-SkyMed (interferometry) file')
EW = gpd.read_file(ew_file)
V = gpd.read_file(v_file)
EWV_coords = fn.get_coordinates(EW)

# Define measurement dates and interval
EWV_cols = EW.columns.values
EWV_datecols = np.array([item for item in EWV_cols if item.startswith('D20')])
EWV_dates = np.array([ datetime.datetime.strptime(item, 'D%Y%m%d') for item in EWV_datecols ])
EWV_intervals = EWV_dates[1:] - EWV_dates[:-1]
EWV_intervals_yr = [ item.days/365 for item in EWV_intervals ]

################################################################################
################################################################################
# DEM and rainfall data
################################################################################
################################################################################


# Load the topographic slope file
slope_array, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)

originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

# Load rainfall data
rain = pd.read_csv(rain_file)
rainlist = [datetime.datetime(FILE_PATHS["rain_start_year"], FILE_PATHS["rain_start_month"], FILE_PATHS["rain_start_day"])]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))
rain['time'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']




################################################################################
################################################################################
# Make ground movement time series
################################################################################
################################################################################

# Define the time period of observations
S_startdate = datetime.datetime.strptime(S_datecols[0], 'D%Y%m%d')
S_enddate = datetime.datetime.strptime(S_datecols[-1], 'D%Y%m%d')

EWV_startdate = datetime.datetime.strptime(EWV_datecols[0], 'D%Y%m%d')
EWV_enddate = datetime.datetime.strptime(EWV_datecols[-1], 'D%Y%m%d')

startdate = min(S_startdate, EWV_startdate)
enddate = min(S_enddate, EWV_enddate)

plot_rainfall_ground_motion = False
make_csv_ground_motion = True

# Run through the pixels
counter = 0

# create a failures array
fail_arr = np.zeros((N_bands, slope_array.shape[0], slope_array.shape[1]), dtype = np.float)

for i,j in itertools.product(range(slope_array.shape[0]), range(slope_array.shape[1])):
	slope = slope_array[i,j]

	if slope >= 0:
		xbox = [originX + j*pixelWidth, originX + (j+1)*pixelWidth]
		ybox = [originY + i*pixelHeight, originY + (i+1)*pixelHeight]

		# find the points that are in the box
		Sentinel_inside = fn.indices_in_box(xbox, ybox, Sentinel_coords)
		EWV_inside = fn.indices_in_box(xbox, ybox, EWV_coords)

		if len(Sentinel_inside) > 0 or len(EWV_inside) > 0:

			print('We have a candidate')
			print(i,j)
			counter += 1

			# Cumulative displacement seen by Sentinel
			s = np.asarray(Sentinel[S_datecols].iloc[Sentinel_inside])
			# Cumulative displacement seen by CSK
			ew = np.asarray(EW[EWV_datecols].iloc[EWV_inside])
			v = np.asarray(V[EWV_datecols].iloc[EWV_inside])
			ewv = (ew**2+v**2)**0.5


			datacounter = 0

			movement = []; movement_dates = []; datasource = []

			all_failures = []

			for (arr, dates, intervals) in [(s, S_dates, S_intervals_yr), (ewv, EWV_dates, EWV_intervals_yr)]:

				if datacounter == 0:
					datasource.append("Sentinel")
				elif datacounter == 1:
					datasource.append("CosmoSkyMed")

				# Denoise displacement by computing moving 10-point average
				arr_av10, dates_av10, intervals_av10 = fn.make_av10(arr, dates, intervals)

				# This is where we determine failure time indices
				failure_indices = fn.find_failure_indices(arr_av10, dates_av10, startdate, enddate, datacounter)

				all_failures.append(failure_indices)
				movement.append(arr_av10)
				movement_dates.append(dates_av10)

				datacounter += 1

				# order the failure indices in the  pixel
				if len(failure_indices) > 0:
					ordered_failures = sorted(failure_indices)
					mini = min(len(ordered_failures), len(fail_arr))
					for k in range(0,mini,1):
						index = ordered_failures[k]
						# store the failure time in seconds since start time
						TF = (dates_av10[index] - startdate).total_seconds()
						if fail_arr[k,i,j] <= 0:
							fail_arr[k,i,j] = TF
						else:
							fail_arr[k,i,j] = min(TF, fail_arr[k,i,j])


			if plot_rainfall_ground_motion == True:
				# this plots don't include curvature or aspect
				fn.plot_disp_failure(movement, movement_dates, all_failures, rain, slope, startdate, enddate,i,j, out_dir, datasource)

			elif make_csv_ground_motion == True:
				fn.save_disp_failure_csv(movement, movement_dates, slope, i,j, out_dir_csv, datasource)




#  save the times of failure (depends on chosen number of bands)
for i in range(N_bands):
	print ('saving band', i+1)
	print (np.amax(fail_arr[i,:,:]), np.amin(fail_arr[i,:,:]))
	fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_dir+"Failtime_"+str(i+1)+"_since_"+startdate.strftime('%Y%m%d')+".bil", pixelWidth, fail_arr[i,:,:])
