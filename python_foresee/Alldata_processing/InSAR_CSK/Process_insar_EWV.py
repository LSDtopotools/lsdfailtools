
"""
Process_insar_EWV.py

This is a file to process the East-West and Vertical InSAR data, which are in the same format and seem to have the same dates.
Here, processing means finding which pixels on our DEM have one or more failures and when.


"""



################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import json
import datetime
import itertools
import numpy as np
import pandas as bb
import geopandas as gpd
import matplotlib.pyplot as plt

import Insar_functions as fn


################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################
with open("files_path_insar.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["out_failure_dir"]))


interferometry_dir = FILE_PATHS["interferometry_dir"]
out_failure_dir = FILE_PATHS["out_failure_dir"]


ew_file = FILE_PATHS["interferometry_EW_CSK"]
vert_file = FILE_PATHS["interferometry_VERT_CSK"]

# Careful, this file is in the Italian EPSG
dem_file = FILE_PATHS["dem_file"]



# Existing axes in data
ew_axis = np.array([1,0,0])
vert_axis = np.array([0,0,1])
# The third, useless axis
ns_axis = np.array([0,1,0])


# Vector amplitude formula
#vector = asc_data*ascending_LoS + ew_data*ew_axis + v_data*vert_axis
#vector_magnitude = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)


################################################################################
################################################################################
# Load the relevant data
################################################################################
################################################################################

# Open the light files. They also have the most (and the same) dates to work on
EW = gpd.read_file(ew_file); V = gpd.read_file(vert_file)

# Define dates
cols = EW.columns.values
datecols = np.array([item for item in cols if item.startswith('D20')])
dates = np.array([ datetime.datetime.strptime(item, 'D%Y%m%d') for item in datecols ])
intervals = dates[1:] - dates[:-1]
intervals_yr = [ item.days/365 for item in intervals ]
velocity_dates = dates[1:]

# Load the topography file
topo_array, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(dem_file)

originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]


################################################################################
################################################################################
# Begin the failure identification
################################################################################
################################################################################
# 0. Define a failure threshold among several options
threshold = [40, 60, 80, 100, 150, 200, 500, 1000] # mm/yr
N_bands = 3

# 1. create 3 nul arrays (Aarr, Darr, and EWVarr) of the shape of topo_array and N_bands bands deep, containing float objects.
EWV_startdate = datetime.datetime.strptime(datecols[0], 'D%Y%m%d')
EWVarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)
preEWVarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)

# time your run
runstart = datetime.datetime.now()



for th in range(len(threshold)):
	print("failure threshold velocity:", threshold[th], 'mm/yr')

	# 2. loop through the points in EW and V (they are the same points)
	for i in range(len(EW)):
		print ('point:', i, '/', len(EW))
		ew = EW.iloc[i]
		v = V.iloc[i]

		# 2.1 assign the point to a pixel
		p = ew['geometry']
		x_id = int( (p.x - originX)/pixelWidth )
		y_id = int( (p.y - originY)/pixelHeight )

		# 2.2 calculate the 2D displacement velocity timeseries in the EW-V plane for that point
		disp_ew = np.array(ew[datecols])
		disp_v = np.array(v[datecols])

		#Calculate "instantaneous" velocity in each direction at all dates
		inst_vel_ew = (disp_ew[1:] - disp_ew[:-1]) / intervals_yr
		inst_vel_v = (disp_v[1:] - disp_v[:-1]) / intervals_yr

		# get the 2D velocity magnitude (squared to save time?)
		sq_magnitude_2D = inst_vel_ew**2 + inst_vel_v**2

		if i > 1:
			break

		# 2.3 every time velocity > threshold, fill a band with the date of failure observation.

		# Find out at which date indices the velocity exceeds failure threshold
		failures = np.where(sq_magnitude_2D > threshold[th]**2)[0]

		# This is the latest failure identification algorithm.
		# if there is one more failure(s)
		if len(failures) > 0:

			# Find indices of failure initiation and the indices just before failure
			# NB: consecutive indices count as one failure (it's the initiation of failure)
			consec_fail =  failures[:-1] - failures [1:]
			to_keep = [0] + list((np.where(consec_fail != -1)[0] + 1))
			failures = failures[to_keep]
			prefailures = failures-1

			# NB: assign time to failures
			faildates = velocity_dates[failures]
			failtimes = faildates - EWV_startdate

			prefaildates = velocity_dates[prefailures]
			prefailtimes = prefaildates - EWV_startdate


			for k in range(len(failtimes)): failtimes[k] = failtimes[k].total_seconds()
			for k in range(len(prefailtimes)): prefailtimes[k] = prefailtimes[k].total_seconds()

			# fill the array "bands"
			# cells are filled if:
			# - they are not filled (0)
			# - there is an existing failure but it is later than the one we just found
			# - there is a non-failing point (-1)
			for k in range(len(failtimes[:N_bands])):
				if EWVarr[y_id, x_id, k] <= 0:
					EWVarr[y_id, x_id, k] = failtimes[k]
					preEWVarr[y_id, x_id, k] = prefailtimes[k]
				else:
					EWVarr[y_id, x_id, k] = min(failtimes[k], EWVarr[y_id, x_id, k])
					preEWVarr[y_id, x_id, k] = min(prefailtimes[k], preEWVarr[y_id, x_id, k])

		# 2.4 if there are no failures, mark the date as -1
		else:
			EWVarr[y_id, x_id, :] = -1
			preEWVarr[y_id, x_id, :] = -1


	# 3. save the times of failure (depends on chosen number of bands)
	for i in range(N_bands):
		print ('saving band', i+1)
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_failure_dir+"EWV_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, EWVarr[:,:,i])
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_failure_dir+"EWV_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, preEWVarr[:,:,i])
