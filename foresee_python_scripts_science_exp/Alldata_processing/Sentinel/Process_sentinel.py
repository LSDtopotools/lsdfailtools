
"""
Process_sentinel.py

This is a file to process the sentinel data.
Here, processing means finding which pixels on our DEM have one or more failures and when.

NOTE TO SELF: you should probably start all over again with the high res DEM they gave you ...
#  .... But hopefully with all this clean code it should be easy ...

"""



################################################################################
################################################################################
#Import packages
################################################################################
################################################################################


import datetime
import itertools
import numpy as np
import pandas as bb
import geopandas as gpd
import matplotlib.pyplot as plt

import Sentinel_functions as fn


################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################
base_dir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Sentinel/"
out_directory = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Data_Marina_tests/Sentinel_data_failure/"

# The coordinates in this file are in UTMZone32N: EPSG32632
sentinel_file = base_dir + "FORESEE_D2.3_TimeSeries_Sentinel1_CaseStudy2.shp"

# Careful, this file is in WGS84
topo_file = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_epsg32632.bil"


# Line of Sight (LoS) and axis vectors are given in (Easting, Northing, Vertical)
LoS_vector = np.array([0.632024, 0.117139, -0.766044])

# Other axes in data
ew_axis = np.array([1,0,0])
vert_axis = np.array([0,0,1])
ns_axis = np.array([0,1,0])


################################################################################
################################################################################
# Load the relevant data
################################################################################
################################################################################

# Open the light files. They also have the most (and the same) dates to work on
Sentinel = gpd.read_file(sentinel_file)

# Define dates
cols = Sentinel.columns.values
datecols = np.array([item for item in cols if item.startswith('D20')])
dates = np.array([ datetime.datetime.strptime(item, 'D%Y%m%d') for item in datecols ])
intervals = dates[1:] - dates[:-1]
intervals_yr = [ item.days/365 for item in intervals ]
velocity_dates = dates[1:]
acceleration_dates = velocity_dates[1:]

# Load the topography file
topo_array, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(topo_file)

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
threshold = [150] # m/yr2
N_bands = 3

# 1. create 3 nul arrays (Aarr, Darr, and EWVarr) of the shape of topo_array and N_bands bands deep, containing float objects.
S_startdate = datetime.datetime.strptime(datecols[0], 'D%Y%m%d')
S_enddate = datetime.datetime.strptime(datecols[-1], 'D%Y%m%d')
Sarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)
preSarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)

# time your run
runstart = datetime.datetime.now()



# Load rainfall data from 03/09/2016 until the end of 2018 for the lols
rain = bb.read_csv("/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Precipitation/GPM_data/2014-01-01_to_2019-12-31_Intensity.csv")
rainlist = [datetime.datetime(2014, 1, 1)]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))
rain['time'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']






for th in range(len(threshold)):
	print("failure threshold velocity:", threshold[th], 'mm/yr')
	n_fail = 0

	# 2. loop through the points
	for i in range(len(Sentinel)):
		print ('point:', i, '/', len(Sentinel))
		s = Sentinel.iloc[i]

		# 2.1 assign the point to a pixel
		p = s['geometry']
		x_id = int( (p.x - originX)/pixelWidth )
		y_id = int( (p.y - originY)/pixelHeight )

		# if the point is in the topo raster
		if y_id < Sarr.shape[0] and x_id < Sarr.shape[1] and y_id > 0 and x_id > 0:

			# 2.2 calculate the 2D displacement velocity and acceleration timeseries for that point

			# This is displacement in mm
			disp = np.array(s[datecols])

			# This is a 10-measurement moving average of displacement to denoise the time series
			av10_disp = []
			for j in range(5, len(disp)-5):
				av10_disp.append(np.mean(disp[j-5:j+5]))
			av10_disp = np.asarray(av10_disp)

			intervals_yr_av10 = intervals_yr[5:-5]

			# Calculate "instantaneous" ABSOLUTE velocity in the LoS direction at all dates
			inst_vel_av10 = abs((av10_disp[1:]) / intervals_yr_av10)
			inst_vel = abs((disp[1:]) / intervals_yr)

			# Calculate "instantaneous" acceleration in the LoS direction at all dates
			inst_acc_av10 = (inst_vel_av10[1:] - inst_vel_av10[:-1]) / intervals_yr_av10[1:]
			inst_acc = (inst_vel[1:] - inst_vel[:-1]) / intervals_yr[1:]

			# Only keep positie acceleration as we are not interested in stuff slowing down
			inst_acc_av10[inst_acc_av10 < 0] = 0
			inst_acc[inst_acc < 0] = 0




			# Find out at which date indices the acceleration exceeds failure threshold: 150m/yr2
			failures = np.where(inst_acc_av10 > threshold[th]*1000.)[0]

			# this is the associated dates
			acc_av10_dates = dates[7:-5]

			# if there is one or more failure(s)
			if len(failures) > 0:
				n_fail+=1
				print ('failure number:', n_fail)

				# Find indices of failure initiation and the indices just before failure
				# NB: consecutive indices count as one failure (it's the initiation of failure)
				consec_fail =  failures[:-1] - failures [1:]
				to_keep = [0] + list((np.where(consec_fail != -1)[0] + 1))
				failures = failures[to_keep]
				prefailures = failures-1

				# NB: assign time to failures
				faildates = acc_av10_dates[failures]
				failtimes = faildates - S_startdate

				prefaildates = acc_av10_dates[prefailures]
				prefailtimes = prefaildates - S_startdate


				for k in range(len(failtimes)): failtimes[k] = failtimes[k].total_seconds()
				for k in range(len(prefailtimes)): prefailtimes[k] = prefailtimes[k].total_seconds()

				# fill the array "bands"
				# cells are filled if:
				# - they are not filled (0)
				# - there is an existing failure but it is later than the one we just found
				# - there is a non-failing point (-1)
				Tfail = []
				Tprefail = []
				for k in range(len(failtimes[:N_bands])):
					if Sarr[y_id, x_id, k] <= 0:
						Sarr[y_id, x_id, k] = failtimes[k]
						preSarr[y_id, x_id, k] = prefailtimes[k]
					else:
						Sarr[y_id, x_id, k] = min(failtimes[k], Sarr[y_id, x_id, k])
						preSarr[y_id, x_id, k] = min(prefailtimes[k], preSarr[y_id, x_id, k])

					Tfail.append(Sarr[y_id, x_id, k])
					Tprefail.append(preSarr[y_id, x_id, k])

				# Plot the failures
				fn.plot_dva_rainfall(i, rain, dates, disp, av10_disp, inst_vel, inst_vel_av10, inst_acc, inst_acc_av10, S_startdate, S_enddate, Tfail, Tprefail, threshold[th])

			# 2.4 if there are no failures, mark the date as -1
			else:
				Sarr[y_id, x_id, :] = -1
				preSarr[y_id, x_id, :] = -1




	# 3. save the times of failure (depends on chosen number of bands)
	for i in range(N_bands):
		print ('saving band', i+1)
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"Sentinel_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"myr2.bil", pixelWidth, Sarr[:,:,i])
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"Sentinel_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"myr2.bil", pixelWidth, preSarr[:,:,i])


	quit()





"""
		BEFORE YOU GO ANY FURTHER, ASK ERLI IF THIS IS DISPLACEMENT (POS. REL TO PREVIOUS POSITION) OR CUMULATIVE DISPLACEMENT (POS. REL. TO ORIGIN POSITION). FOR THE MOMENT, FAVOUR THE FORMER

		ALSO, MAYBE RUN THE CALIBRATION ON SHORTER RAINFALL TIMESERIES. 1., IT WILL ACCELERATE THE PROCESS, AND 2., IT WILL AVOID IMPOSSIBLE CONVERGENCE ON SOME OF THE WEIRDER POINTS ..... THINK ABOUT THAT MORE

		Ballpark number: there are about 8k points in the region of interest






		# 2.3 every time velocity > threshold, fill a band with the date of failure observation.




		# Find out at which date indices the velocity exceeds failure threshold
		failures = np.where(sq_magnitude_2D > threshold[th]**2)[0]


		#NOTE: Maybe this is not the best way to identify failures ....
		# Why not try something with "instantaneous" acceleration?
		# I guess it depends on you definition of failure ...
		# It's always the same problem: the definition varies depending on your interests.
		# Let's try out something on the plots

		# Figure out a way to pick out a rain-induced failure!
		# and also make sure your displacement is the right one ...




		# This is the latest failure identification algorithm.
		# if there is one or more failure(s)
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
	#for i in range(N_bands):
		#print ('saving band', i+1)
		#fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"EWV_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, EWVarr[:,:,i])
		#fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"EWV_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, preEWVarr[:,:,i])




	#quit()


"""

# The approach is:
# 0. Define a failure threshold for velocity. e.g.: 80 mm/yr

# 1. create 3 nul arrays (Aarr, Darr, and EWVarr) of the shape of topo_array and 10 bands deep, containing float objects.
# Note: EW and V have identical point geometries and flyover dates.

# 2. loop through the points in EW or V
#	2.1 assign the point to a pixel
#	2.2 calculate the 2D displacement velocity timeseries in the EW-V plane for that point
#	2.3 every time velocity > threshold, fill a band with the date of failure observation.
#	NB: the date is the time in seconds after simulation start (i.e. after the first measurement with zero displacement).
#	NB: actual failure will have occurred before that
#	NB if there are more than 10 failures, the threshold might be wrong
#	2.4 if there are no failures, mark the date as -1

# 3. save the time of first, second and third failure (that should be enough)


# 4. repeat for the ascending and descending data
