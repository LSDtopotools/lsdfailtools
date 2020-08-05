
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
import pandas as bb
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt


import Combo_functions as fn

################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

with open("../../../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print "The base output directory is {}".format(FILE_PATHS["ground_motion_failure"])



#base_dir = "/home/willgoodwin/PostDoc/Foresee/Data/"
out_dir = FILE_PATHS["ground_motion_failure"]

sentinel_dir = FILE_PATHS["sentinel_dir"]
# INTERFEROMETRY DATA from CosmoSkyMed
cosmo_dir = FILE_PATHS["interferometry_dir"]


# The coordinates in thes files are in UTMZone33N: EPSG32633.
ew_file = cosmo_dir + "FORESEE_D2.7_TimeSeries_EW_CSK_CaseStudy2.shp"
v_file = cosmo_dir + "FORESEE_D2.7_TimeSeries_VERT_CSK_CaseStudy2.shp"

# For some unknown reason the coordinates in this file are in UTMZone32N: EPSG32632
sentinel_file_badproj = sentinel_dir + "FORESEE_D2.3_TimeSeries_Sentinel1_CaseStudy2.shp"
sentinel_file = sentinel_dir + "FORESEE_D2.3_TimeSeries_Sentinel1_CaseStudy2_epsg32633.shp"

# Let's put it back in EPSG32633 to make it easy for ourselves, of it's not done already
if not os.path.isfile(sentinel_file):
	fn.reproject_shp(sentinel_file_badproj, 32633)

# The coordinates in thes files are in UTMZone33N: EPSG32633.
topo_dir = FILE_PATHS["topo_dir"]
slope_file = topo_dir+"eu_dem_AoI_epsg32633_SLOPE.bil"

# Line of Sight (LoS) and axis vectors are given in (Easting, Northing, Vertical)
Sentinel_LoS = np.array([0.632024, 0.117139, -0.766044])
# Other axes in data
ew_axis = np.array([1,0,0])
vert_axis = np.array([0,0,1])
ns_axis = np.array([0,1,0])

#Here comes the rain again
rain_dir = FILE_PATHS["rain_dir"]
rain_file = rain_dir + "2014-01-01_to_2019-12-31_Intensity.csv"

N_bands = 3


################################################################################
################################################################################
# Load the relevant data
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

# Open the Cosmo-SkyMed files. They have the same dates
print ('Opening Cosmo-SkyMed file')
EW = gpd.read_file(ew_file)
V = gpd.read_file(v_file)
EWV_coords = fn.get_coordinates(EW)

# Define measurement dates and interval
EWV_cols = EW.columns.values
EWV_datecols = np.array([item for item in EWV_cols if item.startswith('D20')])
EWV_dates = np.array([ datetime.datetime.strptime(item, 'D%Y%m%d') for item in EWV_datecols ])
EWV_intervals = EWV_dates[1:] - EWV_dates[:-1]
EWV_intervals_yr = [ item.days/365 for item in EWV_intervals ]

# Load the topographic slope file
slope_array, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slope_file)

originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

# Load rainfall data
rain = bb.read_csv(rain_file)
rainlist = [datetime.datetime(2014, 1, 1)]
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

				#else:
				# 	if fail_arr[:,i,j].all() <= 0:
				#		fail_arr[:,i,j] = -1 # -1 means the pixel was looked at but we found no failure




			fn.plot_disp_failure(movement, movement_dates, all_failures, rain, slope, startdate, enddate,i,j, out_dir, datasource)






			#if counter >= 200:
			#	break

			print (fail_arr[:,i,j])



#  save the times of failure (depends on chosen number of bands)
for i in range(N_bands):
	print ('saving band', i+1)
	print (np.amax(fail_arr[i,:,:]), np.amin(fail_arr[i,:,:]))
	#print ( np.where(fail_arr[i,:,:] >  0) )
	fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_dir+"Failtime_"+str(i+1)+"_since_"+startdate.strftime('%Y%m%d')+".bil", pixelWidth, fail_arr[i,:,:])





		#S2N = fn.signal2noiseratio(s_av10, axis = 1, ddof = 0)

		# Plot the displacement for all data sources
		#fn.plot_movement(s_av10, dates_s_av10, ewv_av10, dates_ewv_av10, rain, slope, startdate, enddate,i,j, out_dir, 'Displacement')

		#fn.plot_movement(sv_av10, dates_sv_av10, ewvv_av10, dates_ewvv_av10, rain, slope, startdate, enddate,i,j, out_dir, 'Velocity')

		#fn.plot_movement(sa_av10, dates_s_av10[2:], ewva_av10, dates_ewv_av10[2:], rain, slope, startdate, enddate,i,j, out_dir, 'Acceleration')





		#cumdisp, av10_cumdisp, inst_vel, inst_vel_av10, inst_acc, inst_acc_av10 = fn.calculate_dva(s, S_datecols, S_intervals_yr)

		#fn.plot_dva_rainfall(i, rain, S_dates, cumdisp, av10_cumdisp, inst_vel, inst_vel_av10, inst_acc, inst_acc_av10, startdate, enddate, [0], [0], 150, out_dir)


	#if len(Sentinel_inside) > 0:




	"""# Make velocity
	vel = fn.derivate_per_yr(arr_av10, intervals_av10)
	vel_dates = dates_av10[1:]; vel_intervals = intervals_av10[1:]
	# also denoise velocity
	#vel_av10, vel_dates_av10, vel_intervals_av10 = fn.make_av10(vel, vel_dates, vel_intervals)
	vel_av10, vel_dates_av10, vel_intervals_av10 = vel, vel_dates, vel_intervals

	# Make acceleration
	acc = fn.derivate_per_yr(vel_av10, vel_intervals_av10)
	acc_dates = vel_dates_av10[1:]; acc_intervals = vel_intervals_av10[1:]
	# also denoise acceleration
	#acc_av10, acc_dates_av10, acc_intervals_av10 = fn.make_av10(acc, acc_dates, acc_intervals)
	acc_av10, acc_dates_av10, acc_intervals_av10 = acc, acc_dates, acc_intervals

	movement.append([arr_av10, vel_av10, acc_av10])
	movement_dates.append([dates_av10, vel_dates_av10, acc_dates_av10])

	if datacounter == 0:
		datasource.append("Sentinel")
	elif datacounter == 1:
		datasource.append("CosmoSkyMed")
	datacounter += 1"""




	#fn.plot_dva(movement, movement_dates, rain, slope, startdate, enddate,i,j, out_dir, datasource)





quit()



# Initiate
failarr = np.zeros((slope_array.shape[0],slope_array.shape[1], N_bands ), dtype = np.float)
prefailarr = np.zeros(failarr.shape, dtype = np.float)


# 0. Define a failure threshold among several options
threshold = [150] # m/yr2
N_bands = 3

# 1. create 3 nul arrays (Aarr, Darr, and EWVarr) of the shape of topo_array and N_bands bands deep, containing float objects.
startdate = datetime.datetime.strptime(datecols[0], 'D%Y%m%d')
enddate = datetime.datetime.strptime(datecols[-1], 'D%Y%m%d')

Sarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)
preSarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)

# time your run
runstart = datetime.datetime.now()









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








		"""
		BEFORE YOU GO ANY FURTHER, ASK ERLI IF THIS IS DISPLACEMENT (POS. REL TO PREVIOUS POSITION) OR CUMULATIVE DISPLACEMENT (POS. REL. TO ORIGIN POSITION). FOR THE MOMENT, FAVOUR THE FORMER

		ALSO, MAYBE RUN THE CALIBRATION ON SHORTER RAINFALL TIMESERIES. 1., IT WILL ACCELERATE THE PROCESS, AND 2., IT WILL AVOID IMPOSSIBLE CONVERGENCE ON SOME OF THE WEIRDER POINTS ..... THINK ABOUT THAT MORE

		Ballpark number: there are about 8k points in the region of interest
		"""




		"""
		# 2.3 every time velocity > threshold, fill a band with the date of failure observation.




		# Find out at which date indices the velocity exceeds failure threshold
		failures = np.where(sq_magnitude_2D > threshold[th]**2)[0]


		NOTE: Maybe this is not the best way to identify failures ....
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
	for i in range(N_bands):
		print ('saving band', i+1)
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"EWV_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, EWVarr[:,:,i])
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"EWV_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, preEWVarr[:,:,i])

	"""



	quit()




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
