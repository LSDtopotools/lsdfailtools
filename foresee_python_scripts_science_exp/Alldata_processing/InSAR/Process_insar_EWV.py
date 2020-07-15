
"""
Process_insar_EWV.py

This is a file to process the East-West and Vertical InSAR data, which are in the same format and seem to have the same dates.
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

import matplotlib
matplotlib.use("Agg")
import Insar_functions as fn


################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

base_dir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Interferometry/"
out_directory = base_dir+"Failure/"

ew_file = base_dir + "FORESEE_D2.7_TimeSeries_EW_CSK_CaseStudy2.shp"
vert_file = base_dir + "FORESEE_D2.7_TimeSeries_VERT_CSK_CaseStudy2.shp"

# Careful, this file is in the Italian EPSG
topo_file = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_AoI_epsg32633.bil"


# Figure out direction of movement
# Line of Sight (LoS) and axis vectors are given in (Easting, Northing, Vertical)

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
threshold = [40, 60, 80, 100, 150, 200, 500, 1000] # mm/yr
N_bands = 3

# 1. create 3 nul arrays (Aarr, Darr, and EWVarr) of the shape of topo_array and N_bands bands deep, containing float objects.
EWV_startdate = datetime.datetime.strptime(datecols[0], 'D%Y%m%d')
EWVarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)
preEWVarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)

# time your run
runstart = datetime.datetime.now()


# not sure where the TestMC data is?

# Start a plot for the lols
fig=plt.figure(1, facecolor='White',figsize=[7, 7])
ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
ax2 =  ax1.twinx()
'''
# Load rainfall data from 03/09/2016 until the end of 2018 for the lols
rain = bb.read_csv("/home/willgoodwin/PostDoc/Foresee/Calib_Valid/TestMC/2016-09-03_to_2018-12-31_Intensity.csv")
rainlist = [datetime.datetime(2016, 9, 3)]
for i in range(1,len(rain)):
	rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))
rain['time'] = rainlist
rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

#plot the rain
ax1.plot(rain['time'], rain['rainfall_mm'], '-b', lw = 0.5)

'''



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


		ax2.plot(velocity_dates, sq_magnitude_2D, c = plt.cm.jet(i*100), lw = 1)

		if i > 1:
			break


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
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"EWV_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, EWVarr[:,:,i])
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+"EWV_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, preEWVarr[:,:,i])

	"""

	ax1.set_xlim(left = datetime.datetime(2016, 9, 3), right = datetime.datetime(2019, 6, 3))
	#plt.show()
	plt.savefig("insar_ewv.png")
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
