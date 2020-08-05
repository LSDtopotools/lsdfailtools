
import json
import datetime
import itertools
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import Insar_functions as fn

with open("../../../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)
#piezo_dir = "/home/willgoodwin/PostDoc/Foresee/Data/Terrestrial/"
piezo_dir = FILE_PATHS["piezo_dir"]


interferometry_dir = FILE_PATHS["interferometry_dir"]

topo_dir = FILE_PATHS["topo_dir"]
topo_file = "Topography/eu_dem_AoI_epsg32633.bil"

threshold = [40, 60, 80, 100, 150, 200, 500, 1000] # mm/yr



for i in range(len(threshold)):

	print ('threshold', threshold[i], 'mm/yr')

	Aarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_dir+"A_failtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	preAarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_dir+"A_prefailtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	A_startdate = datetime.datetime(2016, 11, 4)

	Darr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_dir+"A_failtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	preDarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_dir+"D_prefailtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	D_startdate = datetime.datetime(2016, 9, 3)

	EWVarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_dir+"A_failtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	preEWVarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_dir+"EWV_prefailtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	EWV_startdate = datetime.datetime(2016, 11, 4)

	Combo_failarr = np.zeros(Aarr.shape, dtype = np.float)
	Combo_prefailarr = np.zeros(Aarr.shape, dtype = np.float)
	Combo_startdate = datetime.datetime(2016, 9, 3)

	for x, y in itertools.product(range(Aarr.shape[0]), range(Aarr.shape[1])):

		# If there is never a failure
		if Aarr[x,y] == -1 and Darr[x,y] == -1 and EWVarr[x,y] == -1:
			Combo_failarr[x,y] = -1
			Combo_prefailarr[x,y] = -1

		# if all three datasets show failures
		# retain the earliest failure time
		if Aarr[x,y] > 0 and Darr[x,y] > 0 and EWVarr[x,y] > 0:

			# get all the dates
			Afaildate = A_startdate + datetime.timedelta(0,int(Aarr[x,y]))
			Aprefaildate = A_startdate + datetime.timedelta(0,int(preAarr[x,y]))

			Dfaildate = D_startdate + datetime.timedelta(0,int(Darr[x,y]))
			Dprefaildate = D_startdate + datetime.timedelta(0,int(preDarr[x,y]))

			EWVfaildate = EWV_startdate + datetime.timedelta(0,int(EWVarr[x,y]))
			EWVprefaildate = EWV_startdate + datetime.timedelta(0,int(preEWVarr[x,y]))

			#find the earliest faildate
			after = [Afaildate, Dfaildate, EWVfaildate]
			before = [Aprefaildate, Dprefaildate, EWVprefaildate]
			soonafter = min(after)

			# find the latest possible date before failure
			where = np.where(np.array(before)<soonafter)[0]
			starts = np.array([A_startdate, D_startdate, EWV_startdate])

			if len(where) == 0:
				starts = starts[starts < soonafter]
				soonbefore = max(starts)

			else:
				before = np.array(before)[where]
				soonbefore = max(before)

			# compare each date to the earliest startdate
			failtime_since_start = soonafter - Combo_startdate
			prefailtime_since_start = soonbefore - Combo_startdate

			failtime_since_start = failtime_since_start.total_seconds()
			prefailtime_since_start = prefailtime_since_start.total_seconds()

			Combo_failarr[x,y] = failtime_since_start
			Combo_prefailarr[x,y] = prefailtime_since_start


	# save the files
	fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), interferometry_dir+"All_1st_failtime__threshold"+str(threshold[i])+"mmyr.bil", pixelWidth, Combo_failarr)
	fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), interferometry_dir+"All_1st_prefailtime__threshold"+str(threshold[i])+"mmyr.bil", pixelWidth, Combo_prefailarr)





quit()



#############################################################################
#############################################################################
#############################################################################


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



# The approach will be:
# 0. Define a failure threshold for velocity. e.g.: 500 mm/yr

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





#########################################################################################
#########################################################################################
#########################################################################################

# 0. Define a failure threshold for velocity. e.g.: 500 mm/yr
threshold = [40, 60, 80, 100, 150, 200, 500, 1000] # mm/yr
N_bands = 3

# 1. create 3 nul arrays (Aarr, Darr, and EWVarr) of the shape of topo_array and N_bands bands deep, containing float objects.
EWV_startdate = datetime.datetime.strptime(datecols[0], 'D%Y%m%d')
EWVarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)
preEWVarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)



runstart = datetime.datetime.now()

for th in range(len(threshold)):
	print("failure threshold velocity:", threshold[th], 'mm/yr')


	# 2. loop through the points in EW or V
	for i in range(len(EW)):
		ew = EW.iloc[i]
		v = V.iloc[i]

		# 2.1 assign the point to a pixel
		p = ew['geometry']
		x_id = int( (p.x - originX)/pixelWidth )
		y_id = int( (p.y - originY)/pixelHeight )

		# 2.2 calculate the 2D displacement velocity timeseries in the EW-V plane for that point
		disp_ew = np.array(ew[datecols])
		disp_v = np.array(v[datecols])

		inst_vel_ew = (disp_ew[1:] - disp_ew[:-1]) / intervals_yr
		inst_vel_v = (disp_v[1:] - disp_v[:-1]) / intervals_yr

		sq_magnitude_2D = inst_vel_ew**2 + inst_vel_v**2

		# 2.3 every time velocity > threshold, fill a band with the date of failure observation.
		failures = np.where(sq_magnitude_2D > threshold[th]**2)[0]
		if len(failures) > 0:
			# NB: consecutive indices count as one failure
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
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), interferometry_dir+"EWV_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, EWVarr[:,:,i])
		fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), interferometry_dir+"EWV_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, preEWVarr[:,:,i])




quit()
