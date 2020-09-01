
import json
import datetime
import itertools
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


import sys
sys.path.insert(0,'../InSAR')
import Insar_functions as fn


with open("../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The interferometry path is {}".format(FILE_PATHS["interferometry_dir"]))

interferometry_out_dir = FILE_PATHS["interferometry_out_dir"]

topo_file = FILE_PATHS["topo_dir"] + "eu_dem_AoI_epsg32633.bil"

threshold = [40, 60, 80, 100, 150, 200, 500, 1000] # mm/yr



for i in range(len(threshold)):

	print ('threshold', threshold[i], 'mm/yr')

	Aarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_out_dir+"A_failtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	preAarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_out_dir+"A_prefailtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	A_startdate = datetime.datetime(2016, 11, 4)

	Darr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_out_dir+"A_failtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	preDarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_out_dir+"D_prefailtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	D_startdate = datetime.datetime(2016, 9, 3)

	EWVarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_out_dir+"A_failtime_1_threshold"+str(threshold[i])+"mmyr.bil")
	preEWVarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(interferometry_out_dir+"EWV_prefailtime_1_threshold"+str(threshold[i])+"mmyr.bil")
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
	fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), interferometry_out_dir+"All_1st_failtime__threshold"+str(threshold[i])+"mmyr.bil", pixelWidth, Combo_failarr)
	fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), interferometry_out_dir+"All_1st_prefailtime__threshold"+str(threshold[i])+"mmyr.bil", pixelWidth, Combo_prefailarr)






#############################################################################
#############################################################################
#############################################################################


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
