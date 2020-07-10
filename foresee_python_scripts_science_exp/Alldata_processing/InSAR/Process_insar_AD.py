

import datetime
import itertools
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import Insar_functions as fn


base_dir = "/home/willgoodwin/PostDoc/Foresee/Data/Interferometry/"
out_directory = base_dir+"Failure/"

ascending_file = directory + "FORESEE_D2.7_TimeSeries_A_CSK_CaseStudy2.shp"
descending_file = directory + "FORESEE_D2.7_TimeSeries_D_CSK_CaseStudy2.shp"
ew_file = directory + "FORESEE_D2.7_TimeSeries_EW_CSK_CaseStudy2.shp"
vert_file = directory + "FORESEE_D2.7_TimeSeries_VERT_CSK_CaseStudy2.shp"

topo_file = "Topography/eu_dem_AoI_epsg32633.bil"


# Figure out direction of movement
# Line of Sight (LoS) vectors are given in (Easting, Northing, Vertical)
ascending_LoS = np.array([-0.523362, -0.100137, 0.8462055])
descending_LoS = np.array([0.481241, -0.091219, 0.871829])

# Existing axes in data
ew_axis = np.array([1,0,0])
vert_axis = np.array([0,0,1])
# The third axis
ns_axis = np.array([0,1,0])



# Vector amplitude formula
#vector = asc_data*ascending_LoS + ew_data*ew_axis + v_data*vert_axis
#vector_magnitude = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)


#############################################################################
#############################################################################
#############################################################################

# Open the required file.
for file in [ascending_file, descending_file]:

	print("opening", file)

	F = gpd.read_file(file)

	if file == ascending_file: name = "A"
	if file == descending_file: name = "D"

	# Define dates
	cols = F.columns.values
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
	F_startdate = datetime.datetime.strptime(datecols[0], 'D%Y%m%d')
	Farr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)
	preFarr = np.zeros((topo_array.shape[0],topo_array.shape[1], N_bands ), dtype = np.float)



	runstart = datetime.datetime.now()

	for th in range(len(threshold)):
		print("failure threshold velocity:", threshold[th], 'mm/yr')


		# 2. loop through the points in EW or V
		for i in range(len(F)):
			f = F.iloc[i]


			# 2.1 assign the point to a pixel
			p = f['geometry']
			x_id = int( (p.x - originX)/pixelWidth )
			y_id = int( (p.y - originY)/pixelHeight )

			# 2.2 calculate the 2D displacement velocity timeseries in the EW-V plane for that point
			disp_f = np.array(f[datecols])


			inst_vel_f = (disp_f[1:] - disp_f[:-1]) / intervals_yr


			sq_magnitude_2D = inst_vel_f**2

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
				failtimes = faildates - F_startdate

				prefaildates = velocity_dates[prefailures]
				prefailtimes = prefaildates - F_startdate

				for k in range(len(failtimes)): failtimes[k] = failtimes[k].total_seconds()
				for k in range(len(prefailtimes)): prefailtimes[k] = prefailtimes[k].total_seconds()

				# fill the array "bands"
				# cells are filled if:
				# - they are not filled (0)
				# - there is an existing failure but it is later than the one we just found
				# - there is a non-failing point (-1)
				for k in range(len(failtimes[:N_bands])):
					if Farr[y_id, x_id, k] <= 0: 
						Farr[y_id, x_id, k] = failtimes[k]
						preFarr[y_id, x_id, k] = prefailtimes[k]
					else:
						Farr[y_id, x_id, k] = min(failtimes[k], Farr[y_id, x_id, k])
						preFarr[y_id, x_id, k] = min(prefailtimes[k], preFarr[y_id, x_id, k])

			# 2.4 if there are no failures, mark the date as -1
			else:
				Farr[y_id, x_id, :] = -1
				preFarr[y_id, x_id, :] = -1


		# 3. save the times of failure (depends on chosen number of bands)
		for i in range(N_bands):
			print ('saving band', i+1)
			fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+name+"_failtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, Farr[:,:,i])
			fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), out_directory+name+"_prefailtime_"+str(i+1)+"_threshold"+str(threshold[th])+"mmyr.bil", pixelWidth, preFarr[:,:,i])




quit()











