#####
# This validation file needs to be run with the data. Otherwise, I can't do anything.
######


import time

print ('Run_validation.py')
print ('This is the file that performs the validation of the iverson model.')
print('Author: GchGoodwin')
print('Last update: 26/06/2020')
print ('pausing for')
print ('3 ... ')
time.sleep(1.0)
print ('2 ... ')
time.sleep(1.0)
print ('1 ... ')
time.sleep(1.0)
print ('GO!')

######################################################
######################################################
# Importing modules
######################################################
######################################################
import sys

sys.path.insert(0,'../../../lsdfailtools-master/lsdfailtools')
# Importing the model
import lsdfailtools.iverson2000 as iverson

# I'll need that to process the outputs
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import shapefile
import itertools


# set the directory to import the functions from
import sys
sys.path.insert(0,'../../lsdfailtools-master/lsdfailtools')

# Importing the model
import lsdfailtools.iverson2000 as iverson

# I'll need that to process the outputs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import product
from datetime import datetime
import pandas as pd
import numpy as np
import shapefile
import itertools

import Validation_functions as fn

import sys
sys.path.insert(0,'../Visualisation')

import Figure_functions as ff



######################################################
######################################################
# set up stuff
######################################################
######################################################

Nodata_value = -9999.

# Model directory
# MR: not sure what this path points to?
#rundir = "/home/willgoodwin/PostDoc/Foresee/Validation/TestMC/"

# Number of MC runs
Nruns = 25

# Number of while loopies
itermax = 50

# Setting the depth resolution vector
depths = np.arange(0.2,3.1,0.1)

# failure threshold
# MR: not sure where this number comes from
threshold = 80 # mm/yr

# Number of points to be calibrated
Num_cal = 200


# failure data files
# MR: cannot find the Interferometry data... - this is InSAR
faildir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Data_Marina_tests/InSAR_data_failure/"
failfile = faildir + "All_1st_failtime__threshold"+str(threshold)+"mmyr.bil"
prefailfile = faildir + "All_1st_prefailtime__threshold"+str(threshold)+"mmyr.bil"

# topography files
topodir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/"
demfile = topodir + "eu_dem_AoI_epsg32633.bil"
slopefile = topodir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road files
roaddir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Road/"
roadfile = roaddir + "Road_line.shp"

# calibrated points files
calibdir = "/exports/csce/datastore/geos/users/s1440040/FORESEE/FORESEE_dev/foresee_python_scripts_science_exp/Calibration/TestMC/"
calibfile = calibdir + "Calibrated.csv"


######################################################
######################################################
# See which points were calibrated
######################################################
######################################################

# 0. Load rasters into arrays for DEM, slope, failtimes and prefailtimes for a given failure threshold. Let's use 80mm/yr for now.
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)
prefailarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(prefailfile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)

# now convert it to pixel coordinates
roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
roadline = roadline.astype('int')
l = mlines.Line2D(roadline[:,0], roadline[:,1])

# read calibrated points
calibrated = pd.read_csv(calibfile)


#######################
# Map calibrated points
#ff.map_calibrated (demarr, calibrated, l, 12, 12, rundir + 'Map_calibrated_pixels.png')


######################
# Map the distribution in terms of failtimes
#ff.plot_failtime (calibrated, 12, 12, rundir + 'Failtime_distribution.png')

######################
# Map the distribution of parameters
#ff.plot_parameters (calibrated, 7, 18, rundir + 'Failure_params.png')


######################
# Map the validation
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#depths = np.arange(0.2,3.1,0.1)
#ff.map_validation(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, 15, 15, rundir + 'Map_validation.png')

######################
# Look at some rain data
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#ff.plot_rain(rain, 15, 15, rundir + 'Rain.png')

######################
# Look at some rain data and failures
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#ff.plot_rain_failures(rain, calibrated, 15, 15, rundir + 'Rain_failures.png')


######################
# Look at some rain data and failures
#rain = pd.read_csv(rundir+"Rainfall_Intensity.csv")
#depths = np.arange(0.2,3.1,0.1)
#ff.plot_rain_failures_valid(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, 15, 15, rundir + 'Rain_failures_validation.png')

######################

# Try a PCA on calibratd points
rain = pd.read_csv("/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Precipitation/GPM_data/2014-01-01_to_2019-12-31_Intensity.csv")
depths = np.arange(0.2,3.1,0.1)
ff.plot_sensitivity(rain, calibrated, 10, 10,  'pca_test.png')


quit()

######################################################
######################################################
# Prepare the selection of pixels to calibrate
######################################################
######################################################



# set up the conditions for calibration to happen
distarr = np.zeros((demarr.shape), dtype = np.float)
selectarr = np.zeros((demarr.shape), dtype = np.float)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)

# now convert it to pixel coordinates
roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
roadline = roadline.astype('int')

# then calculate a matrix of distances to it!
print ('calculating distances')
for i,j in itertools.product(range(demarr.shape[0]), range(demarr.shape[1])):
	distarr[i,j] = min(j-roadline[:,0]) **2 + min(i-roadline[:,1])**2
distarr[distarr <= 0.] = 1




# now figure out a way to select based on distance.
selectarr = 0* distarr
selectarr = (100 - distarr**(1/2.5) ) / 5000
selectarr[selectarr <=0.] = 0.0

final_selectarr = 0 * selectarr

npoints = 0
iterations = 0
while npoints < Num_cal:
	print ('iteration:', iterations)
	for i,j in product (range(demarr.shape[0]), range(demarr.shape[1])):
		die_roll = np.random.rand()

		if selectarr[i,j] > die_roll and final_selectarr[i,j] != 1  and failarr[i,j]-prefailarr[i,j] > 2*24*3600 and failarr[i,j]-prefailarr[i,j] < 100*24*3600 and npoints < Num_cal:
			final_selectarr[i,j] = 1
			npoints += 1

	iterations +=1

print (np.sum(final_selectarr))






######################################################
######################################################
# Start the calibration
######################################################
######################################################

# 1. Load rainfall data from 03/09/2016 until the end of 2018 (say). - the file is ready
rain = pd.read_csv("/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Precipitation/GPM_data/2014-01-01_to_2019-12-31_Intensity.csv")

# to make the file:
# python PPT_CMD_RUN.py --ProdTP GPM_D --StartDate 2016-09-03 --EndDate 2018-12-31 --ProcessDir /home/willgoodwin/PostDoc/Foresee/Data/Precipitation/GPM_data/ --SptSlc /home/willgoodwin/PostDoc/Foresee/Data/Topography/eu_dem_v11_E40N20_AoI.bil --OP --DirOut /home/willgoodwin/PostDoc/Foresee/Calibration/TestMC/

# initialisation
work_df_exists = 0
storage_df_exists = 0
npoints = 0

start = datetime.now()

# Run through the pixels
for i,j in product (range(demarr.shape[0]), range(demarr.shape[1])):

	#  Define the
	Z = demarr[i,j]
	S = slopearr[i,j]
	F = failarr[i,j]
	P = prefailarr[i,j]

	# If the selection condition is met
	if final_selectarr[i,j] == 1:
		die_roll = np.random.rand()
		print ('Calibrating pixel of coordinates:', i, j)

		# This is the interval of time in which the observed failure occurred
		failinterval = F-P
		print ('Observed failure interval:', failinterval/(24*3600), 'days')

		if work_df_exists == 0:
			# run the initial MC simulation
			results = fn.MC_initial(rain, S, depths, Nruns, rundir)
		else:
			# run the initial MC simulation
			results = fn.MC_assisted(rain, S, depths, Nruns, rundir, work_df, Z, i, j)

		# Select the results with the best fitness
		selected = fn.assess_fitness(results, F, P, Nruns)

		# store the most succesful runs in a dataframe
		work_df = fn.storage_df(results, selected, S, Z, i, j, F, P)
		work_df_exists = 1

		inbounds_ID = []

		n = 0
		while len(inbounds_ID) < 2 and n < itermax:
			print ('iteration:', n)

			# run the MC
			results = fn.MC_loop (rain, S, depths, Nruns, rundir, work_df)

			# Select the results with the best fitness
			selected = fn.assess_fitness(results, F, P, Nruns)

			# store the most succesful runs (inbounds) in a dataframe
			temp_df = fn.storage_df(results, selected, S, Z, i, j, F, P)

			# Append that dataframe to the existing one
			work_df = work_df.append(temp_df, ignore_index = True)
			A = np.sign(work_df['insar_failtime'] - work_df['time_of_failure'])
			B = np.sign(work_df['time_of_failure'] - work_df['insar_prefailtime'])
			inbounds_ID = np.where(np.logical_and(A > 0, B > 0))[0]

			if len(inbounds_ID) >1:
				print ('Attempt number', n, ': we have a successful calibration!')
				print ('concerned pixels :', i, j)
				work_df = work_df.iloc[inbounds_ID]

				if storage_df_exists == 0:
					print ('initiating storage of calibrated locations')
					storage_df = work_df.copy(deep = True)
					storage_df_exists = 1

				else:
					print ('adding a location to storage')
					storage_df = storage_df.append(work_df, ignore_index = True)

				print (storage_df[['row', 'col', 'time_of_failure', 'insar_failtime', 'insar_prefailtime']])
				break

			n+=1
			print ()

		npoints +=1

		if npoints >= Num_cal: # Runtime safety

			print(storage_df['insar_failtime'] - storage_df['time_of_failure'])
			print(storage_df['time_of_failure'] - storage_df['insar_prefailtime'])

			# Now save all these things to a .csv
			storage_df.to_csv(rundir + 'Calibrated.csv')

			print ('calibrated', npoints, 'pixels in ', datetime.now()-start)
			print ()

			quit()

















	# 3. What do you do about GW depth? data shows it anywhere between 0.2m and 10m ...

	# 4. try further optimising by running MC in conjunction with GA stuff.

	# 5. Figure out this failure threshold thingy. But for now just use 80 mm/yr.
