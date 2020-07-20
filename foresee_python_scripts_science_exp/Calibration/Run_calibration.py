


import time
'''
print ('Run_calibration.py')
print ('This is the file that performs the calibration of the iverson model.')
print('Author: GchGoodwin')
print('Last update: 25/06/2020')
print ('pausing for')
print ('3 ... ')
time.sleep(1.0)
print ('2 ... ')
time.sleep(1.0)
print ('1 ... ')
time.sleep(1.0)
print ('GO!')
'''
######################################################
######################################################
# Importing modules
######################################################
######################################################

import sys
sys.path.insert(0,'../../lsdfailtools-master/lsdfailtools')# Importing the model
import iverson2000 as iverson

# I'll need that to process the outputs
from datetime import datetime
import pandas as pd
import numpy as np
import shapefile
import itertools

# importing custom functions
import Calibration_functions as fn



######################################################
######################################################
# set important parameters
######################################################
######################################################
Nodata_value = -9999.

# Number of MonteCarlo runs
Nruns = 25

# Max Number of iterations of the MC process
itermax = 50

# Depth resolution vector for the iverson model
depths = np.arange(0.2,3.1,0.1)

# Velocity threshold for failure
threshold = 80 # mm/yr

# Number of points to calibrate
Num_cal = 200


######################################################
######################################################
# set paths to important files nd open relevant data
######################################################
######################################################
# Model directory
rundir = "/exports/csce/datastore/geos/users/s1440040/FORESEE/FORESEE_dev/foresee_python_scripts_science_exp/Calibration/TestMC/"
# failure data files
faildir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Data_Marina_tests/InSAR_data_failure/"
failfile = faildir + "All_1st_failtime__threshold"+str(threshold)+"mmyr.bil"
prefailfile = faildir + "All_1st_prefailtime__threshold"+str(threshold)+"mmyr.bil"

# topography files
topodir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/"
demfile = topodir + "eu_dem_AoI_epsg32633.bil"
slopefile = topodir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road file
roaddir = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Road/"
roadfile = roaddir + "Road_line.shp"

# Load rasters into arrays for DEM, slope, failtimes and prefailtimes for a given failure threshold.
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)
prefailarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(prefailfile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)


######################################################
######################################################
# Select the  pixels to calibrate
######################################################
######################################################

# calculate distances to main road.
# We wanted to calibrate the points closest to the road, as this is where the landslides are likely to have a greater impact on the road
distarr = fn.calc_dist2road(demarr.shape, roadline, geotransform,demarr)


# Now select pixels based on the distance array and the number of pixels we want.
# Note: the selection process could definitely be refined.
final_selectarr = fn.select_pixels(distarr, Num_cal,demarr, failarr, prefailarr)



######################################################
######################################################
# Run the calibration
######################################################
######################################################


# 1. Load rainfall data from 03/09/2016 until the end of 2018 (say). - the file is ready
rain = pd.read_csv("/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Precipitation/GPM_data/2014-01-01_to_2019-12-31_Intensity.csv")
# to make this rain file:
# python PPT_CMD_RUN.py --ProdTP GPM_D --StartDate 2016-09-03 --EndDate 2018-12-31 --ProcessDir /home/willgoodwin/PostDoc/Foresee/Data/Precipitation/GPM_data/ --SptSlc /home/willgoodwin/PostDoc/Foresee/Data/Topography/eu_dem_v11_E40N20_AoI.bil --OP --DirOut /home/willgoodwin/PostDoc/Foresee/Calibration/TestMC/

# Now calibrate the points
fn.calibrate_points_MC(final_selectarr, demarr, slopearr, failarr, prefailarr, rain, depths, Nruns, rundir, itermax, Num_cal)




# PICK UP HERE!

# Things to do to improve this thing:
# 1.



# 3. What do you do about GW depth? data shows it to be anywhere between 0.2m and 10m ...

# 4. try further optimising by running MC in conjunction with GA stuff.

# 5. Figure out this failure threshold thingy. But for now just use 80 mm/yr.





"""fig_height = 7; fig_width = 7
fig=plt.figure(6, facecolor='White',figsize=[fig_width, fig_height])

ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)


dem_mask = np.ma.masked_where(demarr <= -10, demarr)
Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

ax1.add_line(l)

Map2 = ax1.imshow(distarr, interpolation='None', cmap=plt.cm.jet, vmin = np.amin(distarr), vmax =  np.amax(distarr), alpha = 0.3)

select_mask = np.ma.masked_where(final_selectarr <= 0, final_selectarr)
Map2 = ax1.imshow(select_mask, interpolation='None', cmap=plt.cm.jet, vmin = 0, vmax =  1, alpha = 1.)

print ('figure saved')
plt.savefig(rundir+ 'selected_points_for_calibration.jpg')"""



######################################################
######################################################
# This is an example of how the MC works
######################################################
######################################################


# set the parameters for the MC runs
'''MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.1, D_0_min = 1e-6,K_sat_min = 1e-8, d_min = 0.5, Iz_over_K_steady_min = 0.1, friction_angle_min = 0.2, cohesion_min = 5000, weight_of_water_min = 9800, weight_of_soil_min = 15000,
      alpha_max = 0.11, D_0_max = 1e-4,K_sat_max = 1e-6, d_max = 3,Iz_over_K_steady_max = 0.8, friction_angle_max = 0.5, cohesion_max = 20000, weight_of_water_max = 9801, weight_of_soil_max = 25000, depths = depths)

# Now run it
MCrun.run_MC_failure_test(df["duration_s"].values, df["intensity_mm_sec"].values,
                          n_process = 2, output_name = "test_MC.csv", n_iterations = 10, replace = True)'''




# here's how a single run works

#mymodel = iverson.iverson_model(alpha = 0.51, D_0 = 5e-6,K_sat = 5e-8, d = 2,Iz_over_K_steady = 0.2, friction_angle = 0.38, cohesion = 12000, weight_of_water = 9800, weight_of_soil = 19000, depths = depths)
#mymodel.run(df.duration_s.values, df.intensity_mm_sec.values)

# and here are the outputs
#mymodel.cppmodel.output_times
#mymodel.cppmodel.output_depthsFS
#mymodel.cppmodel.output_minFS
#mymodel.cppmodel.output_PsiFS
#mymodel.cppmodel.output_durationFS
#mymodel.cppmodel.output_intensityFS
#mymodel.cppmodel.output_failure_times
#mymodel.cppmodel.output_failure_mindepths
#mymodel.cppmodel.output_failure_maxdepths
#mymodel.cppmodel.output_Psi_timedepth
#mymodel.cppmodel.output_FS_timedepth
#mymodel.cppmodel.output_failure_bool
