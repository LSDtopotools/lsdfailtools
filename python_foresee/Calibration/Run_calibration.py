
######################################################
######################################################
# Importing modules
######################################################
######################################################

# Importing the model
import sys
sys.path.insert(0,'../../lsdfailtools-master/lsdfailtools')# Importing the model
import iverson2000 as iverson

# I'll need that to process the outputs
import os
import json
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import shapefile
import itertools

# importing custom functions
import Calibration_functions as fn




################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

with open("../../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


# Model directory
rundir = FILE_PATHS["rain_intensity_caliv_valid"]

# parameter files
# the params that fix the iterations and the number of calibrated points in our "sortaMarkovChainMonteCarlo"
Cal_params_file = rundir + "Calibration_parameters.csv"
# the params used to define the physical soil properties in the iverson MC runs
Iverson_MC_params_file = rundir + "Iverson_MC_parameters.csv"

# failure data files
faildir = FILE_PATHS["ground_motion_failure"]

failfile = faildir + "Failtime_1_since_20141020.bil"

# topography files
topo_dir = FILE_PATHS["topo_dir"]

demfile = topo_dir + "eu_dem_AoI_epsg32633.bil"
slopefile = topo_dir + "eu_dem_AoI_epsg32633_SLOPE.bil"
cutfile = topo_dir + "eu_dem_v11_E40N20_AoI.bil" # this one needs to be in WGS84 to interface with NASA

# road file
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp" # this is in EPSG:32633

# Rain files - This is the path to creat the rainfall input

# Rain file
raindir = FILE_PATHS["rain_dir"]
PPT_dir = FILE_PATHS["PPT_dir"]
PPT_cmd_run = PPT_dir + "PPT_CMD_RUN.py"

# piezometry files
piezo_path = FILE_PATHS["piezo_dir"]
piezo_data_file = piezo_path + "data_piezometer.csv"

# NOTE: the rainfile is a bit different since we create it for the simulation


######################################################
######################################################
# Read the files to create the necessary variables
######################################################
######################################################
Nodata_value = -9999.

# Read the calibration params
Cal_params = pd.read_csv(Cal_params_file)

# Define the dates to generate the rainfall file
StartDate = Cal_params.at[0,'StartDate']
EndDate = Cal_params.at[0,'EndDate']

# Read the Iverson params
Iverson_MC_params = pd.read_csv(Iverson_MC_params_file)

# Load rasters into arrays for slope, failtimes and prefailtimes for a given failure threshold.
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)


######################################################
######################################################
# Make/read the inputs
######################################################
######################################################



# CREATE then Read the rainfall - that's a model input
#os.system("python " + PPT_cmd_run + " --ProdTP GPM_D --StartDate " + StartDate + " --EndDate " + EndDate + " --ProcessDir " + raindir + " --SptSlc " + cutfile + "--OP --DirOut " + rundir)
# to make this file:
#python PPT_CMD_RUN.py --ProdTP GPM_D --StartDate 2014-01-01 --EndDate 2019-12-31 --ProcessDir /home/willgoodwin/PostDoc/Foresee/Data/Precipitation/GPM_data/ --SptSlc /home/willgoodwin/PostDoc/Foresee/Data/Topography/eu_dem_v11_E40N20_AoI.bil --OP --DirOut /home/willgoodwin/PostDoc/Foresee/Calib_Valid/Current_test/
# or use the module Boris made

# DOESNAE WORK, for now do with the existing files
rainfile = rundir + StartDate + "_to_" + EndDate + "_Intensity.csv"
rain = pd.read_csv(rainfile)

# Read the piezo data - that's the other model input
Piezo_data = pd.read_csv(piezo_data_file)
GW_d_ini = fn.GW_depth_ini(Piezo_data, StartDate)






######################################################
######################################################
# Select pixels
######################################################
######################################################

# calculate distances to main road.
# We wanted to calibrate the points closest to the road, as this is where the landslides are likely to have a greater impact on the road
distarr = fn.calc_dist2road(slopearr.shape, roadline, geotransform)

# Now select pixels based on the distance array and the number of pixels we want.
# Note: the selection process could definitely be refined.
final_selectarr = fn.select_pixels(distarr, failarr, Cal_params.at[0,'Num_cal'])


######################################################
######################################################
# Run the calibration
######################################################
######################################################


# Now calibrate the points
fn.calibrate_points_MC(final_selectarr, demarr, slopearr, failarr, rain, GW_d_ini, Cal_params, Iverson_MC_params, rundir)




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
"""MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.1, D_0_min = 1e-6,K_sat_min = 1e-8, d_min = 0.5, Iz_over_K_steady_min = 0.1, friction_angle_min = 0.2, cohesion_min = 5000, weight_of_water_min = 9800, weight_of_soil_min = 15000,
      alpha_max = 0.11, D_0_max = 1e-4,K_sat_max = 1e-6, d_max = 3,Iz_over_K_steady_max = 0.8, friction_angle_max = 0.5, cohesion_max = 20000, weight_of_water_max = 9801, weight_of_soil_max = 25000, depths = depths)

# Now run it
MCrun.run_MC_failure_test(df["duration_s"].values, df["intensity_mm_sec"].values,
                          n_process = 2, output_name = "test_MC.csv", n_iterations = 10, replace = True)"""




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
