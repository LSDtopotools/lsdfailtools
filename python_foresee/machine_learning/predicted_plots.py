################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import os
import re
import json
import gdal
import datetime
import itertools
import shapefile
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import functions_ground_motion as fgm

################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################
with open("../file_with_paths.json") as file_with_paths :
	FILE_PATHS = json.load(file_with_paths)


out_dir = FILE_PATHS["time_series_ml"]

topo_dir = FILE_PATHS["topo_dir"]

# new 10m DEM data for the AoI in the 32633 projection
slopefile = topo_dir + "10m_DEM_tinitaly/w45510_s10_SLOPE_AoI_32633.bil"
curvaturefile = topo_dir + "10m_DEM_tinitaly/w45510_s10_CURV_AoI_32633.bil"
aspectfile = topo_dir + "10m_DEM_tinitaly/w45510_s10_ASPECT_AoI_32633.bil"
demfile = topo_dir + "10m_DEM_tinitaly/w45510_s10_AoI_32633.bil"

# road file
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp" # this is in EPSG:32633

# load the results from the machine learning model (produced in the jupyter notebook)
machinelearning_dir = FILE_PATHS['time_series_ml']

# y_obs_gm and y_pred_gm are the ground motion results from the model.
y_obs_gm = machinelearning_dir +'true_data.csv'
y_pred_gm = machinelearning_dir + 'predicted_data.csv'

# these follow a time indexing system
y_obs_gm = pd.read_csv(y_obs_gm, header = None)
y_pred_gm = pd.read_csv(y_pred_gm, header = None)

# create dataframes with the observed and predicted data

y_obs_gm_df = pd.DataFrame(y_obs_gm)
y_obs_gm_df.columns = ['true_gm']

y_pred_gm_df = pd.DataFrame(y_pred_gm)
y_pred_gm_df.columns = ['predicted_gm']


# input_data_complete is the complete dataframe with all the information
input_data_complete = machinelearning_dir + 'non_duplicates_testing_final.csv'
input_data_complete = pd.read_csv(input_data_complete)#, index_col=0)
input_data_complete = pd.DataFrame(input_data_complete)

input_data_complete_time = input_data_complete.set_index('time')

input_data_complete_time.index = pd.to_datetime(input_data_complete_time.index)
input_data_complete_time_test = input_data_complete_time['20170501':'20180501']




# predicted_px_positions is an array where the entries correspond to the index in the input_data_complete
# where a new pixel starts.
predicted_px_positions = machinelearning_dir + 'test_array_rows_data.csv'
predicted_px_positions = pd.read_csv(predicted_px_positions, header=None)

# convert the DEM file froma raster to a 2D array for ingestion by the map
demarr, pixelWidth, (geotransform, inDs) = fgm.ENVI_raster_binary_to_2d_array(demfile)

# Read the road file
road = shapefile.Reader(roadfile)
roadline = np.array(road.shapes()[0].points)

# now convert it to pixel coordinates
roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
roadline = roadline.astype('int')
line = mlines.Line2D(roadline[:,0], roadline[:,1], linewidth = 1., color='black')


# what is the actual input to the model? We only need the array with the predicted ground motion values
# we need the data with the row and column number for each of the pixels
# we need to match the pred data with the pixels. They are in the same order as in the full dataframe (input_data_complete)

input_data_complete_time_test['new_pixel'] = input_data_complete_time_test['cols'].diff()
input_data_complete_time_test = input_data_complete_time_test.reset_index()

new_pixel_entries = input_data_complete_time_test.loc[(input_data_complete_time_test.new_pixel != 0)]

# position of a new pixel.This is the index of the first entry in the dataframe for a new pixel.
new_pixel_index = new_pixel_entries.index.to_list()


#input_data_complete_time_test.to_csv(out_dir + "input_data_complete_time_test.csv")
# the "pixel row" column is column 1 and the "pixel column" column is column 2
#print(input_data_complete.iloc[new_pixel_index[0],1])

# we need to make the array 3D so that we can set the time coordinate as well
#data_array = 0*demarr
# length of the time dimension

time_axis = new_pixel_index[1]-new_pixel_index[0]
data_array = np.zeros((demarr.shape[0], demarr.shape[1], time_axis), dtype=np.float32)

# the range of the loop is the length of the complete data
index = input_data_complete_time_test.index
number_of_rows = len(index)
timeseries_length = new_pixel_index[1] - new_pixel_index[0]

#print(new_pixel_index[2] - new_pixel_index[1])

pixel_count = 0



for i in range (len(new_pixel_index)+1):
#for i in range (2):
	# the x coord of the pixel
	x = input_data_complete_time_test.iloc[new_pixel_index[i],1]
	print("x{}".format(x))
	# the y coord of the pixel
	y = input_data_complete_time_test.iloc[new_pixel_index[i],2]
	print("y{}".format(y))
	#print("x and y locations: {},{}".format(x,y))
	for j in range ((new_pixel_index[i+1]-new_pixel_index[i])):
		#print(new_pixel_index[i+1]-new_pixel_index[i])
		print(i)

		print(j+((new_pixel_index[i+1]-new_pixel_index[i])*i))
		print("j value: {}".format(j))
		print(new_pixel_index[i+1]-new_pixel_index[i])
		# the time coordinate of the pixel

		t = input_data_complete_time_test.iloc[j+((new_pixel_index[i+1]-new_pixel_index[i])*i),0]
		print("t value: {}".format(t))
		print("x,y,t vals: {},{},{}".format(x,y,t))
		data_array[x,y,j] = y_pred_gm_df.iloc[j+((new_pixel_index[i+1]-new_pixel_index[i])*i),0]
		#data_array[1,2,3] = 1

		print("data array value:{}".format(data_array[x,y,j]))
		#print(data_array[-1,-1,-1])
	pixel_count += 1
print(np.shape(data_array))
print(data_array[ 2580,2684,:])

#data_array.to_csv("data_array.csv")


def map_points (demarr, points, road, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)


	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	ax1.add_line(road)

	points_arr = 0* demarr
	for i in range(len(points)):
		x = points['cols'].iloc[i]
		y = points['rows'].iloc[i]
		points_arr[y,x] = points['ground_motion']

	calib_mask = np.ma.masked_where(points_arr == 0., points_arr)
	Map1 = ax1.imshow(calib_mask, interpolation='None', cmap=plt.cm.autumn,
	    vmin = 0, vmax = 1, alpha = 1.)
	plt.title("points Failure Points", fontsize = 26, pad = 10.)
	plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
	left=False,        # ticks along the top edge are off
    labelbottom=False,
	labelleft=False) # labels along the bottom edge are off
	plt.tight_layout()
	plt.savefig(fig_name)

#map_points(demarr, y_pred_gm, line, 10, 10, out_dir + 'test_predicted_gm_map_ml')
