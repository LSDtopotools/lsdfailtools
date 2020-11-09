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

#####################################################################################
#PRE-PROCESSING
#####################################################################################
# what is the actual input to the model? We only need the array with the predicted ground motion values
# we need the data with the row and column number for each of the pixels
# we need to match the pred data with the pixels. They are in the same order as in the full dataframe (input_data_complete)

predicted_gm = y_pred_gm_df["predicted_gm"]

input_data_complete_time_test = input_data_complete_time_test.reset_index()

# add the predicted values to the dataframe
input_data_complete_time_test = input_data_complete_time_test.join(predicted_gm)

#print(input_data_complete_time_test.head())


def process_dataframes(input_dataframe, data_source):

	# create 2 datasets depending on the datasource
	input_data = input_dataframe[input_dataframe['datasource'] == data_source]
	print(input_data.head(30))
	input_data = input_data.reset_index(drop=True)
	print(input_data.head(30))
	input_data['new_pixel_col'] = input_data['cols'].diff(periods=1)
	input_data['new_pixel_row'] = input_data['rows'].diff(periods=1)
	input_data['new_pixel'] = input_data['new_pixel_col'].abs() + input_data['new_pixel_row'].abs()
	print(input_data.head())
	#input_data['new_pixel'] = input_data['rows'].diff(periods=1)

	#input_data.to_csv("data_array_to_check_cosmo.csv")

	print(input_data.head(50))
	# select indices corresponding to new pixels
	#new_pixel_entries = input_data.loc[(input_data.new_pixel < 0)&(input_data.new_pixel > 0), "new_pixel"]
	#new_pixel_entries = input_data.loc[(input_data["new_pixel"] != 0.0), "new_pixel"]
	array =input_data["new_pixel"].to_numpy()
	print(type(array))
	#new_pixel_index = np.asarray(array)
	new_pixel_index = array.nonzero()[0]
	#new_pixel_index = list(new_pixel_index)
	print(new_pixel_index[770])
	#new_pixel_index = np.array(new_pixel_index).tolist()
	print(type(new_pixel_index))

	#new_pixel_index = new_pixel_index.tolist()

	#new_pixel_entries = input_data[(input_data.select_dtypes(include=['new_pixel']) != 0).any(1)]
	#df.loc[(df["B"] > 50) & (df["C"] == 900), "A"]
	# position of a new pixel.This is the index of the first entry in the dataframe for a new pixel.
	#new_pixel_index = new_pixel_entries.index.to_list()
	#print(new_pixel_index[770])
	#print(new_pixel_index)
	# the number of days for each pixel depends on whether it's cosmo skymed or Sentinel
	time_axis = new_pixel_index[1]-new_pixel_index[0]

	# we need a data array for sentinel and another one for cosmo skymed.
	data_array_empty = np.zeros((demarr.shape[0], demarr.shape[1], time_axis), dtype=np.float32)
	return data_array_empty, time_axis, new_pixel_index, input_data

#data_array_empty_sentinel, time_axis_sentinel, pixel_indices_sentinel, input_data_sentinel = process_dataframes(input_data_complete_time_test,0)
data_array_empty_cosmo, time_axis_cosmo, pixel_indices_cosmo, input_data_cosmo = process_dataframes(input_data_complete_time_test,1)

#print(input_data_sentinel)
print(np.shape(data_array_empty_cosmo))



def df_to_numpy(input_dataframe, pixel_indices, data_array_to_fill, time_axis):
	pixel_count = 0
	for i in range (len(pixel_indices)):
	#for i in range (2):
		# the x coord of the pixel
		x = input_dataframe.iloc[pixel_indices[i],1]
		#print("x{}".format(x))
		# the y coord of the pixel
		y = input_dataframe.iloc[pixel_indices[i],2]
		#print("y{}".format(y))
		for j in range ((time_axis)):
			#print(new_pixel_index[i+1]-new_pixel_index[i])
			#print(i)
			#print(j+((time_axis)*i))
			#print("j value: {}".format(j))
			#print(time_axis)
			# the time coordinate of the pixel

			t = input_dataframe.iloc[j+((time_axis)*i),0]
			print("x,y,t vals: {},{},{}".format(x,y,t))
			data_array_to_fill[x,y,j] = input_dataframe.iloc[j+((time_axis)*i),10]
			print("data array value:{}".format(data_array_to_fill[x,y,j]))
		pixel_count += 1
	return data_array_to_fill


#data_array_sentinel = df_to_numpy(input_data_sentinel, pixel_indices_sentinel,data_array_empty_sentinel, time_axis_sentinel)
#print(np.shape(data_array_sentinel))
#print(data_array_sentinel[2580,2684,:])
for i in range(len(pixel_indices_cosmo)-1):
	px = pixel_indices_cosmo[i+1]-pixel_indices_cosmo[i]
	if px != 22:
		print(px)


data_array_cosmo = df_to_numpy(input_data_cosmo, pixel_indices_cosmo,data_array_empty_cosmo, time_axis_cosmo)
print(np.shape(data_array_cosmo))
print(data_array_cosmo[1775,2448,:])
print(input_data_cosmo[824500:824510])
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

map_points(demarr, y_pred_gm, line, 10, 10, out_dir + 'test_predicted_gm_map_ml')
