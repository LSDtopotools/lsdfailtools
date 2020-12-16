import matplotlib as mpl


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import DivergingNorm

import matplotlib.patches as mpatches
from itertools import product
from itertools import islice
from collections import OrderedDict
import seaborn as sns
import  datetime
import pandas as pd
import numpy as np
import shapefile
import itertools
import sys
import json
import os
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.font_manager as fm

# Importing the model
import lsdfailtools.iverson2000 as iverson


with open("file_paths_visualisation.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["figures_dir"]))


# Model directory
rain_start_day = FILE_PATHS["rain_start_day"]
rain_start_month = FILE_PATHS["rain_start_month"]
rain_start_year = FILE_PATHS["rain_start_year"]



######################################################
######################################################
# A figure to map calibrated points
######################################################
######################################################
def map_calibrated(demarr, calibrated, road, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)


	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	ax1.add_line(road)

	calib_arr = 0* demarr
	for i in range(len(calibrated)):
		x = calibrated['col'].iloc[i]
		y = calibrated['row'].iloc[i]
		calib_arr[y-2:y+2,x-2:x+2] = 1

	calib_mask = np.ma.masked_where(calib_arr == 0., calib_arr)
	Map1 = ax1.imshow(calib_mask, interpolation='None', cmap=plt.cm.autumn,
	    vmin = 0, vmax = 1, alpha = 1.)
	plt.title("Calibrated Failure Points", fontsize = 26, pad = 10.)
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




######################################################
######################################################
# A figure to plot the distribution of calibrated points
######################################################
######################################################
def plot_failtime(calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	ax1.set_xlabel('time (days)')

	late_failtimes = sorted(list(set(list(calibrated['observed_failtime']))))

	for i in range(len(late_failtimes)):
		df = calibrated[calibrated['observed_failtime'] == late_failtimes[i]]
		df['time_of_failure'] = df['time_of_failure']/(24*3600)

		ax1.hist(df['time_of_failure'], bins = 20, color = plt.cm.jet(i/len(late_failtimes)), lw = 0, density = True, alpha = 0.7)

	plt.tight_layout()
	plt.savefig(fig_name)

######################################################
######################################################
# Figure to map the distribution of failtimes
# (calibrated and validated points) along with precipitation data
######################################################
######################################################


def plot_failtime_calib_valid(calibrated, validated, rain, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax12 = ax1.twiny()


	plot_colour = ['k', 'r', 'b']
	# plot rainfall data
	rainfall = ax12.fill_between(rain['rainfall_mm'], 0, rain['time_s']/(3600*24), facecolor = plot_colour[0], lw = 0.1, alpha = 0.5, label = "Rainfall")



	for i in range(len(validated)):

		O = validated['time_of_failure'].iloc[i]/(24*3600)
		C = validated['observed_failtime'].iloc[i]/(24*3600)

		valid = ax1.scatter(C,O, marker = '+', facecolor = plot_colour[2], lw = 2, alpha = 0.7, label = "Validation")
	for i in range(len(calibrated)):

		O = calibrated['time_of_failure'].iloc[i]/(24*3600)
		C = calibrated['observed_failtime'].iloc[i]/(24*3600)

		calib = ax1.scatter(C,O, marker = 'o', facecolor = plot_colour[1], lw = 0.0, alpha = 0.7, label = "Calibration")

	# plots the x = y black time around which the calibration points sit.
	ax1.plot([0,max(calibrated['time_of_failure'])/(24*3600)], [0,max(calibrated['time_of_failure'])/(24*3600)], '-k', lw = 2)

	ax1.set_xlabel('Observed failure time (days)', fontsize = 16, labelpad = 10. )
	ax1.set_ylabel('Modelled failure time (days)', fontsize = 16, labelpad = 10.)
	ax12.set_xlabel('Rainfall (mm/day)', fontsize = 16, labelpad = 10.)
	ax1.set_ylim([-20,max(rain['time_s']/(3600*24))+20])
	ax1.set_xlim([0,max(validated['observed_failtime']/(3600*24))+20])
	ax12.set_xlim([0,max(rain['rainfall_mm'])+5])
	ax1.tick_params(axis='x', labelsize=16)
	ax1.tick_params(axis='y', labelsize=16)
	ax12.tick_params(axis='x', labelsize=16)
	plt.legend((rainfall, calib, valid),
           ("Rainfall", "Calibration", "Validation"),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=14)

	plt.savefig(fig_name)

######################################################
######################################################
# A figure to plot how the model parameters change with slope and elevation
######################################################
######################################################
def plot_parameters (calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

	paramlist = list(calibrated.columns.values[2:10])
	paramlist.remove('weight_of_water')


	for i in range(len(paramlist)):

		ax1 =  plt.subplot2grid((2,len(paramlist)),(0,i),colspan=1, rowspan=1)
		ax1.set_xlabel('Slope (m/m)')
		ax1.set_ylabel(paramlist[i])

		ax2 =  plt.subplot2grid((2,len(paramlist)),(1,i),colspan=1, rowspan=1)
		ax2.set_xlabel('Elevation (m/m)')
		ax2.set_ylabel(paramlist[i])

		shist, sbins = np.histogram(calibrated['S'], bins  = 7)

		for s in range(len(sbins)-1):
			sdf = calibrated[calibrated['S'] >= sbins[s]]
			sdf = sdf[sdf['S'] < sbins[s+1]]

			phist, pbins = np.histogram(sdf[paramlist[i]], bins = 10, density = True)
			scale = 0.05*sbins[s]/max(phist)
			ax1.fill_betweenx(pbins[1:], sbins[s]-phist*scale, sbins[s]+phist*scale, alpha = 0.7)

		shist, sbins = np.histogram(calibrated['Z'], bins  = 7)

		for s in range(len(sbins)-1):
			sdf = calibrated[calibrated['Z'] >= sbins[s]]
			sdf = sdf[sdf['Z'] < sbins[s+1]]

			phist, pbins = np.histogram(sdf[paramlist[i]], bins = 10, density = True)
			scale = 0.05*sbins[s]/max(phist)
			ax2.fill_betweenx(pbins[1:], sbins[s]-phist*scale, sbins[s]+phist*scale, alpha = 0.7)

	plt.tight_layout()
	plt.savefig(fig_name)




def map_validation_updated(rain, depths, calibrated, validated, road, demarr, slopearr, failarr, failinterval, fig_height, fig_width, fig_name):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	calib_arr = 0* demarr
	for i in range(len(calibrated)):
		x = calibrated['col'].iloc[i]
		y = calibrated['row'].iloc[i]
		calib_arr[y-2:y+2,x-2:x+2] = 1


	valid_arr = 0* demarr
	for i in range(len(validated)):
		x = int(validated['col'].iloc[i])
		y = int(validated['row'].iloc[i])
		if x >=2 and y >= 2 and x <= len(demarr[0]) - 2 and y <= len(demarr) -2:

			if validated['time_of_failure'].iloc[i] <= validated['observed_failtime'].iloc[i] + failinterval and validated['time_of_failure'].iloc[i] >= validated['observed_failtime'].iloc[i] - failinterval :
				valid_arr[y-2:y+2,x-2:x+2] = 4 #success
			elif validated['time_of_failure'].iloc[i] > validated['observed_failtime'].iloc[i] + failinterval:
				valid_arr[y-2:y+2,x-2:x+2] = 3 # too late
			elif validated['time_of_failure'].iloc[i] < validated['observed_failtime'].iloc[i] - failinterval:
				valid_arr[y-2:y+2,x-2:x+2] = 2 # too soon


	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	ax1.add_line(road)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	calib_valid = np.where(valid_arr < 1, calib_arr, valid_arr)

	calib_valid_mask = np.ma.masked_where(calib_valid <= 0, calib_valid)

	unique_values = [1.0, 2.0, 3.0, 4.0]


	Map2 = ax1.imshow(calib_valid_mask, interpolation='None', cmap=plt.cm.jet_r, vmin = np.amin(calib_valid_mask) , vmax=  np.amax(calib_valid_mask), alpha = 1.)


	float_list_values = list(map(float, unique_values))

	unique_values_categories = [ "Calibrated", "Pre failure", "Post failure", "At failure" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = [Map2.cmap(Map2.norm(value)) for value in unique_values]
	# create a patch (proxy artist) for every color
	#need to make a dictionary to relate the values and the timing of the failure
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = list(unique_values_dict.keys())[i])) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)



	# print the proportions of each
	unique, counts = np.unique(calib_valid_mask, return_counts=True)
	print(unique, counts)


	plt.tight_layout()
	plt.savefig(fig_name)
	ax1.clear()


def map_validation_arrays(rain, depths, calibrated, validated, road, demarr, slopearr, failarr, failinterval, fig_height, fig_width, fig_name):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	calib_arr = 0* demarr
	for i in range(len(calibrated)):
		x = calibrated['col'].iloc[i]
		y = calibrated['row'].iloc[i]
		calib_arr[y-2:y+2,x-2:x+2] = 1

	valid_arr = 0* demarr
	at_failure = 0*demarr
	pre_failure = 0 * demarr
	post_failure = 0 * demarr
	for i in range(len(validated)):
		x = int(validated['col'].iloc[i])
		y = int(validated['row'].iloc[i])
		if x >=2 and y >= 2 and x <= len(demarr[0]) - 2 and y <= len(demarr) -2:

			if validated['time_of_failure'].iloc[i] <= validated['observed_failtime'].iloc[i] + failinterval and validated['time_of_failure'].iloc[i] >= validated['observed_failtime'].iloc[i] - failinterval :
				at_failure[y-2:y+2,x-2:x+2] = 4 #success
			elif validated['time_of_failure'].iloc[i] > validated['observed_failtime'].iloc[i] + failinterval:
				post_failure[y-2:y+2,x-2:x+2] = 3 # too late
			elif validated['time_of_failure'].iloc[i] < validated['observed_failtime'].iloc[i] - failinterval:
				pre_failure[y-2:y+2,x-2:x+2] = 2 # too soon

	ax1.add_line(road)
	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)


	at_failure_mask = np.ma.masked_where(at_failure <= 0, at_failure)
	pre_failure_mask = np.ma.masked_where(pre_failure <= 0, pre_failure)
	post_failure_mask = np.ma.masked_where(post_failure <= 0, post_failure)
	calib_arr_mask = np.ma.masked_where(calib_arr <= 0, calib_arr)

	Map4 = ax1.imshow(post_failure_mask, interpolation='None', cmap=plt.cm.winter, alpha = 1.) # dark blue
	Map3 = ax1.imshow(pre_failure_mask, interpolation='None', cmap=plt.cm.cool, alpha = 1.) # light blue

	Map2 = ax1.imshow(at_failure_mask, interpolation='None', cmap=plt.cm.cool_r, alpha = 1.) # pink
	Map5 = ax1.imshow(calib_arr_mask, interpolation='None', cmap=plt.cm.autumn_r, alpha = 1.) # yellow

	unique_values = [1.0, 2.0, 3.0, 4.0]

	float_list_values = list(map(float, unique_values))

	unique_values_categories = [ "Calibrated", "Pre failure", "Post failure", "At failure" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = ['yellow', 'cyan', 'blue', 'fuchsia']
	# create a patch (proxy artist) for every color
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = list(unique_values_dict.keys())[i])) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, fontsize = 20, bbox_to_anchor=(0, 0, 0.5, 0.5), loc='lower left')#, borderaxespad=0.5)
	plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
	left=False,        # ticks along the top edge are off
    labelbottom=False,
	labelleft=False) # labels along the bottom edge are off

	# image size is 993x1405 px
	# in km this is 24x35 km
	pxl_size = 35/1405
	scalebar = ScaleBar(pxl_size, 'km', font_properties = {"size": 20})
	plt.gca().add_artist(scalebar)
	plt.tight_layout()
	plt.savefig(fig_name)
	ax1.clear()
	plt.cla()


def map_validation_arrays_zoom(rain, depths, calibrated, validated, road, demarr, slopearr, failarr, failinterval, fig_height, fig_width, fig_name):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)


	calib_arr = 0* demarr
	for i in range(len(calibrated)):
		x = calibrated['col'].iloc[i]
		y = calibrated['row'].iloc[i]
		calib_arr[y-2:y+2,x-2:x+2] = 1

	valid_arr = 0* demarr
	at_failure = 0*demarr
	pre_failure = 0 * demarr
	post_failure = 0 * demarr
	for i in range(len(validated)):
		x = int(validated['col'].iloc[i])
		y = int(validated['row'].iloc[i])
		if x >=2 and y >= 2 and x <= len(demarr[0]) - 2 and y <= len(demarr) -2:

			if validated['time_of_failure'].iloc[i] <= validated['observed_failtime'].iloc[i] + failinterval and validated['time_of_failure'].iloc[i] >= validated['observed_failtime'].iloc[i] - failinterval :
				at_failure[y-2:y+2,x-2:x+2] = 4 #success
			elif validated['time_of_failure'].iloc[i] > validated['observed_failtime'].iloc[i] + failinterval:
				post_failure[y-2:y+2,x-2:x+2] = 3 # too late
			elif validated['time_of_failure'].iloc[i] < validated['observed_failtime'].iloc[i] - failinterval:
				pre_failure[y-2:y+2,x-2:x+2] = 2 # too soon

	ax1.add_line(road)

	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)


	at_failure_mask = np.ma.masked_where(at_failure <= 0, at_failure)
	pre_failure_mask = np.ma.masked_where(pre_failure <= 0, pre_failure)
	post_failure_mask = np.ma.masked_where(post_failure <= 0, post_failure)
	calib_arr_mask = np.ma.masked_where(calib_arr <= 0, calib_arr)

	Map4 = ax1.imshow(post_failure_mask, interpolation='None', cmap=plt.cm.winter, alpha = 1.) # dark blue
	Map3 = ax1.imshow(pre_failure_mask, interpolation='None', cmap=plt.cm.cool, alpha = 1.) # light blue

	Map2 = ax1.imshow(at_failure_mask, interpolation='None', cmap=plt.cm.cool_r, alpha = 1.) # pink
	Map5 = ax1.imshow(calib_arr_mask, interpolation='None', cmap=plt.cm.autumn_r, alpha = 1.) # yellow

	unique_values = [1.0, 2.0, 3.0, 4.0]

	float_list_values = list(map(float, unique_values))

	unique_values_categories = [ "Calibrated", "Pre failure", "Post failure", "At failure" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = ['yellow', 'cyan', 'blue', 'fuchsia']
	# create a patch (proxy artist) for every color
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = list(unique_values_dict.keys())[i])) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, fontsize = 20, bbox_to_anchor=(0, 0, 0.5, 0.5), loc='lower left')#, borderaxespad=0.5)
	plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
	left=False,        # ticks along the top edge are off
    labelbottom=False,
	labelleft=False) # labels along the bottom edge are off

	plt.xlim(300,1100)
	plt.ylim(700, 300)
	# image size is 993x1405 px
	# in km this is 24x35 km
	pxl_size = 35/1405
	scalebar = ScaleBar(pxl_size, 'km', font_properties = {"size": 20})
	plt.gca().add_artist(scalebar)
	plt.tight_layout()
	plt.savefig(fig_name)
	ax1.clear()
	plt.cla()



def map_validation_colorbar(rain, depths, calibrated, validated, road, demarr, slopearr, failarr, failinterval, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	calib_arr = 0* demarr
	for i in range(len(calibrated)):
		x = calibrated['col'].iloc[i]
		y = calibrated['row'].iloc[i]
		calib_arr[y-2:y+2,x-2:x+2] = 1


	valid_arr = 0* demarr
	for i in range(len(validated)):
		x = int(validated['col'].iloc[i])
		y = int(validated['row'].iloc[i])
		if x >=2 and y >= 2 and x <= len(demarr[0]) - 2 and y <= len(demarr) -2:

			if validated['time_of_failure'].iloc[i] <= validated['observed_failtime'].iloc[i] + failinterval and validated['time_of_failure'].iloc[i] >= validated['observed_failtime'].iloc[i] - failinterval :
				valid_arr[y-2:y+2,x-2:x+2] = 4
			elif validated['time_of_failure'].iloc[i] > validated['observed_failtime'].iloc[i] + failinterval:
				valid_arr[y-2:y+2,x-2:x+2] = 2
			elif validated['time_of_failure'].iloc[i] < validated['observed_failtime'].iloc[i] - failinterval:
				valid_arr[y-2:y+2,x-2:x+2] = 1
			valid_arr[y-2:y+2,x-2:x+2] = (validated['time_of_failure'].iloc[i] - validated['observed_failtime'].iloc[i]) /(24*3600)

	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	ax1.add_line(road)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)




	valid_mask = np.ma.masked_where(valid_arr == -0, valid_arr)
	Map2 = ax1.imshow(valid_mask, interpolation='None', norm=DivergingNorm(0), cmap=plt.cm.jet, vmin = np.amin(valid_mask), vmax = np.amax(valid_mask), alpha = 1.)


	calib_mask = np.ma.masked_where(calib_arr == 0., calib_arr)
	Map1 = ax1.imshow(calib_mask, interpolation='None', cmap=plt.cm.cool, vmin = 0, vmax = 1, alpha = 1.)



	unique_values = [1.0]

	float_list_values = list(map(float, unique_values))

	unique_values_categories = [ "Calibrated"]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, according to the colormap used by imshow
	colors = ['fuchsia']
	# create a patch (proxy artist) for every color
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = list(unique_values_dict.keys())[i])) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, fontsize = 12, bbox_to_anchor=(0, 0, 0.5, 0.5), loc='lower left')#, borderaxespad=0.5)
	plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
	left=False,        # ticks along the top edge are off
    labelbottom=False,
	labelleft=False) # labels along the bottom edge are off
	plt.title('Difference in modelled and observed failure times (days)', fontsize = 20, pad = 10.)
	norm = mpl.colors.Normalize(vmin=np.amin(valid_mask), vmax=np.amax(valid_mask))
	cmap = plt.cm.jet
	cax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
	cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='proportional')
	plt.savefig(fig_name)
	plt.cla()


######################################################
######################################################
# A figure to map validation results
######################################################
######################################################

def plot_rain(rain, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	rainlist = [datetime.datetime(rain_start_year, rain_start_month, rain_start_day)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))

	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])
	ax1.set_xlabel("Year", fontsize = 20, labelpad = 10)
	ax1.set_ylabel("Precipitation (mm/day)", fontsize = 20, labelpad = 10 )
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	plt.tight_layout()
	plt.savefig(fig_name)

######################################################
######################################################
# A figure to plot precipitation and when the failures happens
######################################################
######################################################

def plot_rain_failures(rain, calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax2 =  ax1.twinx()

	rainlist = [datetime.datetime(rain_start_year, rain_start_month, rain_start_day)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))

	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])

	for i in range(len(calibrated)):
		time = datetime.datetime(rain_start_year, rain_start_month, rain_start_day) + datetime.timedelta(0,int(calibrated['time_of_failure'].iloc[i]))
		ax2.scatter(time, calibrated['S'].iloc[i])

	ax1.set_xlim(left = datetime.datetime(rain_start_year, rain_start_month, rain_start_day), right = datetime.datetime(2018, 12, 31))
	ax1.set_xlabel("Year", fontsize = 20, labelpad = 10)
	ax1.set_ylabel("Precipitation (mm/day)", fontsize = 20, labelpad = 10 )
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.tight_layout()
	plt.savefig(fig_name)



def density_plot(validated, fig_width, fig_height, fig_name):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax2 =  ax1.twinx()

	observed_failtime_list = []
	modelled_failtime_list = []
	for i in range(len(validated)):
		observed_failtime = validated['observed_failtime'].iloc[i]/(24*3600)
		observed_failtime_list.append(observed_failtime)

		modelled_failtime = validated['time_of_failure'].iloc[i]/(24*3600) # this is the time since start date
		modelled_failtime_list.append(modelled_failtime)


	df_failures = pd.DataFrame()
	df_failures['observed_failures'] = observed_failtime_list
	df_failures['modelled_failures'] = modelled_failtime_list

	for a in [df_failures['observed_failures'], df_failures['modelled_failures']]:
		ax1= sns.histplot(df_failures['observed_failures'], bins=range(1, int(max(modelled_failtime_list)), 50), ax=ax1, kde=False, color = 'tab:blue', alpha=0.5)
		ax1= sns.histplot(df_failures['modelled_failures'], bins=range(1, int(max(modelled_failtime_list)), 50), ax=ax1, kde=False, color = 'tab:orange', alpha=0.5)

	ax1.set_xlim([0, max(modelled_failtime_list)])
	ax2 = sns.kdeplot(data = df_failures, x = 'observed_failures', label = 'Observed Failures')
	ax2 = sns.kdeplot(data = df_failures, x = 'modelled_failures', label = 'Modelled Failures')

	ax2.set_ylabel('Probability density function', fontsize = 16, labelpad = 10.)
	ax1.set_xlabel('Failure Time (days)', fontsize = 16, labelpad = 10.)
	ax1.set_ylabel('Number of failure events', fontsize = 16, labelpad = 10.)
	plt.xlabel('Failure time (days)') # might be good to convert this into a date axis rather than absolute values
	plt.legend()
	plt.legend(loc='upper right',ncol=1,fontsize=14)
	plt.tight_layout()
	plt.savefig(fig_name)


def time_interval (row):
	if row['modelled_failures']<=150 :
		return 150
	if (row['modelled_failures']>150)&(row['modelled_failures']<=300):
		return 300
	if (row['modelled_failures']>300)&(row['modelled_failures']<=450):
		return 450
	if (row['modelled_failures']>450)&(row['modelled_failures']<=600):
		return 600
	if (row['modelled_failures']>600)&(row['modelled_failures']<=750):
		return 750
	if (row['modelled_failures']>750)&(row['modelled_failures']<=900):
		return 900
	if (row['modelled_failures']>900)&(row['modelled_failures']<=1050):
		return 1050
	if (row['modelled_failures']>1050)&(row['modelled_failures']<=1200):
		return 1200
	if (row['modelled_failures']>1200)&(row['modelled_failures']<=1350):
		return 1350
	if (row['modelled_failures']>1350)&(row['modelled_failures']<=1500):
		return 1500
	if row['modelled_failures']>1500 :
		return 1700


def time_split_violin_plot(validated, fig_width, fig_height, fig_name):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	observed_failtime_list = []
	modelled_failtime_list = []
	for i in range(len(validated)):
		observed_failtime = validated['observed_failtime'].iloc[i]/(24*3600)
		observed_failtime_list.append(observed_failtime)

		modelled_failtime = validated['time_of_failure'].iloc[i]/(24*3600) # this is the time since start date
		modelled_failtime_list.append(modelled_failtime)


	df_failures = pd.DataFrame()
	df_failures['observed_failures'] = observed_failtime_list
	df_failures['modelled_failures'] = modelled_failtime_list
	print(df_failures.head(5))

	df_failures['time_interval'] = df_failures.apply (lambda row: time_interval(row), axis=1)

	print(df_failures.head(5))

	ax = sns.violinplot(x="time_interval", y="observed_failures", data=df_failures)


	plt.xlabel('Model Failure time interval (days)', fontsize=14) # might be good to convert this into a date axis rather than absolute values
	plt.ylabel('Observed failure time(days)', fontsize=14) # might be good to convert this into a date axis rather than absolute values

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.savefig(fig_name)
