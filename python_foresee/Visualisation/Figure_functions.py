# I'll need that to process the outputs
import matplotlib as mpl
#only use this backend if using PuTTy
#mpl.use('Agg')

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

import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.font_manager as fm




import sys
sys.path.insert(0,'../Alldata_processing/InSAR')

import Insar_functions as fn
#import functions as fn

# Importing the model
import lsdfailtools.iverson2000 as iverson





######################################################
######################################################
# A figure to map calibrated points
######################################################
######################################################
def map_calibrated (demarr, calibrated, road, fig_height, fig_width, fig_name):

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
# A figure to plot the distribution of  calibrated points
######################################################
######################################################
def plot_failtime (calibrated, fig_height, fig_width, fig_name):

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

def plot_failtime_calib_valid(calibrated, validated, rain, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	#ax11 = ax1.twinx()
	ax12 = ax1.twiny()


	#ax11.fill_between(rain['time_s']/(3600*24), 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.5)
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
	#plt.legend(prop={'size': 14})
	#plt.tight_layout()
	plt.savefig(fig_name)

######################################################
######################################################
# A figure to plot how the moedel parameters change with slope and elevation
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



######################################################
######################################################
# A figure to map validation results
######################################################
######################################################
def map_validation(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, road, fig_height, fig_width, fig_name):

	confusion = 0*np.copy(slopearr)
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	# how do you select the parameters?
	# based on location and slope

	# make slope bins in the calibrated df
	#shist, sbins = np.histogram(calibrated['S'], bins  = 10)
	sbins = np.arange(0,np.amax(slopearr), 0.05)

	for i,j in product(range(slopearr.shape[0]), range(slopearr.shape[1])):
	#for i,j in product(range(400,410,1), range(800,810,1)):

		if failarr[i,j] > 0.:
			print (i,j)

			# this is our slope
			S = slopearr[i,j]
			print (S)

			# find out in which bin it is
			lesser = sbins[np.where(sbins < S)[0][-1]]
			greater = sbins[np.where(sbins >= S)[0][0]]

			#find the points that have been calibrated in this range
			sdf = calibrated[calibrated['S'] >= lesser]
			sdf = sdf[sdf['S'] < greater]

			# If there are indeed points in this category
			if len(sdf)<1:
				print ('woops, there is nothing here')
			else:
				dist = np.sqrt((j-sdf['row'])**2 + (i-sdf['col'])**2)
				where = np.where(np.asarray(dist) == min(np.asarray(dist)))[0]

				select_df = sdf.iloc[where]
				mean_df = select_df.mean(axis = 0)

				mymodel = iverson.iverson_model(alpha = S,
					D_0 = mean_df['D_0'],
					K_sat = mean_df['K_sat'],
					d = mean_df['d'],
					Iz_over_K_steady = mean_df['Iz_over_K_steady'],
          			friction_angle = mean_df['friction_angle'],
          			cohesion = mean_df['cohesion'],
          			weight_of_water = 9800,
          			weight_of_soil = mean_df['weight_of_soil'],
          			depths = depths)

				mymodel.run(rain.duration_s.values, rain.intensity_mm_sec.values)

				failures = mymodel.cppmodel.output_failure_times
				failures = failures[failures > 1.][0]

				error = 25 * 24*3600

				if failures <= failarr[i,j] + error and failures >= prefailarr[i,j] - error:
					confusion[i,j] = 3 # success
					confusion[i-2:i+2,j-2:j+2] = 3 # success - MR: why is there a margin here??
				elif failures < prefailarr[i,j] - error:
					confusion[i,j] = 2 # too soon
					confusion[i-2:i+2,j-2:j+2] = 2 # too soon
				elif failures > failarr[i,j] + error:
					confusion[i,j] = 1 # too late
					confusion[i-2:i+2,j-2:j+2] = 1 # too late

		#if i > 350:
		#	break


	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)


	calib_arr = 0* demarr
	for i in range(len(calibrated)):
		x = calibrated['col'].iloc[i]
		y = calibrated['row'].iloc[i]
		calib_arr[y-2:y+2,x-2:x+2] = 4

	new_arr_test = np.where(confusion <= 0, calib_arr, confusion)
	Cmask = np.ma.masked_where(new_arr_test <= 0, new_arr_test)
	unique_values = [1.0, 2.0, 3.0, 4.0]

	Map2 = ax1.imshow(Cmask, interpolation='None', cmap=plt.cm.Set1,vmin = np.amin(unique_values), vmax = np.amax(unique_values), alpha = 1.)

	# MR: the error could be that this is Map1 as well instead of Map2
	#Map2 = ax1.imshow(Cmask, interpolation='None', cmap=plt.cm.jet_r, vmin = np.amin(Cmask), vmax = np.amax(Cmask), alpha = 1.)
	#ax1.add_line(road)
	#Map2 = ax1.imshow(Cmask, interpolation='none')
	#unique_values = np.unique(new_arr_test.ravel())
	print(unique_values)
	float_list_values = list(map(float, unique_values))

	unique_values_categories = [ "Post failure", "Pre failure", "At failure", "Calibrated" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = [Map2.cmap(Map2.norm(value)) for value in unique_values]
	# create a patch (proxy artist) for every color
	#MR: need to make a dictionary to relate the values and the timing of the failure
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = list(unique_values_dict.keys())[i])) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)



	# print the proportions of each
	unique, counts = np.unique(new_arr_test, return_counts=True)
	print(dict(zip(unique, counts)))

	#quit()



	#plt.show()


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
		print(calib_arr[i,j])


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
			#valid_arr[y-2:y+2,x-2:x+2] = abs(validated['time_of_failure'].iloc[i] - validated['observed_failtime'].iloc[i]) /(24*3600)


	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	R = ax1.add_line(road)

	calib_valid = np.where(valid_arr < 1, calib_arr, valid_arr)

	calib_valid_mask = np.ma.masked_where(calib_valid <= 0, calib_valid)
	#print(calib_valid_mask)
	#Map2 = ax1.imshow(valid_mask, interpolation='None', cmap=plt.cm.jet_r, vmin = 1, vmax = 4, alpha = 1.)

	#calib_valid = np.where(valid_arr == 0, calib_arr, valid_arr)

	#calib_valid_mask = np.ma.masked_where(calib_valid <= 0., calib_valid)
	#Map1 = ax1.imshow(calib_mask, interpolation='None', cmap=plt.cm.cool, alpha = 1.)

	#calib_valid = np.where(valid_arr <= 0, calib_arr, valid_arr)
	#calib_valid_mask = np.ma.masked_where(calib_valid <= 0, calib_valid)
	####

	unique_values = [1.0, 2.0, 3.0, 4.0]
	print(np.amin(calib_valid_mask))
	print(np.amax(calib_valid_mask))

	Map2 = ax1.imshow(calib_valid_mask, interpolation='None', cmap=plt.cm.jet_r, vmin = np.amin(calib_valid_mask) , vmax=  np.amax(calib_valid_mask), alpha = 1.)

	# MR: the error could be that this is Map1 as well instead of Map2
	#Map2 = ax1.imshow(Cmask, interpolation='None', cmap=plt.cm.jet_r, vmin = np.amin(Cmask), vmax = np.amax(Cmask), alpha = 1.)
	#ax1.add_line(road)
	#Map2 = ax1.imshow(Cmask, interpolation='none')
	#unique_values = np.unique(new_arr_test.ravel())

	print(unique_values)
	float_list_values = list(map(float, unique_values))

	#unique_values_categories = [ "Post failure", "Pre failure", "At failure", "Calibrated" ]
	unique_values_categories = [ "Calibrated", "Pre failure", "Post failure", "At failure" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = [Map2.cmap(Map2.norm(value)) for value in unique_values]
	# create a patch (proxy artist) for every color
	#MR: need to make a dictionary to relate the values and the timing of the failure
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = list(unique_values_dict.keys())[i])) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)



	# print the proportions of each
	unique, counts = np.unique(calib_valid_mask, return_counts=True)
	print(unique, counts)


	plt.tight_layout()
	plt.savefig(fig_name)

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
			#valid_arr[y-2:y+2,x-2:x+2] = abs(validated['time_of_failure'].iloc[i] - validated['observed_failtime'].iloc[i]) /(24*3600)

	print(np.shape(demarr))
	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	R = ax1.add_line(road)
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

	#unique_values_categories = [ "Post failure", "Pre failure", "At failure", "Calibrated" ]
	unique_values_categories = [ "Calibrated", "Pre failure", "Post failure", "At failure" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = ['yellow', 'cyan', 'blue', 'fuchsia']
	# create a patch (proxy artist) for every color
	#MR: need to make a dictionary to relate the values and the timing of the failure
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
			#valid_arr[y-2:y+2,x-2:x+2] = abs(validated['time_of_failure'].iloc[i] - validated['observed_failtime'].iloc[i]) /(24*3600)


	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	R = ax1.add_line(road)
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

	#unique_values_categories = [ "Post failure", "Pre failure", "At failure", "Calibrated" ]
	unique_values_categories = [ "Calibrated", "Pre failure", "Post failure", "At failure" ]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = ['yellow', 'cyan', 'blue', 'fuchsia']
	# create a patch (proxy artist) for every color
	#MR: need to make a dictionary to relate the values and the timing of the failure
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
	#plt.title("Distribution of predicted failures", fontsize = 26, pad = 10.)

	# print the proportions of each
	#unique, counts = np.unique(calib_valid_mask, return_counts=True)
	#print(unique, counts)

	plt.xlim(300,1100)
	plt.ylim(700, 300)
	# image size is 993x1405 px
	# in km this is 24x35 km
	pxl_size = 35/1405
	scalebar = ScaleBar(pxl_size, 'km', font_properties = {"size": 20})
	plt.gca().add_artist(scalebar)
	plt.tight_layout()
	plt.savefig(fig_name)



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
			# if time_of_failure - observed_failtime < 0 : modelled failure detected before observed failure
			# if time_of_failure - observed_failtime > 0 : modelled failure detected after observed failure
			valid_arr[y-2:y+2,x-2:x+2] = (validated['time_of_failure'].iloc[i] - validated['observed_failtime'].iloc[i]) /(24*3600)

	dem_mask = np.ma.masked_where(demarr <= -10, demarr)
	Map1 = ax1.imshow(dem_mask, interpolation='None', cmap=plt.cm.Greys_r, vmin = np.amin(dem_mask), vmax = np.amax(dem_mask), alpha = 1.)

	R = ax1.add_line(road)


	valid_mask = np.ma.masked_where(valid_arr == -0, valid_arr)
	Map2 = ax1.imshow(valid_mask, interpolation='None', norm=DivergingNorm(0), cmap=plt.cm.flag, vmin = np.amin(valid_mask), vmax = np.amax(valid_mask), alpha = 1.)


	calib_mask = np.ma.masked_where(calib_arr == 0., calib_arr)
	Map1 = ax1.imshow(calib_mask, interpolation='None', cmap=plt.cm.cool, vmin = 0, vmax = 1, alpha = 1.)



	unique_values = [1.0]

	float_list_values = list(map(float, unique_values))

	#unique_values_categories = [ "Post failure", "Pre failure", "At failure", "Calibrated" ]
	unique_values_categories = [ "Calibrated"]

	unique_values_dict = OrderedDict(zip(unique_values_categories, unique_values))
	print(unique_values_dict)
	# get the colors of the values, accordiang to the colormap used by imshow
	colors = ['fuchsia']
	# create a patch (proxy artist) for every color
	#MR: need to make a dictionary to relate the values and the timing of the failure
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
	plt.title('Difference between modelled and observed failure times (days)', fontsize = 26, pad = 10.)
	print(np.amin(valid_mask), np.amax(valid_mask))
	norm = mpl.colors.Normalize(vmin=np.amin(valid_mask), vmax=np.amax(valid_mask))
	cmap = plt.cm.flag
	cax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
	cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='proportional')

	#3lt.tight_layout()
	plt.savefig(fig_name)


######################################################
######################################################
# A figure to map validation results
######################################################
######################################################

def plot_rain(rain, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	rainlist = [datetime.datetime(2014, 1, 1)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))

	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])


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

	rainlist = [datetime.datetime(2014, 1, 1)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))

	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])

	for i in range(len(calibrated)):
		time = datetime.datetime(2014, 1, 1) + datetime.timedelta(0,int(calibrated['time_of_failure'].iloc[i]))
		ax2.scatter(time, calibrated['S'].iloc[i])

	ax1.set_xlim(left = datetime.datetime(2014, 1, 1), right = datetime.datetime(2018, 12, 31))
	ax1.set_xlabel("Year", fontsize = 20, labelpad = 10)
	ax1.set_ylabel("Precipitation (mm/day)", fontsize = 20, labelpad = 10 )

	plt.tight_layout()
	plt.savefig(fig_name)

######################################################
######################################################
# A figure to map validation results
######################################################
######################################################

def plot_rain_failures_valid(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax2 =  ax1.twinx()


	# plot the rain
	rainlist = [datetime.datetime(2014, 1, 1)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))
	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])


	# plot the data
	for i in range(len(calibrated)):
		failtime = datetime.datetime(2016, 9, 3) + datetime.timedelta(0,int(calibrated['time_of_failure'].iloc[i]))
		time = datetime.datetime(2016, 9, 3) + datetime.timedelta(0,int(calibrated['insar_failtime'].iloc[i]))
		pretime = datetime.datetime(2016, 9, 3) + datetime.timedelta(0,int(calibrated['insar_prefailtime'].iloc[i]))
		ax2.scatter(time, calibrated['S'].iloc[i], marker = '+', facecolor = 'm')
		ax2.scatter(pretime, calibrated['S'].iloc[i], marker = '_', facecolor = 'm')
		ax2.scatter(failtime, calibrated['S'].iloc[i], marker = '.', facecolor = 'g')


	# plot the model runs
	sbins = np.arange(0,np.amax(slopearr), 0.05)
	#for i,j in product(range(slopearr.shape[0]), range(slopearr.shape[1])):
	for i,j in product(range(400,500,1), range(800,900,1)):

		if failarr[i,j] > 0.:
			print (i,j)

			# this is our slope
			S = slopearr[i,j]
			print (S)

			# find out in which bin it is
			lesser = sbins[np.where(sbins < S)[0][-1]]
			greater = sbins[np.where(sbins >= S)[0][0]]

			#find the points that have been calibrated in this range
			sdf = calibrated[calibrated['S'] >= lesser]
			sdf = sdf[sdf['S'] < greater]

			# If there are indeed points in this category
			if len(sdf)<1:
				print ('woops, there is nothing here')
			else:
				dist = np.sqrt((j-sdf['row'])**2 + (i-sdf['col'])**2)
				where = np.where(np.asarray(dist) == min(np.asarray(dist)))[0]

				select_df = sdf.iloc[where]
				mean_df = select_df.mean(axis = 0)

				mymodel = iverson.iverson_model(alpha = S,
					D_0 = mean_df['D_0'],
					K_sat = mean_df['K_sat'],
					d = mean_df['d'],
					Iz_over_K_steady = mean_df['Iz_over_K_steady'],
          			friction_angle = mean_df['friction_angle'],
          			cohesion = mean_df['cohesion'],
          			weight_of_water = 9800,
          			weight_of_soil = mean_df['weight_of_soil'],
          			depths = depths)

				mymodel.run(rain.duration_s.values, rain.intensity_mm_sec.values)

				failures = mymodel.cppmodel.output_failure_times
				failures = failures[failures > 1.][0]

				time = datetime.datetime(2016, 9, 3) + datetime.timedelta(0,int(failures))
				ax2.scatter(time, S, marker = '.', facecolor = 'r')

	ax1.set_xlim(left = datetime.datetime(2016, 9, 3), right = datetime.datetime(2017, 2, 3))


	plt.tight_layout()
	plt.savefig(fig_name)

##############################################
##############################################
#Making nice violin plots
##############################################
##############################################


def cool_violin_1D(position, data, step, axis, quantiles = [10,25,50,75,90], kerntype = 'gaussian', colour = 'k'):

	# stats
	q_values = []
	for q in quantiles:
		q_values.append(np.percentile(data,q))

	datarange = np.arange(np.amin(data), np.amax(data), step)
	fmt_datarange = datarange[:, np.newaxis]

	kde = KernelDensity(kernel=kerntype, bandwidth=5*step).fit(data)
	density = np.exp(kde.score_samples(fmt_datarange))

	upupper = 0; upper = 0; lower = 0; lolower = 0

	for q in range(len(q_values)):

		lesser = np.where(datarange < q_values[q])[0][-1]
		greater = np.where(datarange > q_values[q])[0][0]

		if q == 0:
			lolower = lesser
			axis.plot([position-density[lesser], position+density[lesser]], [q_values[q], q_values[q]], '--', c = colour, alpha = 0.5)
		if q == 4:
			upupper = greater
			axis.plot([position-density[greater], position+density[greater]], [q_values[q], q_values[q]], '--', c = colour, alpha = 0.5)

		if q == 2:
			axis.plot([position-density[greater], position+density[greater]], [q_values[q], q_values[q]], c = colour, alpha = 0.9, lw = 1.5)
			axis.plot([position-density[lesser], position+density[lesser]], [q_values[q], q_values[q]], c = colour, alpha = 0.9, lw = 1.5)

		if q == 1:	lower = lesser
		if q == 3:	upper = greater

	axis.plot(position+density[lolower-1:upupper+1], datarange[lolower-1:upupper+1], lw = 0.5, c = colour)
	axis.plot(position-density[lolower-1:upupper+1], datarange[lolower-1:upupper+1], lw = 0.5, c = colour)

	axis.plot(position+density[:lolower], datarange[:lolower], lw = 0.5, c = colour, alpha = 0.5)
	axis.plot(position-density[:lolower], datarange[:lolower], lw = 0.5, c = colour, alpha = 0.5)

	axis.plot(position+density[upupper:], datarange[upupper:], lw = 0.5, c = colour, alpha = 0.5)
	axis.plot(position-density[upupper:], datarange[upupper:], lw = 0.5, c = colour, alpha = 0.5)

	axis.fill_betweenx(datarange[lower:upper], position-density[lower:upper], position+density[lower:upper], lw = 0, facecolor = colour, alpha = 0.5)




def density_plot(validated, fig_width, fig_height):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax2 =  ax1.twinx()

	observed_failtime_list = []
	modelled_failtime_list = []
	for i in range(len(validated)):
		observed_failtime = validated['observed_failtime'].iloc[i]/(24*3600)
		observed_failtime_list.append(observed_failtime)

		modelled_failtime = validated['time_of_failure'].iloc[i]/(24*3600) # this is the time since 2014 start date
		modelled_failtime_list.append(modelled_failtime)


	df_failures = pd.DataFrame()
	df_failures['observed_failures'] = observed_failtime_list
	df_failures['modelled_failures'] = modelled_failtime_list
	#ax1 = sns.histplot(modelled_failtime_list, binwidth=50, kde=True, label='Modelled failures')#, x = "Observed failure time (days)"

	#fig, ax = plt.subplots()
	for a in [df_failures['observed_failures'], df_failures['modelled_failures']]:
		ax1= sns.histplot(df_failures['observed_failures'], bins=range(1, int(max(modelled_failtime_list)), 50), ax=ax1, kde=False, color = 'tab:blue', alpha=0.5)
		ax1= sns.histplot(df_failures['modelled_failures'], bins=range(1, int(max(modelled_failtime_list)), 50), ax=ax1, kde=False, color = 'tab:orange', alpha=0.5)

	ax1.set_xlim([0, max(modelled_failtime_list)])
	ax2 = sns.kdeplot(data = df_failures, x = 'observed_failures', label = 'Observed Failures')
	ax2 = sns.kdeplot(data = df_failures, x = 'modelled_failures', label = 'Modelled Failures')

	ax2.set_ylabel('Probability density function', fontsize = 16, labelpad = 10.)
	#ax1.set_ylim([-20,max(rain['time_s']/(3600*24))+20])
	#ax1.set_xlim([0,max(validated['observed_failtime']/(3600*24))+20])
	#ax12.set_ylim([0,max(rain['rainfall_mm'])+5])
	#ax1.tick_params(axis='x', labelsize=16)
	#ax1.tick_params(axis='y', labelsize=16)
	#ax12.tick_params(axis='x', labelsize=16)
	# plt.xlabel('Modelled failure time (days)') # might be good to convert this into a date axis rather than absolute values
	# plt.savefig("modelled_pdf")
	# ax1.clear()
	#ax2 = sns.histplot(observed_failtime_list, binwidth=50,kde = True, label='Observed failures')#, x = "Observed failure time (days)"
	ax1.set_xlabel('Failure Time (days)', fontsize = 16, labelpad = 10.)
	ax1.set_ylabel('Number of failure events', fontsize = 16, labelpad = 10.)
	plt.xlabel('Failure time (days)') # might be good to convert this into a date axis rather than absolute values
	plt.legend()
	plt.legend(loc='upper right',ncol=1,fontsize=14)
	plt.tight_layout()
	plt.savefig("observed_vs_modelled_pdf_update")


	#sns.histplot(df_failures, legend = True)
	# sns.distplot(df_failures['observed_failures'], label='Observed failures')
	# sns.distplot(df_failures['modelled_failures'], label='Modelled failures')
	# plt.xlabel('Failure time (days)')
	# plt.ylabel('Number of failures')
	# plt.legend()
	# plt.savefig("modelled_vs_observed_failtimes")

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

	# df_failures['intervals1'] = np.where(df_failures['modelled_failures']<=150, '150', '0')
	# df_failures['intervals2'] = np.where((df_failures['modelled_failures']>150)&(df_failures['modelled_failures']<=300), '300', '0')
	# df_failures['intervals3'] = np.where((df_failures['modelled_failures']>300)&(df_failures['modelled_failures']<=450), '450', '0')
	# df_failures['intervals4'] = np.where((df_failures['modelled_failures']>450)&(df_failures['modelled_failures']<=600), '600', '0')
	# df_failures['intervals5'] = np.where((df_failures['modelled_failures']>600)&(df_failures['modelled_failures']<=750), '750', '0')
	# df_failures['intervals6'] = np.where((df_failures['modelled_failures']>600), '1000', '0')

def time_split_violin_plot(validated, fig_width, fig_height):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	observed_failtime_list = []
	modelled_failtime_list = []
	for i in range(len(validated)):
		observed_failtime = validated['observed_failtime'].iloc[i]/(24*3600)
		observed_failtime_list.append(observed_failtime)

		modelled_failtime = validated['time_of_failure'].iloc[i]/(24*3600) # this is the time since 2014 start date
		modelled_failtime_list.append(modelled_failtime)


	df_failures = pd.DataFrame()
	df_failures['observed_failures'] = observed_failtime_list
	df_failures['modelled_failures'] = modelled_failtime_list
	print(df_failures.head(5))

	df_failures['time_interval'] = df_failures.apply (lambda row: time_interval(row), axis=1)

	print(df_failures.head(5))

	ax = sns.violinplot(x="time_interval", y="observed_failures", data=df_failures)


	plt.xlabel('Model Failure time (days)') # might be good to convert this into a date axis rather than absolute values

	plt.tight_layout()
	plt.savefig("observed_vs_modelled_violin_plot")


def time_split_violin_plot_new(validated, fig_width, fig_height):
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	observed_failtime_list = []
	modelled_failtime_list = []
	all_failtime_list = np.arange(4802)
	all_failtime_list_types = []

	for i in range(len(validated)):
		observed_failtime = validated['observed_failtime'].iloc[i]/(24*3600)
		observed_failtime_list.append(observed_failtime)

		modelled_failtime = validated['time_of_failure'].iloc[i]/(24*3600) # this is the time since 2014 start date
		modelled_failtime_list.append(modelled_failtime)

	# for i in range(len(all_failtime_list)):
	# 	if i is in observed_failtime_list:
	# 		all_failtime_list_types.append("Obs")
	# 	if i is in modelled_failtime_list:
	# 		all_failtime_list_types.append("Mod")


	df_failures = pd.DataFrame()
	df_failures['observed_failures'] = observed_failtime_list
	df_failures['modelled_failures'] = modelled_failtime_list
	df_failures["times"] = 0
	df.loc[(df['observed_failures'] > 0) & (df['observed_failures'] <= 10), 'times'] = 'xxx'
	print(df_failures.head(5))

	df_failures['time_interval'] = df_failures.apply (lambda row: time_interval(row), axis=1)

	print(df_failures.head(5))

	ax = sns.violinplot(x="time_interval", y="observed_failures", data=df_failures)


	plt.xlabel('Model Failure time (days)') # might be good to convert this into a date axis rather than absolute values

	plt.tight_layout()
	plt.savefig("observed_vs_modelled_violin_plot_new")




#modelled_failtime = validated['time_of_failure'].iloc[i]/(24*3600)

##############################################
##############################################
# Running and plotting a PCA on calibrated data
##############################################
##############################################

def plot_sensitivity(rain, calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

	# plot the rain
	rainlist = [0]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']



	cols = calibrated.columns.values[1:].tolist()
	cols.remove('row')
	cols.remove('col')
	cols.remove('weight_of_water')
	cols.remove('time_of_failure')
	cols.remove('insar_failtime')
	cols.remove('insar_prefailtime')
	cols.remove('S')
	cols.remove('Z')

	X = np.asarray(calibrated[cols])
	y = np.asarray(calibrated['time_of_failure'])


	print (X.shape[1])



	from SALib.sample import saltelli
	from SALib.analyze import sobol

	boundaries = []
	for i in range(X.shape[1]):
		boundaries.append([min(X[:,i]), max(X[:,i])])


	problem = {'num_vars': X.shape[1]+1,
	           'names': cols,
	           'bounds': boundaries
	           }

	print(problem)
	# Perform analysis
	Si = sobol.analyze(problem, y, print_to_console=False)

	S1 = Si['S1']

	axis = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	print (cols, len(cols))
	print (S1, len(S1))

	axis.bar(cols, S1)
	#axis.set_yscale('log')
	axis.set_ylabel('1st order sensitivity')

	plt.savefig(fig_name)

	quit()


	from sklearn import decomposition

	pca = decomposition.PCA(n_components=X.shape[1])
	pca.fit(X)
	X = pca.transform(X)

	Xe = pca.explained_variance_ratio_


	axis = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	axis.bar(cols, Xe)
	axis.set_yscale('log')
	axis.set_ylabel('Percentage explained variance')



	plt.savefig(fig_name)






def plot_rain_parameters_correlation(rain, calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

	axis = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)



	# plot the rain
	rainlist = [0]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ int(rain['duration_s'].iloc[i]))
	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	for i in range(len(calibrated)):
		failtime = calibrated['time_of_failure'].iloc[i]
		if failtime > 0:

			print ()
			print ('index is:', i)
			print (failtime)
			before = np.where(rain['time'] < failtime)[0][-1]

			print (before)
			print (rain['rainfall_mm'].iloc[before])

			if before < 3:
				maxrain = max(rain['rainfall_mm'].iloc[:before])
			else:
				tempmax = 0.
				for j in range(len(rain.iloc[:before])):
					print()
					print (j)

					print (  rain['rainfall_mm'].iloc[before-3-j:before-j]  )

					if len( rain['rainfall_mm'].iloc[before-3-j:before-j] ) >= 1:
						newmax = max( rain['rainfall_mm'].iloc[before-3-j:before-j] )
					else:
						newmax = 0.

					if newmax < tempmax:
						maxrain = tempmax
						break
					else:
						tempmax = newmax

			axis.scatter(calibrated['alxpha'].iloc[i], maxrain)







	plt.show()
	quit()



	cols = calibrated.columns.values[1:].tolist()
	cols.remove('row')
	cols.remove('col')
	cols.remove('weight_of_water')
	cols.remove('time_of_failure')
	cols.remove('insar_failtime')
	cols.remove('insar_prefailtime')
	cols.remove('S')
	cols.remove('Z')

	X = np.asarray(calibrated[cols])
	y = np.asarray(calibrated['time_of_failure'])


	print (X.shape[1])



	from SALib.sample import saltelli
	from SALib.analyze import sobol

	boundaries = []
	for i in range(X.shape[1]):
		boundaries.append([min(X[:,i]), max(X[:,i])])


	problem = {'num_vars': X.shape[1]+1,
	           'names': cols,
	           'bounds': boundaries
	           }

	print(problem)
	# Perform analysis
	Si = sobol.analyze(problem, y, print_to_console=False)

	S1 = Si['S1']

	axis = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	print (cols, len(cols))
	print (S1, len(S1))

	axis.bar(cols, S1)
	#axis.set_yscale('log')
	axis.set_ylabel('1st order sensitivity')

	plt.savefig(fig_name)

	quit()
