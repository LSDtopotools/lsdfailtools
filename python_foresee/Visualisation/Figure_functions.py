# I'll need that to process the outputs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from itertools import product
import  datetime
import pandas as pd
import numpy as np
import shapefile
import itertools


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
	Map1 = ax1.imshow(calib_mask, interpolation='None', cmap=plt.cm.jet,
	    vmin = 0, vmax = 1, alpha = 1.)

	plt.tight_layout()
	plt.savefig(fig_name)




######################################################
######################################################
# A figure to map calibrated points
######################################################
######################################################
def plot_failtime (calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	ax1.set_xlabel('time (days)')

	late_failtimes = sorted(list(set(list(calibrated['insar_failtime']))))

	for i in range(len(late_failtimes)):
		df = calibrated[calibrated['insar_failtime'] == late_failtimes[i]]
		df['time_of_failure'] = df['time_of_failure']/(24*3600)

		ax1.hist(df['time_of_failure'], bins = 20, color = plt.cm.jet(i/len(late_failtimes)), lw = 0, density = True, alpha = 0.7)

	plt.tight_layout()
	plt.savefig(fig_name)


######################################################
######################################################
# A figure to map calibrated points
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
def map_validation(rain, depths, calibrated, demarr, slopearr, failarr, prefailarr, fig_height, fig_width, fig_name):

	confusion = 0*np.copy(slopearr)

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	# how do you select the parameters?
	# based on location and slope

	# make slope bins in the calibrated df
	#shist, sbins = np.histogram(calibrated['S'], bins  = 10)
	sbins = np.arange(0,np.amax(slopearr), 0.05)

	#for i,j in product(range(slopearr.shape[0]), range(slopearr.shape[1])):
	for i,j in product(range(400,410,1), range(800,810,1)):

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

				error = 5 * 24*3600

				if failures <= failarr[i,j] + error and failures >= prefailarr[i,j] - error:
					confusion[i,j] = 3 # success
					confusion[i-2:i+2,j-2:j+2] = 3 # success
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


	# show the calibrated points
	for i in range(len(calibrated)):
		confusion[calibrated['row'].iloc[i], calibrated['row'].iloc[i]] = 4


	Cmask = np.ma.masked_where(confusion <= 0, confusion)
	# MR: the error could be that this is Map1 as well instead of Map2
	Map2 = ax1.imshow(Cmask, interpolation='None', cmap=plt.cm.jet_r, vmin = np.amin(Cmask), vmax = np.amax(Cmask), alpha = 1.)

	unique_values = np.unique(Cmask.ravel())
	unique_values_categories = ["Calibrated", "Pre failure", "At failure", "Post failure" ]
	unique_values_dict = dict(zip(unique_values_categories, unique_values))

	# get the colors of the values, accordiang to the colormap used by imshow
	colors = [Map2.cmap(Map2.norm(value)) for value in unique_values]
	# create a patch (proxy artist) for every color
	#MR: need to make a dictionary to relate the values and the timing of the failure
	patches = [mpatches.Patch(color=colors[i], label="{l}".format(l = unique_values_dict.keys())) for i in range(len(unique_values)) ]
	#put those patches as legend-handles into the legend
	plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)



	# print the proportions of each
	unique, counts = np.unique(confusion, return_counts=True)
	print(dict(zip(unique, counts)))

	#quit()



	#plt.show()


	plt.tight_layout()
	plt.savefig(fig_name)


######################################################
######################################################
# A figure to map validation results
######################################################
######################################################

def plot_rain(rain, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	rainlist = [datetime.datetime(2016, 9, 3)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))

	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])


	plt.tight_layout()
	plt.savefig(fig_name)

######################################################
######################################################
# A figure to map validation results
######################################################
######################################################

def plot_rain_failures(rain, calibrated, fig_height, fig_width, fig_name):

	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax2 =  ax1.twinx()

	rainlist = [datetime.datetime(2016, 9, 3)]
	for i in range(1,len(rain)):
		rainlist.append(rainlist[-1]+ datetime.timedelta(0,int(rain['duration_s'].iloc[i]), 0))

	rain['time'] = rainlist
	rain['rainfall_mm'] = rain['duration_s']*rain['intensity_mm_sec']

	ax1.plot(rain['time'], rain['rainfall_mm'])

	for i in range(len(calibrated)):
		time = datetime.datetime(2016, 9, 3) + datetime.timedelta(0,int(calibrated['time_of_failure'].iloc[i]))
		ax2.scatter(time, calibrated['S'].iloc[i])

	ax1.set_xlim(left = datetime.datetime(2016, 9, 3), right = datetime.datetime(2017, 12, 3))


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
	rainlist = [datetime.datetime(2016, 9, 3)]
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
