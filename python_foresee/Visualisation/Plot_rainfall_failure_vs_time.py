import os
import sys
import csv
import json
import numpy
import datetime
import pandas as bb
from itertools import product
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr

import sys
sys.path.insert(0,'../Alldata_processing/InSAR')

import Insar_functions as fn


nodata_value = 999.

with open("../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


# Model directory
rundir = FILE_PATHS["run_dir"]

fig_out_dir = FILE_PATHS["figures_dir"]

# Prepare rainfall data
rainfile = FILE_PATHS["rain_dir"] + "2014-01-01_to_2019-12-31_Intensity.csv"
raindata = bb.read_csv(rainfile)

rainfile_file = rainfile.split('/')[-1]

raindata_start = rainfile_file.split('_')[0]
raindata_start = datetime.datetime.strptime (raindata_start, '%Y-%m-%d')

raintimes_list = [raindata_start]
for i in range(1,len(raindata)):
	dt = raindata['duration_s'].iloc[i]
	raintimes_list.append(raintimes_list[-1]+datetime.timedelta(0,int(dt),0))

raindata['DATE'] = raintimes_list
raindata['Rainfall_mm'] = raindata['duration_s'] * raindata['intensity_mm_sec']



# Read slope data
slopefile = FILE_PATHS["topo_dir"] + "eu_dem_AoI_epsg32633_SLOPE.bil"
slope, post, geoinfo = fn.ENVI_raster_binary_to_2d_array(slopefile)

# Prepare failure data
faildir = FILE_PATHS["interferometry_out_dir"]
failtypes = ["A", "D", "EWV"]
failcolours = ['r', 'g', 'k']
# MR: not sure what these start dates are??
startdates = [datetime.datetime(2016,11,4), datetime.datetime(2016,9,3), datetime.datetime(2016,11,4)]

# thresholds = [40, 60, 80, 100, 150, 200, 500, 1000]
# thresholds = [80, 100, 150, 200, 500, 1000]
thresholds = [500]



for i in range(len(thresholds)):



	for j in range(len(failtypes)):

		plt.figure(figsize=(8, 8))
		ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
		ax2 =  ax1.twinx()

		ax2.fill_between(raindata['DATE'], 0, raindata['Rainfall_mm'], alpha = 0.8, lw = 0)

		F1, post, geoinfo = fn.ENVI_raster_binary_to_2d_array(faildir+failtypes[j]+"_failtime_1_threshold"+str(thresholds[i])+"mmyr.bil")
		F2, post, geoinfo = fn.ENVI_raster_binary_to_2d_array(faildir+failtypes[j]+"_failtime_2_threshold"+str(thresholds[i])+"mmyr.bil")
		F3, post, geoinfo = fn.ENVI_raster_binary_to_2d_array(faildir+failtypes[j]+"_failtime_3_threshold"+str(thresholds[i])+"mmyr.bil")

		startdate = startdates[j]


		F1_cut = F1[F1 > 0].ravel()
		slope_F1 = slope[F1 > 0].ravel()

		F2_cut = F2[F2 > 0].ravel()
		slope_F2 = slope[F2 > 0].ravel()

		F3_cut = F3[F3 > 0].ravel()
		slope_F3 = slope[F3 > 0].ravel()


		for k in range(len(F1_cut)):
			print (k, '/', len(F1_cut))
			if k == 0:
				ax1.scatter(startdates[j] + datetime.timedelta(0,int(F1_cut[k])), slope_F1[k], marker = '$1$', alpha = 0.6, facecolor = failcolours[j], label = failtypes[j])
			else:
				ax1.scatter(startdates[j] + datetime.timedelta(0,int(F1_cut[k])), slope_F1[k], marker = '$1$', alpha = 0.6, facecolor = failcolours[j])

		for k in range(len(F2_cut)):
			print (k, '/', len(F2_cut))
			if k == 0:
				ax1.scatter(startdates[j] + datetime.timedelta(0,int(F2_cut[k])), slope_F2[k], marker = '$2$', alpha = 0.6, facecolor = failcolours[j], label = failtypes[j])
			else:
				ax1.scatter(startdates[j] + datetime.timedelta(0,int(F2_cut[k])), slope_F2[k], marker = '$2$', alpha = 0.6, facecolor = failcolours[j])


		for k in range(len(F3_cut)):
			print (k, '/', len(F3_cut))
			if k == 0:
				ax1.scatter(startdates[j] + datetime.timedelta(0,int(F3_cut[k])), slope_F3[k], marker = '$3$', alpha = 0.6, facecolor = failcolours[j], label = failtypes[j])
			else:
				ax1.scatter(startdates[j] + datetime.timedelta(0,int(F3_cut[k])), slope_F3[k], marker = '$3$', alpha = 0.6, facecolor = failcolours[j])

	ax2.fill_between(raindata['DATE'], 0, raindata['Rainfall_mm'], facecolor = plt.cm.Blues(150), alpha = 0.6, lw = 0)


	ax1.legend(loc = 1)


	ax2.set_ylim(bottom = 0)
	ax1.set_xlim(left = datetime.datetime(2016,10,1), right = datetime.datetime(2019,1,31))

	ax1.set_ylabel('DEM slope (m/m)')
	ax2.set_ylabel('Daily rainfall (mm)')

	plt.savefig(fig_out_dir+failtypes[j]+'_Failtimes_threshold'+str(thresholds[i])+'_slope_rainfall_MR.png')




quit()

# MR: not sure what the next bit dones - need to investigate

for a in range(len(cols)):

    if twin[a] is False:
        axes[a].set_title(title[a])

    cluster_range = np.arange(0,max(clusters)+0, 1)
    c_means = []

    for c in cluster_range:
        c_df = df_classify.loc[df_classify['clusters'] == c+1]
        c_arr = np.asarray(c_df[cols[a]])
        c_mean = np.mean(c_arr); c_means.append(c_mean)
        c_std = np.std(c_arr)
        if twin[a] is False:
            axes[a].scatter(c,c_mean, s = 80, marker = 'o', lw = 0)
        else:
            axes[a].scatter(c,c_mean, s = 80, marker = '*', edgecolor = 'k', lw = 0.4)
        axes[a].plot([c,c],[c_mean-c_std, c_mean+c_std], '-k', lw = 0.2)

    # plot limits
    if min(c_means) > 0:
        axes[a].set_ylim(bottom = 0, top = 1.1*max(c_means))
    elif max(c_means) < 0:
        axes[a].set_ylim(top = 1.1*min(c_means), bottom = 0)


plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=1.2)
plt.savefig(fig_out_dir + 'Figure1_500.png')
