import os
import sys
import json
import numpy
import datetime
from osgeo import gdal, ogr, osr
import csv
import matplotlib.pyplot as plt
import pandas as bb


nodata_value = 999.

# python PPT_CMD_RUN.py --ProdTP GPM_D --StartDate 2014-01-01 --EndDate 2019-12-31 --ProcessDir /home/willgoodwin/PostDoc/Foresee/Data/Precipitation/GPM_data --SptSlc /home/willgoodwin/PostDoc/Foresee/Data/Topography/eu_dem_v11_E40N20_AoI.bil --DirOut /home/willgoodwin/PostDoc/Foresee/Data/Precipitation/GPM_data/


with open("../../../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


piezofile = FILE_PATHS["piezo_dir"] + "data_piezometer.csv"
piezodata = bb.read_csv(piezofile)

piezodata['DATE'] =  bb.to_datetime(piezodata['DATE'], format='%d/%m/%Y')




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




for i in range(1,max(piezodata['ID']+1)):

	df = piezodata[piezodata['ID'] == i]

	plt.figure(figsize=(8, 8))
	ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
	ax2 =  ax1.twinx()

	start = df['DATE'].iloc[0]-datetime.timedelta(10,0,0)
	stop = df['DATE'].iloc[-1]+datetime.timedelta(10,0,0)

	ax2.fill_between(raindata['DATE'], 0, raindata['Rainfall_mm'], alpha = 0.8, lw = 0)

	ax1.plot([start, stop],[0,0], '-k', lw = 2)


	for header in ["LIV1","LIV2","LIV3","LIV4"]:
		df = df[df[header] != nodata_value]
		ax1.plot(df['DATE'], -df[header], '+-', label = header)
	ax1.legend(loc = 1)

	ax1.set_xlim(start, stop)
	ax2.set_ylim(bottom = 0)

	ax1.set_ylabel('Groundwater depth (m)')
	ax2.set_ylabel('Daily rainfall (mm)')

	plt.savefig('Figures/Piezo'+str(i)+'_GW_rainfall.png')

quit()



quit()



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
plt.savefig(dir + 'Figure1.png')
