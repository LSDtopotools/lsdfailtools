################################################################################
################################################################################
"""Import Python packages"""
################################################################################
################################################################################

import os
import sys
import argparse
import time
import shutil
import re
import numpy
import tkinter
from tkinter import filedialog
import platform
import argparse
import datetime
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import csv
import matplotlib.pyplot as plt
import numpy as np



################################################################################
################################################################################
"""Import internal modules"""
################################################################################
################################################################################

#from gpm_download_month_V06B import gpm_month_download
#from gpm_download_day_V06B import gpm_day_download
#from gpm_download_30min_V06B import gpm_30min_download

#AncillaryData
#from image_process import process_HDF5, process_nc4
#from get_info import get_info


################################################################################
################################################################################
"""Argument Parser"""
################################################################################
################################################################################

def parseArguments():

	parser = argparse.ArgumentParser(prog='Precipitation Processing Tool')

	parser.add_argument('--ProdTP', choices= ['GPM_M','GPM_D','GPM_30min'], default='GPM_30min', dest='ProdTP',  help='GPM_M: GPM Monthly (IMERGM v6);\n GPM_D: GPM Daily (IMERGDF v6); \n GPM_30min: GPM Half-hourly (IMERGHHE v6)\n')

	StartDF = '01-06-2000'
	parser.add_argument('--StartDate',dest='StartDate', help='Insert the start date',default=StartDF,type=str)

	EndDF = str((datetime.datetime.now()).strftime('%Y-%m-%d'))
	parser.add_argument('--EndDate',dest='EndDate', help='Insert the end date',default=EndDF,type=str)

	parser.add_argument('--ProcessDir',dest='ProcessDir', help='Insert the processing directory path',type=str)

	parser.add_argument('--SptSlc',dest='SptSlc', nargs="?", help='Insert the slice feature path',type=str)

	parser.add_argument('--OP', dest='OP',action="store_true", help='Call this argument if you only want to process the data. Make sure you have a directory with a raw files subfolder.')

	parser.add_argument('--DirOut', dest='DirOut', help='This is the output directory for the final timeseries file. Defaults to current working directory.', default = os.getcwd(),type=str)

	args = parser.parse_args();
	return args


################################################################################
################################################################################
def ENVI_raster_binary_to_2d_array(file_name):
    """
    This function transforms a raster into a numpy array.

    Args:
        file_name (ENVI raster): the raster you want to work on.
        gauge (string): a name for your file

    Returns:
        image_array (2-D numpy array): the array corresponding to the raster you loaded
        pixelWidth (geotransform, inDs) (float): the size of the pixel corresponding to an element in the output array.

    Source: http://chris35wills.github.io/python-gdal-raster-io/
    """


    driver = gdal.GetDriverByName('ENVI')

    driver.Register()

    inDs = gdal.Open(file_name, GA_ReadOnly)

    if inDs is None:
        print ("Couldn't open this file: " + file_name)
        print ("Perhaps you need an ENVI .hdr file? ")
        sys.exit("Try again!")
    else:
        print ("%s opened successfully" %file_name)

        #print '~~~~~~~~~~~~~~'
        #print 'Get image size'
        #print '~~~~~~~~~~~~~~'
        cols = inDs.RasterXSize
        rows = inDs.RasterYSize
        bands = inDs.RasterCount

        #print "columns: %i" %cols
        #print "rows: %i" %rows
        #print "bands: %i" %bands

        #print '~~~~~~~~~~~~~~'
        #print 'Get georeference information'
        #print '~~~~~~~~~~~~~~'
        geotransform = inDs.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]

        #print "origin x: %i" %originX
        #print "origin y: %i" %originY
        #print "width: %2.2f" %pixelWidth
        #print "height: %2.2f" %pixelHeight

        # Set pixel offset.....
        #print '~~~~~~~~~~~~~~'
        #print 'Convert image to 2D array'
        #print '~~~~~~~~~~~~~~'
        band = inDs.GetRasterBand(1)
        #print band
        image_array = band.ReadAsArray(0, 0, cols, rows)
        image_array_name = file_name
        #print type(image_array)
        #print image_array.shape

        return image_array, pixelWidth, (geotransform, inDs)




################################################################################
################################################################################
def ENVI_raster_binary_from_2d_array(envidata, file_out, post, image_array):
    """
    This function transforms a numpy array into a raster.

    Args:
        envidata: the geospatial data needed to create your raster
        file_out (string): the name of the output file
        post: coordinates for the goegraphical transformation
        image_array (2-D numpy array): the input raster

    Returns:
        new_geotransform
        new_projection: the projection in which the raster
        file_out (ENVI raster): the raster you wanted

    Source: http://chris35wills.github.io/python-gdal-raster-io/
    """

    driver = gdal.GetDriverByName('ENVI')

    original_geotransform, inDs = envidata

    #print 'WOOO'
    #print envidata
    #print original_geotransform
    #print inDs
    #print inDs.GetProjection()

    rows, cols = image_array.shape
    bands = 1

    # Creates a new raster data source
    outDs = driver.Create(file_out, cols, rows, bands, gdal.GDT_Float32)

    # Write metadata
    originX = original_geotransform[0]
    originY = original_geotransform[3]

    outDs.SetGeoTransform([originX, post, 0.0, originY, 0.0, -post])
    outDs.SetProjection(inDs.GetProjection())

    #Write raster datasets
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(image_array)

    new_geotransform = outDs.GetGeoTransform()
    new_projection = outDs.GetProjection()

    print ("Output binary saved: ", file_out)
    return new_geotransform,new_projection,file_out



######################################################################
######################################################################

def maps_to_timeseries(working_dir, arglist, output_dir):
	# List the .bil files
	files = sorted(os.listdir(working_dir)); bilfiles = []
	for i in range(len(files)):
		ending = files[i][-4:]
		if ending == '.bil':
			bilfiles.append(files[i])


	#Timelist = []
	#CumTimelist = []
	#Intlist = []
	Full_list = []
	for i in range(len(bilfiles)):

		print (arglist)

		if arglist[0] == 'GPM_D':
			timer = datetime.datetime.strptime(bilfiles[i], '3B-DAY_%Y%m%d_S%H%M%S_V06_precipitationCal.bil')
		elif arglist[0] == 'GPM_30min':
			timer = datetime.datetime.strptime(bilfiles[i], '3B-HHR-E_%Y%m%d_S%H%M%S_V06B_precipitationCal.bil')



		# Add to the rainfall list
		Rainarr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_2d_array(working_dir + bilfiles[i])
		Rain = numpy.mean(Rainarr)
		Intensity = Rain/(30*60) # Intensity of rainfall during the period (mm/sec)
		#Intlist.append(Intensity)

		# Save it all together
		Full_list.append([30*60, Intensity])

	# Now save the stuff
	with open(output_dir+'/'+arglist[1]+"_to_"+arglist[2]+"_Intensity.csv", "w", newline="") as f:
	    writer = csv.writer(f)
	    writer.writerow(['duration_s','intensity_mm_sec'])
	    writer.writerows(Full_list)
	print ('saved file:', output_dir+arglist[1]+"_to_"+arglist[2]+"_Intensity.csv")




################################################################################
################################################################################

def plot_dva_rainfall(i, rain, dates, disp, av10_disp, inst_vel, inst_vel_av10, inst_acc, inst_acc_av10, S_startdate, S_enddate, Tfail, Tprefail, th):
    fig=plt.figure(1, facecolor='White',figsize=[7, 7])
    ax1 =  plt.subplot2grid((3,1),(0,0),colspan=1, rowspan=1)
    ax2 =  plt.subplot2grid((3,1),(1,0),colspan=1, rowspan=1)
    ax3 =  plt.subplot2grid((3,1),(2,0),colspan=1, rowspan=1)

    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    ax33 = ax3.twinx()

    #plot the rain
    ax11.fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)
    ax22.fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)
    ax33.fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)


    ax1.plot(dates, disp, '--', c = plt.cm.jet(0), lw = 0.5)
    ax1.plot(dates[5:-5], av10_disp, c = plt.cm.jet(0), lw = 1)
    
    ax2.plot(dates[1:], inst_vel/1000., '--', c = plt.cm.jet(0), lw = 0.5)
    ax2.plot(dates[6:-5], inst_vel_av10/1000., c = plt.cm.jet(0), lw = 1)
    
    ax3.plot(dates[2:], inst_acc/(1000.), '--', c = plt.cm.jet(0), lw = 0.5)
    ax3.plot(dates[7:-5], inst_acc_av10/(1000.), c = plt.cm.jet(0), lw = 1)

    for j in range(len(Tfail)):
        time_p = datetime.timedelta(0,Tprefail[j], 0)
        time_f = datetime.timedelta(0,Tfail[j], 0)
        ax3.plot([S_startdate+time_p, S_startdate+time_f], [th,th], c = plt.cm.jet(255), lw = 2)


    ax1.set_xlim(left = S_startdate, right = S_enddate)
    ax2.set_xlim(left = S_startdate, right = S_enddate)
    ax3.set_xlim(left = S_startdate, right = S_enddate)

    ax3.set_xlabel('Time (yr)')
    ax11.set_ylabel('Precipitation (mm)')
    ax22.set_ylabel('Precipitation (mm)')
    ax33.set_ylabel('Precipitation (mm)')

    ax1.set_ylabel('Displacement (mm)')
    ax2.set_ylabel('Abs. velocity (m/yr)')
    ax3.set_ylabel('Displacement acceleration (m/yr2)')

    plt.savefig('/home/willgoodwin/PostDoc/Foresee/Figures/Sentinel_failure/'+str(i)+'.png')
