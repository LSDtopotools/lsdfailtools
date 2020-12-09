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

gcs_path = "/home/willgoodwin/Software/anaconda3/envs/coastalsat/lib/python3.7/site-packages/fiona/gdal_data/"


################################################################################
################################################################################
"""Import internal modules"""
################################################################################
################################################################################

#GPM MONTH
from gpm_download_month_V06B import gpm_month_download
#GPM DAY
from gpm_download_day_V06B import gpm_day_download
#GPM 30min
from gpm_download_30min_V06B import gpm_30min_download

#AncillaryData
from image_process import process
from get_info import get_info
import General_functions as fn


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



def maps_to_timeseries(arglist, working_dir):
	# List the .bil files
	print('Got to the timeseries function')
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
		print (bilfiles[i])
		timer = datetime.datetime.strptime(bilfiles[i], 'Calib_rainfall_%Y%m%d-S%H%M%S-V06B_cut.bil')

		# Add to the rainfall list
		Rainarr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_2d_array(working_dir + bilfiles[i])
		Rain = numpy.mean(Rainarr)
		Intensity = Rain/(30*60) # Intensity of rainfall during the period (mm/sec)
		#Intlist.append(Intensity)

		# Save it all together
		Full_list.append([30*60, Intensity])

	# Now save the stuff
	with open(working_dir+arglist[1]+"_to_"+arglist[2]+"_Intensity.csv", "w", newline="") as f:
	    writer = csv.writer(f)
	    writer.writerow(['duration_s','intensity_mm_sec'])
	    writer.writerows(Full_list)
	print ('DOOOOONE')
	print (working_dir)




################################################################################
################################################################################

def download_months(arglist, zero_list, zero_dir, fst_dir, backslh):
	if zero_list[n].endswith('.HDF5') > -1 and zero_list[n].find('.xml') == -1 and zero_list[n].find('.aux') == -1 and zero_list[n].find('.tfw') == -1:
			if 	zero_list[n].find('.HDF5') > -1:
				extract_subdata = 'HDF5:"%s%s%s"://Grid/precipitation' % (zero_dir,backslh,zero_list[n])
				outfile = '%s%s%s.tif' % (fst_dir,backslh,zero_list[n][:-5])

				process(outfile,extract_subdata,arglist[0])
				raster_crop(arglist, outfile)
				extract_subdata = outfile = None


def download_days(arglist, zero_list, zero_dir, fst_dir, backslh):
	if zero_list[n].endswith('.nc4') > -1 and zero_list[n].find('.xml') == -1 and zero_list[n].find('.aux') == -1 and zero_list[n].find('.tfw') == -1:
			extract_subdata = 'HDF5:"%s%s%s"://precipitationCal' % (zero_dir, backslh, zero_list[n])
			outfile = '%s%s%s_precipitationCal.tif' % (fst_dir,backslh, zero_list[n][:-4])
			
			process(outfile,extract_subdata,arglist[0])
			raster_crop(arglist, outfile)
			extract_subdata = outfile = None



def download_hhs(arglist, zero_list, zero_dir, fst_dir, backslh, n):
	if zero_list[n].endswith('.HDF5') > -1 and zero_list[n].find('.xml') == -1 and zero_list[n].find('.aux') == -1 and zero_list[n].find('.tfw') == -1:
		if 	zero_list[n].find('.HDF5') > -1:
			extract_subdata = "%s%s%s" % (zero_dir,backslh,zero_list[n])
			#extract_subdata = 'HDF5:"%s%s%s"://Grid/precipitation' % (zero_dir,backslh,zero_list[n])
			outfile = '%s%s%s.bil' % (fst_dir,backslh,zero_list[n][:-5])

			process(outfile,extract_subdata,arglist[0])
			raster_crop(arglist, outfile)
			extract_subdata = outfile = None



########
# Crop the raster
########
def raster_crop(arglist, outfile):
	if arglist[4] != 'None':
		cutfile = arglist[4]
		to_cut = outfile
		cutted_file = outfile[:-4] + '_cut.bil'
		# Give it a nice and easy name
		A = cutted_file.split('.')
		AA = A[0].split('/')[:-1]
		AAA = '/'.join(AA)+'/'
		B = A[4][:-7]
		BB = A[6]

		cutted_file = AAA+'Calib_rainfall_' + B + BB + '.bil'

		# Cut the raster to your desired extent
		os.system('gdalwarp -overwrite -of ENVI -t_srs EPSG:4326 -cutline ' + cutfile + ' -crop_to_cutline ' + to_cut + ' ' + cutted_file)
		#os.system('gdalwarp --config GDAL_DATA ' + gcs_path + ' -overwrite -of ENVI -t_srs EPSG:4326 -cutline ' + cutfile + ' -crop_to_cutline ' + to_cut + ' ' + cutted_file)
		# Get rid of the big files
		os.system('rm ' + outfile)
		os.system('rm ' + cutted_file+'.aux.xml')
		os.system('rm ' + outfile[:-4] + '.hdr')
