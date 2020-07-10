###
# MR: this has been checked for issues with imports. Runs OK
###

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
import pandas as pd
import numpy as np

# set the directory to import the functions from
import sys
sys.path.insert(0,'../../lsdfailtools-master/lsdfailtools')

from lsdfailtools import iverson2000 as iverson



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
        post: coordinates for the geographical transformation
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

def process_months(arglist, zero_list, zero_dir, fst_dir, backslh):
	if zero_list.endswith('.HDF5') > -1 and zero_list.find('.xml') == -1 and zero_list.find('.aux') == -1 and zero_list.find('.tfw') == -1:
			if 	zero_list.find('.HDF5') > -1:
				extract_subdata = 'HDF5:"%s%s%s"://Grid/precipitation' % (zero_dir,backslh,zero_list)
				outfile = '%s%s%s.tif' % (fst_dir,backslh,zero_list[:-5])

				process_HDF5(outfile,extract_subdata,arglist)
				extract_subdata = outfile = None


def process_days(arglist, zero_list, zero_dir, fst_dir, backslh):
	if zero_list.endswith('.nc4') > -1 and zero_list.find('.xml') == -1 and zero_list.find('.aux') == -1 and zero_list.find('.tfw') == -1:
			extract_subdata = '%s%s%s' % (zero_dir, backslh, zero_list)
			outfile = '%s%s%s_precipitationCal.bil' % (fst_dir,backslh, zero_list[:-4])

			process_nc4(outfile,extract_subdata,arglist)
			extract_subdata = outfile = None



def process_hhs(arglist, zero_list, zero_dir, fst_dir, backslh):
	if zero_list.endswith('.HDF5') > -1 and zero_list.find('.xml') == -1 and zero_list.find('.aux') == -1 and zero_list.find('.tfw') == -1:
			if 	zero_list.find('.HDF5') > -1:
				extract_subdata = "%s%s%s" % (zero_dir,backslh,zero_list)
				outfile = '%s%s%s_precipitationCal.bil' % (fst_dir,backslh,zero_list[:-5])

				process_HDF5(outfile,extract_subdata,arglist)
				extract_subdata = outfile = None


################################################################################
################################################################################
def storage_df(results, selected, S, Z, i, j, F, P):
    work_df = results.iloc[selected]
    work_df['S'] = S
    work_df['Z'] = Z
    work_df['row'] = i
    work_df['col'] = j
    work_df['insar_failtime'] = F
    work_df['insar_prefailtime'] = P

    return work_df



################################################################################
################################################################################
def assess_fitness (results, F, P, Nruns):
    # extract the failtimes
        failtimes = np.asarray(results['time_of_failure'])

        # calculate the time differences
        notlater = F - failtimes
        notsooner = failtimes - P

        # Which failtimes are within observed times?
        inbounds_ID = np.where(np.logical_and(notlater > 0, notsooner>0))[0]

        # if not all the runs are "sucessfully calibrated"
        if len(inbounds_ID) < Nruns:

            # Where is the closest "out of bounds" point?
            distance = abs(np.min([notlater, notsooner], axis = 0))
            if len(inbounds_ID) >= 1:
                distance[inbounds_ID] = 10**10
            closest_ID = np.where(distance == min(distance))[0]

            # merge the inbounds and closest outbouds indices to select "fit" points
            selected = list(inbounds_ID)+list(closest_ID)

            # use the extreme values of parameters of these points to define parameter ranges for the next MC. Don't forget to pad on either side
        else:
            selected = list(inbounds_ID)

        return selected


################################################################################
################################################################################
def MC_initial (rain, S, depths, Nruns, rundir):

    # set the parameters for initial the MC runs
        MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.95*S, D_0_min = 1e-6,K_sat_min = 1e-8, d_min = 0.5, Iz_over_K_steady_min = 0.1, friction_angle_min = 0.2, cohesion_min = 5000, weight_of_water_min = 9800, weight_of_soil_min = 15000,
            alpha_max = 1.05*S, D_0_max = 1e-4,K_sat_max = 1e-6, d_max = 3,Iz_over_K_steady_max = 0.8, friction_angle_max = 0.5, cohesion_max = 20000, weight_of_water_max = 9801, weight_of_soil_max = 25000, depths = depths)

        # Now run it
        MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                          n_process = 2, output_name = "test_MC.csv", n_iterations = Nruns, replace = True)

        # now open the MC test file
        results = pd.read_csv(rundir+"test_MC.csv")

        return results



################################################################################
################################################################################
def MC_assisted(rain, S, depths, Nruns, rundir, work_df, Z, i, j):

    #dS = abs(np.asarray(work_df['S']) - S)
    #dZ = abs(np.asarray(work_df['Z']) - Z)

    # prepare the parameters based on previous calibrations
    D0 = [min(work_df['D_0']), max(work_df['D_0'])]
    d = [min(work_df['d']), max(work_df['d'])]
    Ksat = [min(work_df['K_sat']), max(work_df['K_sat'])]
    IzKs = [min(work_df['Iz_over_K_steady']), max(work_df['Iz_over_K_steady'])]
    Frangle = [min(work_df['friction_angle']), max(work_df['friction_angle'])]
    coh = [min(work_df['cohesion']), max(work_df['cohesion'])]
    Wsoil = [min(work_df['weight_of_soil']), max(work_df['weight_of_soil'])]

    farmin = 0.75
    farmax = 1.25

    # set the parameters for initial the MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.95*S, D_0_min = farmin*D0[0],K_sat_min = farmin*Ksat[0], d_min = farmin*d[0], Iz_over_K_steady_min = farmin*IzKs[0], friction_angle_min = farmin*Frangle[0], cohesion_min = farmin*coh[0], weight_of_water_min = 9800, weight_of_soil_min = farmin*Wsoil[0],
        alpha_max = 1.05*S, D_0_max = farmax*D0[1],K_sat_max = farmax*Ksat[1], d_max = farmax*d[1],Iz_over_K_steady_max = farmax*IzKs[1], friction_angle_max = farmax*Frangle[1], cohesion_max = farmax*coh[1], weight_of_water_max = 9801, weight_of_soil_max = farmax*Wsoil[1], depths = depths)

    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = "test_MC.csv", n_iterations = Nruns, replace = True)

    # now open the MC test file
    results = pd.read_csv(rundir+"test_MC.csv")

    return results



################################################################################
################################################################################
def MC_loop (rain, S, depths, Nruns, rundir, work_df):

    # In here, we run 3 different MC simulations
    N1 = int(np.floor(Nruns /4.)); N2 = int(np.ceil(Nruns /4.)); N3 = int(np.ceil(Nruns /2.))

    # prepare the parameters
    D0 = [min(work_df['D_0']), max(work_df['D_0'])]
    d = [min(work_df['d']), max(work_df['d'])]
    Ksat = [min(work_df['K_sat']), max(work_df['K_sat'])]
    IzKs = [min(work_df['Iz_over_K_steady']), max(work_df['Iz_over_K_steady'])]
    Frangle = [min(work_df['friction_angle']), max(work_df['friction_angle'])]
    coh = [min(work_df['cohesion']), max(work_df['cohesion'])]
    Wsoil = [min(work_df['weight_of_soil']), max(work_df['weight_of_soil'])]

    closemin = 0.9
    closemax = 1.1
    farmin = 0.75
    farmax = 1.25


    # set parameters for the 1st set of MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.95*S, D_0_min = closemin*D0[0],K_sat_min = closemin*Ksat[0], d_min = closemin*d[0], Iz_over_K_steady_min = closemin*IzKs[0], friction_angle_min =closemin*Frangle[0], cohesion_min = closemin*coh[0], weight_of_water_min = 9800, weight_of_soil_min = closemin*Wsoil[0],
        alpha_max = 1.05*S, D_0_max = closemax*D0[1],K_sat_max = closemax*Ksat[1], d_max = closemax*d[1],Iz_over_K_steady_max = closemax*IzKs[1], friction_angle_max = closemax*Frangle[1], cohesion_max = closemax*coh[1], weight_of_water_max = 9801, weight_of_soil_max = closemax*Wsoil[1], depths = depths)
    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = "test_MC_close.csv", n_iterations = N1, replace = True)

    # now open the MC test filexx
    results_close = pd.read_csv(rundir+"test_MC_close.csv")


    # set parameters for the 2nd set of MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.95*S, D_0_min = farmin*D0[0],K_sat_min = farmin*Ksat[0], d_min = farmin*d[0], Iz_over_K_steady_min = farmin*IzKs[0], friction_angle_min = farmin*Frangle[0], cohesion_min = farmin*coh[0], weight_of_water_min = 9800, weight_of_soil_min = farmin*Wsoil[0],
        alpha_max = 1.05*S, D_0_max = farmax*D0[1],K_sat_max = farmax*Ksat[1], d_max = farmax*d[1],Iz_over_K_steady_max = farmax*IzKs[1], friction_angle_max = farmax*Frangle[1], cohesion_max = farmax*coh[1], weight_of_water_max = 9801, weight_of_soil_max = farmax*Wsoil[1], depths = depths)
    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = "test_MC_far.csv", n_iterations = N2, replace = True)

    # now open the MC test filexx
    results_far = pd.read_csv(rundir+"test_MC_far.csv")
    results = results_close.append(results_far, ignore_index = True)


    # set initial parameters for the MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = 0.95*S, D_0_min = 1e-6,K_sat_min = 1e-8, d_min = 0.5, Iz_over_K_steady_min = 0.1, friction_angle_min = 0.2, cohesion_min = 5000, weight_of_water_min = 9800, weight_of_soil_min = 15000,
        alpha_max = 1.05*S, D_0_max = 1e-4,K_sat_max = 1e-6, d_max = 3,Iz_over_K_steady_max = 0.8, friction_angle_max = 0.5, cohesion_max = 20000, weight_of_water_max = 9801, weight_of_soil_max = 25000, depths = depths)
    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = "test_MC_ini.csv", n_iterations = N3, replace = True)

    # now open the MC test file
    results_ini = pd.read_csv(rundir+"test_MC_ini.csv")
    results = results.append(results_ini, ignore_index = True)

    return results
