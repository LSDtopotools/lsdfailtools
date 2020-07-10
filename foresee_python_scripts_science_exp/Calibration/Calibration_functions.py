###
# Has been checked for imports not working. 
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
import lsdfailtools.iverson2000 as iverson

import matplotlib.lines as mlines
from itertools import product


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


################################################################################
################################################################################
def calc_dist2road(dimensions, roadline):

    print ('Calculating distance to road')

    # set up the conditions for calibration to happen
    distarr = np.zeros((dimensions), dtype = np.float)

    # now convert it to pixel coordinates
    roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
    roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
    roadline = roadline.astype('int')
    l = mlines.Line2D(roadline[:,0], roadline[:,1])


    # then calculate a matrix of distances to it!
    print ('calculating distances')
    for i,j in itertools.product(range(demarr.shape[0]), range(demarr.shape[1])):
            distarr[i,j] = min((j-roadline[:,0])**2 + (i-roadline[:,1])**2)
    distarr[distarr <= 0.] = 1

    return distarr


################################################################################
################################################################################
def select_pixels(distarr, Num_cal):
    # this bit can definitely be improved on!

    print ('Selecting pixels for calibration')

    selectarr = (100 - distarr**(1/2) ) / 5000
    selectarr[selectarr <=0.] = 0.0

    final_selectarr = 0 * selectarr

    npoints = 0
    iterations = 0
    while npoints < Num_cal:
        print ('iteration:', iterations)
        for i,j in product (range(demarr.shape[0]), range(demarr.shape[1])):
            die_roll = np.random.rand()

            if selectarr[i,j] > die_roll and final_selectarr[i,j] != 1  and failarr[i,j]-prefailarr[i,j] > 3*24*3600 and failarr[i,j]-prefailarr[i,j] < 100*24*3600 and npoints < Num_cal:

                final_selectarr[i,j] = 1
                npoints += 1

        iterations +=1

    return final_selectarr


################################################################################
################################################################################
def make_storage_df(results, selected, S, Z, i, j, F, P):
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
def calibrate_points_MC(final_selectarr, demarr, slopearr, failarr, prefailarr, rain, depths, Nruns, rundir):

    print ('Starting calibration')

    # Time the calibration
    start = datetime.now()

    # initialisation
    work_df_exists = 0
    storage_df_exists = 0
    npoints = 0

    # Run through the pixels
    for i,j in product (range(demarr.shape[0]), range(demarr.shape[1])):

        # If the selection condition is met
        if final_selectarr[i,j] == 1:
            print ('Calibrating pixel of coordinates:', i, j)

            #  Define the pixel values from arrays
            Z = demarr[i,j]
            S = slopearr[i,j]
            F = failarr[i,j]
            P = prefailarr[i,j]

            # Define the interval of time in which the failure is observed on InSAR data
            failinterval = F-P
            print ('Observed failure interval:', failinterval/(24*3600), 'days')

            # if this is the first pixel to be calibrated
            if work_df_exists == 0:
                # run the initial MC simulation
                results = MC_initial(rain, S, depths, Nruns, rundir)
            else:
                # run the initial MC simulation
                results = MC_assisted(rain, S, depths, Nruns, rundir, work_df, Z, i, j)

            # Select the results with the best fitness - NEEDS WORK!
            selected = assess_fitness(results, F, P, Nruns)

            # store the most succesful runs in a dataframe
            work_df = make_storage_df(results, selected, S, Z, i, j, F, P)
            work_df_exists = 1

            # These are the indices in the work_df that are correctly calibrated
            inbounds_ID = []

            # Now that we have initiated calibration, let's run through it until the point is calibrated
            # Note: itermax fixes a runtime safety on the while loop but I may have been a bit strict, which means that points where failure is observed in a very small window are usually not succesfully calibrated and end up being skipped
            n = 0
            while len(inbounds_ID) < 2 and n < itermax:
                print ('iteration:', n)

                # run the MC - NEEDS WORK!
                results = MC_loop (rain, S, depths, Nruns, rundir, work_df)

                # Select the results with the best fitness - NEEDS WORK!
                selected = assess_fitness(results, F, P, Nruns)

                # store the most succesful runs (inbounds) in a dataframe
                temp_df = make_storage_df(results, selected, S, Z, i, j, F, P)

                # Append that dataframe to the existing one
                work_df = work_df.append(temp_df, ignore_index = True)

                # for each selected run, A > 0 if the run predicts a failure before the end of the time interval observed on InSAR data
                A = np.sign(work_df['insar_failtime'] - work_df['time_of_failure'])
                # for each selected run, B > 0 if the run predicts a failure after the beginning of the time interval observed on InSAR data
                B = np.sign(work_df['time_of_failure'] - work_df['insar_prefailtime'])

                # Those runs have produced a "correctly calibrated" result
                inbounds_ID = np.where(np.logical_and(A > 0, B > 0))[0]

                # if there is more than one successful run
                if len(inbounds_ID) > 1:
                    print ('Attempt number', n, ': we have a successful calibration!')
                    # keep the successful runs
                    work_df = work_df.iloc[inbounds_ID]

                    # if this is the first point we have calibrated successfully
                    if storage_df_exists == 0:
                        print ('initiating storage of succesfully calibrated pixels')
                        storage_df = work_df.copy(deep = True)
                        storage_df_exists = 1
                    else:
                        print ('adding a calibrated pixel to storage')
                        storage_df = storage_df.append(work_df, ignore_index = True)

                    # we can stop here, the pixel is calibrated
                    break

                n+=1
                print ()

            npoints +=1
            # once we have calibrated all the desired pixels (or we have run out of pixels to calibrate because we've skipped some ... ).
            # This IF statement is not super useful, it just stops the script from running through pixels not selected for calibration.

            if npoints >= Num_cal:

                # Save all these pixels to a .csv file
                storage_df.to_csv(rundir + 'Calibrated.csv')

                print ('Calibrated', npoints, 'pixels in ', datetime.now()-start)
                print ()

                break







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
