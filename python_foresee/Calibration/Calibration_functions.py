################################################################################
################################################################################
"""Import Python packages"""
################################################################################
################################################################################

import os
import re
import sys
import csv
import time
import shutil
import tkinter
import platform
import argparse
import datetime
import numpy as np
import pandas as pd
from itertools import product
from tkinter import filedialog
import matplotlib.lines as mlines
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *

import lsdfailtools.iverson2000 as iverson




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

        cols = inDs.RasterXSize
        rows = inDs.RasterYSize
        bands = inDs.RasterCount

        geotransform = inDs.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]

        band = inDs.GetRasterBand(1)

        image_array = band.ReadAsArray(0, 0, cols, rows)
        image_array_name = file_name


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
def calc_dist2road(dimensions, roadline, geotransform):

    print ('Calculating distance to road')

    # set up the conditions for calibration to happen
    distarr = np.zeros((dimensions), dtype = np.float)

    # now convert it to pixel coordinates
    roadline[:,0] = (roadline[:,0] - geotransform[0]) / geotransform[1] # X_coord
    roadline[:,1] = (roadline[:,1] - geotransform[3]) / geotransform[5] # Y_coord
    roadline = roadline.astype('int')
    l = mlines.Line2D(roadline[:,0], roadline[:,1])


    # then calculate a matrix of distances to it
    for i,j in product(range(dimensions[0]), range(dimensions[1])):
            distarr[i,j] = min((j-roadline[:,0])**2 + (i-roadline[:,1])**2)
    distarr[distarr <= 0.] = 1

    return distarr

################################################################################
################################################################################
def GW_depth_ini(Piezo_data, Start):
    GW_depth = Piezo_data[ ['DATE', 'LIV1', 'LIV2', 'LIV3', 'LIV4'] ]

    StartDate = datetime.datetime.strptime(Start,'%Y-%m-%d').strftime('%d/%m/%Y')

    indices = []
    for i in range(len(GW_depth)):
        if GW_depth['DATE'].iloc[i][3:5] == StartDate[3:5]:
            print (i)
            print ( StartDate[3:5] )
            indices.append(i)

    GW = np.asarray( GW_depth[['LIV1', 'LIV2', 'LIV3', 'LIV4'] ].iloc[indices] )
    GW_mean = np.mean(GW[~np.isnan(GW)])

    return GW_mean




################################################################################
################################################################################
def select_pixels(distarr, failarr, Num_cal):
    print ('Selecting pixels for calibration')

    # this is too strict for Sentinel points, which are further than most of those we had with InSAr
    selectarr = (100 - distarr**(1/2) ) / 5000

    selectarr[selectarr <=0.] = 0.0


    final_selectarr = 0 * selectarr

    npoints = 0
    iterations = 0
    while npoints < Num_cal:
        print ('iteration:', iterations)
        for i,j in product (range(distarr.shape[0]), range(distarr.shape[1])):

            if failarr[i,j] > 0: # if there is a failure

                die_roll = np.random.rand()

                if selectarr[i,j] > die_roll :

                    final_selectarr[i,j] = 1
                    npoints += 1

        iterations +=1

    return final_selectarr


################################################################################
################################################################################
def make_storage_df(results, selected, S, Z, i, j, F):
    work_df = results.iloc[selected]
    work_df['S'] = S
    work_df['Z'] = Z
    work_df['row'] = i
    work_df['col'] = j
    work_df['observed_failtime'] = F


    return work_df


################################################################################
################################################################################
def calibrate_points_MC(final_selectarr, demarr, slopearr, failarr, rain, GW, Cal_params, Iverson_MC_params, rundir):


    # Number of MonteCarlo runs
    Nruns = Cal_params.at[0,'Nruns']
    # Max Number of iterations of the MC process
    itermax = Cal_params.at[0,'itermax']
    # Number of points to calibrate
    Num_cal = Cal_params.at[0,'Num_cal']

    # define the accepted window to simulate acceptable failure times
    failinterval = Cal_params.at[0,'failinterval'] * 24 * 3600


    print ('Starting calibration')
    # Time the calibration
    start = datetime.datetime.now()

    # initialisation
    work_df_exists = 0
    storage_df_exists = 0
    npoints = 0

    # Run through the pixels
    while npoints < Num_cal:
        for i,j in product (range(demarr.shape[0]), range(demarr.shape[1])):

            # If the selection condition is met
            if final_selectarr[i,j] == 1:
                print ('Examining pixel of coordinates:', i, j)
                # make sure you don't calibrate them in order so that you can run it in several sittings without influencing the distribution too much
                proba = np.random.rand()
                print ('Does', proba, 'exceed 0.9 ?')
                if 1-proba > 0.9:
                    print ('Rejoice! The dice has chosen you for calibration.')

                    #  Define the pixel values from arrays
                    Z = demarr[i,j]
                    S = slopearr[i,j]
                    F = failarr[i,j]


                    # if this is the first pixel to be calibrated
                    if work_df_exists == 0:
                        # run the initial MC simulation
                        results = MC_initial(rain, GW, S, Nruns, Iverson_MC_params, rundir)
                    else:
                        # run the initial MC simulation
                        results = MC_assisted(rain, GW, S, Nruns, Iverson_MC_params, rundir, work_df, Z, i, j)

                    # Select the results with the best fitness
                    selected = assess_fitness(results, F, failinterval, Nruns)

                    # store the most succesful runs in a dataframe
                    work_df = make_storage_df(results, selected, S, Z, i, j, F)
                    work_df_exists = 1

                    # These are the indices in the work_df that are correctly calibrated
                    inbounds_ID = []

                    # Now that we have initiated calibration, let's run through it until the point is calibrated
                    # Note: itermax fixes a runtime safety on the while loop but I may have been a bit strict, which means that points where failure is observed in a very small window are usually not succesfully calibrated and end up being skipped
                    n = 0
                    while len(inbounds_ID) < 2 and n < itermax:
                        print ('MC iteration:', n)

                        # run the MC
                        results = MC_loop (rain, GW, S, Nruns, Iverson_MC_params, rundir, work_df)

                        # Select the results with the best fitness
                        selected = assess_fitness(results, F, failinterval, Nruns)

                        # store the most succesful runs (inbounds) in a dataframe
                        temp_df = make_storage_df(results, selected, S, Z, i, j, F)

                        # Append that dataframe to the existing one
                        work_df = work_df.append(temp_df, ignore_index = True)



                        Diff = np.asarray(work_df['time_of_failure'] - work_df['observed_failtime']) / (24*3600)
                        Pos = Diff[Diff>=0]
                        Neg = Diff[Diff<0]

                        if len(Pos) > 0:
                            print ('Mean pos:', np.mean(Pos), 'best pos', np.amin(Pos))
                        if len(Neg) > 0:
                            print ('Mean neg:', np.mean(Neg), 'best neg', np.amax(Neg))

                        inbounds_ID = np.where(np.logical_and(Diff <= failinterval/(24*3600), Diff >= -failinterval/(24*3600)))[0]

                        print (inbounds_ID)

                        # if there is more than one successful run
                        if len(inbounds_ID) >= 1:
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

                        if n == itermax:
                            print ("Warning: exceeded runtime limit. The point was not calibrated")

                        print ()

                    npoints +=1
                    # once we have calibrated all the desired pixels (or we have run out of pixels to calibrate because we've skipped some ... ).
                    # This IF statement is not super useful, it just stops the script from running through pixels not selected for calibration.



                    # Save all these pixels to a .csv file
                    storage_df.to_csv(rundir + 'Calibrated_FoS_depth.csv')

                    print ('Calibrated', npoints, 'pixels in ', datetime.datetime.now()-start)
                    print ()

                    if npoints >= Num_cal:

                        break







################################################################################
################################################################################
def assess_fitness (results, F, failinterval, Nruns):
    # extract the failtimes
        failtimes = np.asarray(results['time_of_failure'])

        # calculate the time differences
        notlater = failtimes + failinterval
        notsooner = failtimes - failinterval

        # Which failtimes are within observed times?
        inbounds_ID = np.where(np.logical_and(notlater >= 0, notsooner>=0))[0]

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
def MC_initial (rain, GW, S, Nruns, P, rundir):

    depths  = np.arange(P.at[0,'depth'], P.at[1,'depth'], 0.2)

    # set the parameters for initial the MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = S, D_0_min = P.at[0,'D_0'], K_sat_min = P.at[0,'K_sat'], d_min = GW, Iz_over_K_steady_min = P.at[0,'Iz_over_K_steady'], friction_angle_min = P.at[0,'friction_angle'], cohesion_min = P.at[0,'cohesion'], weight_of_water_min = P.at[0,'weight_of_water'], weight_of_soil_min = P.at[0,'weight_of_soil'],
        alpha_max = S, D_0_max = P.at[1,'D_0'], K_sat_max = P.at[1,'K_sat'], d_max = GW,Iz_over_K_steady_max = P.at[1,'Iz_over_K_steady'], friction_angle_max = P.at[1,'friction_angle'], cohesion_max = P.at[1,'cohesion'], weight_of_water_max = P.at[1,'weight_of_water'], weight_of_soil_max = P.at[1,'weight_of_soil'], depths = depths)

    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = rundir+"test_MC.csv", n_iterations = Nruns, replace = True)

    # now open the MC test file
    results = pd.read_csv(rundir+"test_MC.csv")

    return results



################################################################################
################################################################################
def MC_assisted(rain, GW, S, Nruns, P, rundir, work_df, Z, i, j):

    depths  = np.arange(P.at[0,'depth'], P.at[1,'depth'], 0.2)

    # prepare the parameters based on previous calibrations
    D0 = [min(work_df['D_0']), max(work_df['D_0'])]
    d = [min(work_df['d']), max(work_df['d'])]
    Ksat = [min(work_df['K_sat']), max(work_df['K_sat'])]
    IzKs = [min(work_df['Iz_over_K_steady']), max(work_df['Iz_over_K_steady'])]
    Frangle = [min(work_df['friction_angle']), max(work_df['friction_angle'])]
    coh = [min(work_df['cohesion']), max(work_df['cohesion'])]
    Wsoil = [min(work_df['weight_of_soil']), max(work_df['weight_of_soil'])]

    farmin = 0.5
    farmax = 1.5

    # set the parameters for initial the MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = S, D_0_min = farmin*D0[0],K_sat_min = farmin*Ksat[0], d_min = GW, Iz_over_K_steady_min = farmin*IzKs[0], friction_angle_min = farmin*Frangle[0], cohesion_min = farmin*coh[0], weight_of_water_min = 9800, weight_of_soil_min = farmin*Wsoil[0],
        alpha_max = S, D_0_max = farmax*D0[1],K_sat_max = farmax*Ksat[1], d_max = GW,Iz_over_K_steady_max = farmax*IzKs[1], friction_angle_max = farmax*Frangle[1], cohesion_max = farmax*coh[1], weight_of_water_max = 9801, weight_of_soil_max = farmax*Wsoil[1], depths = depths)

    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = rundir+"test_MC.csv", n_iterations = Nruns, replace = True)

    # now open the MC test file
    results = pd.read_csv(rundir+"test_MC.csv")

    return results



################################################################################
################################################################################
def MC_loop (rain, GW, S, Nruns, P, rundir, work_df):

    depths  = np.arange(P.at[0,'depth'], P.at[1,'depth'], 0.2)

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

    closemin = 0.8
    closemax = 1.2
    farmin = 0.5
    farmax = 1.5


    # set parameters for the 1st set of MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = S, D_0_min = closemin*D0[0],K_sat_min = closemin*Ksat[0], d_min = GW, Iz_over_K_steady_min = closemin*IzKs[0], friction_angle_min =closemin*Frangle[0], cohesion_min = closemin*coh[0], weight_of_water_min = 9800, weight_of_soil_min = closemin*Wsoil[0],
        alpha_max = S, D_0_max = closemax*D0[1],K_sat_max = closemax*Ksat[1], d_max = GW,Iz_over_K_steady_max = closemax*IzKs[1], friction_angle_max = closemax*Frangle[1], cohesion_max = closemax*coh[1], weight_of_water_max = 9801, weight_of_soil_max = closemax*Wsoil[1], depths = depths)
    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = rundir+"test_MC_close.csv", n_iterations = N1, replace = True)

    # now open the MC test file
    results_close = pd.read_csv(rundir+"test_MC_close.csv")




    # set the parameters for initial the MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = S, D_0_min = P.at[0,'D_0'], K_sat_min = P.at[0,'K_sat'], d_min = GW, Iz_over_K_steady_min = P.at[0,'Iz_over_K_steady'], friction_angle_min = P.at[0,'friction_angle'], cohesion_min = P.at[0,'cohesion'], weight_of_water_min = P.at[0,'weight_of_water'], weight_of_soil_min = P.at[0,'weight_of_soil'],
        alpha_max = S, D_0_max = P.at[1,'D_0'], K_sat_max = P.at[1,'K_sat'], d_max = GW,Iz_over_K_steady_max = P.at[1,'Iz_over_K_steady'], friction_angle_max = P.at[1,'friction_angle'], cohesion_max = P.at[1,'cohesion'], weight_of_water_max = P.at[1,'weight_of_water'], weight_of_soil_max = P.at[1,'weight_of_soil'], depths = depths)

    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = rundir+"test_MC_far.csv", n_iterations = N2, replace = True)

    # now open the MC test file
    results_far = pd.read_csv(rundir+"test_MC_far.csv")
    results = results_close.append(results_far, ignore_index = True)


    # set initial parameters for the MC runs
    MCrun = iverson.MonteCarlo_Iverson( alpha_min = S, D_0_min = P.at[0,'D_0'], K_sat_min = P.at[0,'K_sat'], d_min = GW, Iz_over_K_steady_min = P.at[0,'Iz_over_K_steady'], friction_angle_min = P.at[0,'friction_angle'], cohesion_min = P.at[0,'cohesion'], weight_of_water_min = P.at[0,'weight_of_water'], weight_of_soil_min = P.at[0,'weight_of_soil'],
        alpha_max = S, D_0_max = P.at[1,'D_0'], K_sat_max = P.at[1,'K_sat'], d_max = GW,Iz_over_K_steady_max = P.at[1,'Iz_over_K_steady'], friction_angle_max = P.at[1,'friction_angle'], cohesion_max = P.at[1,'cohesion'], weight_of_water_max = P.at[1,'weight_of_water'], weight_of_soil_max = P.at[1,'weight_of_soil'], depths = depths)
    # Now run it
    MCrun.run_MC_failure_test(rain["duration_s"].values, rain["intensity_mm_sec"].values,
                      n_process = 2, output_name = rundir+"test_MC_ini.csv", n_iterations = N3, replace = True)

    # now open the MC test file
    results_ini = pd.read_csv(rundir+"test_MC_ini.csv")
    results = results.append(results_ini, ignore_index = True)

    return results
