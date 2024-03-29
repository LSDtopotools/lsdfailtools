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
#sys.path.insert(0,'../lsdfailtools/')
#sys.path.insert(0,'/exports/csce/datastore/geos/users/s1440040/projects/lsdfailtools/lsdfailtools/')
#Importing the model
import iverson2000 as iverson
from itertools import product
import matplotlib.lines as mlines



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


    # then calculate a matrix of distances to it!
    for i,j in product(range(dimensions[0]), range(dimensions[1])):
            distarr[i,j] = min((j-roadline[:,0])**2 + (i-roadline[:,1])**2)
    distarr[distarr <= 0.] = 1

    return distarr

######################################################
######################################################
# A figure to show the FoS vs min depth for a single point
######################################################
######################################################

def FoS_vs_failure_depth(factor_of_safety, depth, points, fig_height, fig_width, fig_name):
	# Load the data
	fig=plt.figure(1, facecolor='White',figsize=[fig_width, fig_height])

	axis = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)

	# Plot the figure
	for i in range(0,points):
		factor_of_safety_temp = factor_of_safety[i]
		print(len(factor_of_safety_temp))
		depth_temp= depth[i]
		print(len(depth_temp))
		axis.scatter(factor_of_safety_temp,depth_temp)

		axis.set_xlabel('Factor of Safety')
		axis.set_ylabel('Min failure depth')
		plt.savefig(fig_name)




######################################################
######################################################
# A figure to map validation results more extensively using a single point
######################################################
######################################################
def run_validation_single_output(rain, depths, calibrated, demarr, slopearr, failarr,rundir):


    # how do you select the parameters?
    # based on location and slope

    # make slope bins in the calibrated df
    #shist, sbins = np.histogram(calibrated['S'], bins  = 10)
    sbins = np.arange(0,np.amax(slopearr), 0.05)

    valid_df= pd.DataFrame(columns=['alpha', 'D_0', 'K_sat', 'd','Iz_over_K_steady','friction_angle','cohesion','weight_of_water','weight_of_soil','time_of_failure','factor_of_safety', 'min_depth','S','Z','row','col','observed_failtime'])
    # Set up the arrays to be written to
    Psi = []
    FoS =[]
    time_index = []
    #for i,j in product(range(slopearr.shape[0]), range(slopearr.shape[1])):
    for i,j in product(range(0,20,1), range(500,820,1)):
        #for i,j in product(range(500,600,1), range(800,900,1)):

        if failarr[i,j] > 0.:
            print (i,j)

            # this is our slope
            S = slopearr[i,j]
            Z = demarr[i,j]

            # find out in which bin it is
            lesser = sbins[np.where(sbins < S)[0][-1]]
            greater = sbins[np.where(sbins >= S)[0][0]]

            #find the points that have been calibrated in this range
            sdf = calibrated[calibrated['S'] >= lesser]
            print(sdf)
            sdf = sdf[sdf['S'] < greater]
            print(sdf)

            # If there are indeed points in this category
            if len(sdf)<1:
                print ('woops, there is nothing here')
            else:
                dist = np.sqrt((j-sdf['row'])**2 + (i-sdf['col'])**2)
                where = np.where(np.asarray(dist) == min(np.asarray(dist)))[0]
                print(where)

                select_df = sdf.iloc[where]
                print(select_df.head())
                mean_df = select_df.iloc[0]
                print(mean_df.head())
                #mean_df = select_df.mean(axis = 0)

                mymodel = iverson.iverson_model(alpha = S,
                    D_0 = mean_df['D_0'],
                    K_sat = mean_df['K_sat'],
                    d = mean_df['d'],
                    Iz_over_K_steady = mean_df['Iz_over_K_steady'],
                    friction_angle = mean_df['friction_angle'],
                    cohesion = mean_df['cohesion'],
                    weight_of_water = mean_df['weight_of_water'],
                    weight_of_soil = mean_df['weight_of_soil'],
                    depths = depths)

                mymodel.run_with_outputs(rain['duration_s'].values, rain['intensity_mm_sec'].values)

                print("Getting outputs")

                Times = mymodel.cppmodel.output_times
                failures = mymodel.cppmodel.output_failure_times
                failures_b = mymodel.cppmodel.output_failure_bool

                # min_depth_temp = mymodel.cppmodel.output_failure_mindepths
                # min_depth.append(min_depth_temp)

                if len(failures) > 0 and len(failures_b[failures_b ==True]) > 1:
                    failures = failures[failures_b ==True][0]
                else:
                    failures = 0

                if failures != 0:
                    print("Found a failure")
                    index = numpy.where(failures_b ==True)[0]
                    index = index[0]
                    time_index.append(index)
                     # Output the Psi data
                    Psi_temp = mymodel.cppmodel.output_Psi_timedepth
                    Psi.append(Psi_temp)
                    # Output the FoS
                    FoS_temp = mymodel.cppmodel.output_FS_timedepth
                    FoS.append(FoS_temp)


                valid_df = valid_df.append({'alpha':S, 'D_0':mean_df['D_0'], 'K_sat':mean_df['K_sat'], 'd':mean_df['d'],'Iz_over_K_steady':mean_df['Iz_over_K_steady'],'friction_angle':mean_df['friction_angle'],'cohesion':mean_df['cohesion'],'weight_of_water':mean_df['weight_of_water'],'weight_of_soil':mean_df['weight_of_soil'],'time_of_failure':failures, 'Time':Times, 'S':S,'Z':Z,'row':i,'col':j,'observed_failtime':failarr[i,j]}, ignore_index=True)
                print(valid_df.head())

    numpy.save(rundir+"Psi", Psi)
    numpy.save(rundir+"fos", FoS)
    numpy.save(rundir+"time_index", time_index)

    # numpy.save(rundir+"min_depth", min_depth)
    valid_df.to_csv(rundir + 'Validated_updated.csv', index=False)




######################################################
######################################################
# A figure to map validation results more extensively using a single point
######################################################
######################################################
# need to clear thisup - we don't need failarr_val file.
def get_fos_point_of_interest(rain, depths, calibrated_df, point_to_test_lat, point_to_test_lon, demarr_val, slopearr_val, failarr_val,rundir):

    #valid_df= pd.DataFrame(columns=['alpha', 'D_0', 'K_sat', 'd','Iz_over_K_steady','friction_angle','cohesion','weight_of_water','weight_of_soil','time_of_failure','factor_of_safety', 'min_depth','S','Z','row','col','observed_failtime'])
    # Set up the arrays to be written to
    Psi = []
    FoS =[]
    time_index = []

    #print(f'this is the failarr value: {failarr_val}')
    if failarr_val == 1000000.:
        print(f'The point ({point_to_test_lat}, {point_to_test_lon}) is unlikely to fail.')

    else:
        print(f'The point ({point_to_test_lat}, {point_to_test_lon}) is likely to fail.')
        print('I am now going to get some extra information and plots for you.')
        print()

        # this is our slope
        S = slopearr_val
        Z = demarr_val

        mymodel = iverson.iverson_model(alpha = S,
            D_0 = calibrated_df['D_0'],
            K_sat = calibrated_df['K_sat'],
            d = calibrated_df['d'],
            Iz_over_K_steady = calibrated_df['Iz_over_K_steady'],
            friction_angle = calibrated_df['friction_angle'],
            cohesion = calibrated_df['cohesion'],
            weight_of_water = calibrated_df['weight_of_water'],
            weight_of_soil = calibrated_df['weight_of_soil'],
            depths = depths)

        mymodel.run_with_outputs(rain['duration_s'].values, rain['intensity_mm_sec'].values)

        print("Getting outputs")

        Times = mymodel.cppmodel.output_times
        # self.cppmodel.output_times:
        ## 1d array of time for each outputs

        failures = mymodel.cppmodel.output_failure_times
        # self.cppmodel.output_failure_times
        ## 1darray (dim = time) erm the time corresponding to the output_failure_bool

        failures_b = mymodel.cppmodel.output_failure_bool

        # self.cppmodel.output_minFS
        ## 1D array (dim=time) giving the minimum FS of the whole depths column (its depths is given by self.cppmodel.output_depthsFS)
        FoS = mymodel.cppmodel.output_minFS


        # self.cppmodel.output_FS_timedepth
        ## 2D array (dims = time,depths) of factor of safety values
        FoS_temp = mymodel.cppmodel.output_FS_timedepth
        ## 1D array (dim=time) of depths where the minimum factor of safety is for each timestep
        ## Might always be the surface that ones!!!
        min_depth = mymodel.cppmodel.output_depthsFS


        numpy.save(rundir + f"FoS_temp_{int(point_to_test_lat)}_{int(point_to_test_lon)}", FoS_temp)
        numpy.save(rundir + f"FoS_{int(point_to_test_lat)}_{int(point_to_test_lon)}", FoS)
        numpy.save(rundir + f"min_depth_{int(point_to_test_lat)}_{int(point_to_test_lon)}", min_depth)

        return point_to_test_lat, point_to_test_lon


######################################################
######################################################
# A figure to map validation results
######################################################
######################################################
def run_validation(rain, depths, calibrated, demarr, slopearr, failarr,rundir, early_late):


    # how do you select the parameters?
    # based on location and slope

    sbins = np.arange(0,np.amax(slopearr), 0.05)
    count = 0

    valid_df= pd.DataFrame(columns=['alpha', 'D_0', 'K_sat', 'd','Iz_over_K_steady','friction_angle','cohesion','weight_of_water','weight_of_soil','time_of_failure','factor_of_safety', 'min_depth','S','Z','row','col','observed_failtime'])

    for i,j in product(range(slopearr.shape[0]), range(slopearr.shape[1])):
        if failarr[i,j] > 0.:
            print (i,j)

            # this is our slope
            S = slopearr[i,j]
            Z = demarr[i,j]

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
                mean_df = select_df.iloc[0]

                mymodel = iverson.iverson_model(alpha = S,
                    D_0 = mean_df['D_0'],
                    K_sat = mean_df['K_sat'],
                    d = mean_df['d'],
                    Iz_over_K_steady = mean_df['Iz_over_K_steady'],
                    friction_angle = mean_df['friction_angle'],
                    cohesion = mean_df['cohesion'],
                    weight_of_water = mean_df['weight_of_water'],
                    weight_of_soil = mean_df['weight_of_soil'],
                    depths = depths)

                mymodel.run(rain['duration_s'].values, rain['intensity_mm_sec'].values)


                failures = mymodel.cppmodel.output_failure_times
                failures_b = mymodel.cppmodel.output_failure_bool
                FoS = mymodel.cppmodel.output_minFS
                min_depth = mymodel.cppmodel.output_depthsFS

                if len(failures) > 0 and len(failures_b[failures_b ==True]) > 1:
                    failures = failures[failures_b ==True][0]
                    FoS = FoS[failures_b ==True][0]
                    min_depth = min_depth[failures_b ==True][0]
                else:
                    failures = 0
                    FoS = 0
                    min_depth = 0

                count += 1
                valid_df = valid_df.append({'alpha':S, 'D_0':mean_df['D_0'], 'K_sat':mean_df['K_sat'], 'd':mean_df['d'],'Iz_over_K_steady':mean_df['Iz_over_K_steady'],'friction_angle':mean_df['friction_angle'],'cohesion':mean_df['cohesion'],'weight_of_water':mean_df['weight_of_water'],'weight_of_soil':mean_df['weight_of_soil'],'time_of_failure':failures, 'factor_of_safety':FoS, 'min_depth':min_depth,'S':S,'Z':Z,'row':i,'col':j,'observed_failtime':failarr[i,j]}, ignore_index=True)

    valid_df.to_csv(rundir + f'{early_late}_Validated_updated_FoS_depth_100_0.csv', index=False)


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
