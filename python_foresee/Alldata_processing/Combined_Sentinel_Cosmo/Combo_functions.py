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
import pandas as pd
import geopandas as gpd
from numpy.linalg import lstsq
from scipy import stats




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

        #Get image size
        cols = inDs.RasterXSize
        rows = inDs.RasterYSize
        bands = inDs.RasterCount

        # Get georeference information
        geotransform = inDs.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]


        # Set pixel offset
        # Convert image to 2D array
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




######################################################################
######################################################################

def reproject_shp(shp_in, epsg_out):
    print('Reprojecting', shp_in)
    shapes_in = gpd.read_file(shp_in)
    shapes_out = shapes_in.to_crs({'init': 'epsg:'+str(int(epsg_out))})
    shapes_out.to_file(driver = 'ESRI Shapefile', filename= shp_in[:-4]+"_epsg"+str(int(epsg_out))+".shp")



######################################################################
######################################################################
def get_coordinates(df):
    df_coords = []
    for i in range(len(df)):
        df_coords.append( [df['geometry'].iloc[i].x, df['geometry'].iloc[i].y] )
    df_coords = np.asarray(df_coords)
    return df_coords

######################################################################
######################################################################
# Find common values of two lists
def common(a,b):
    c = [value for value in a if value in b]
    return c

######################################################################
######################################################################
def indices_in_box(xbox, ybox, coords):
    inside_x = np.where( np.logical_and (coords[:,0] > xbox[0], coords[:,0] < xbox[1]) )[0]
    inside_y = np.where( np.logical_and (coords[:,1] > ybox[1], coords[:,1] < ybox[0]) )[0]

    if len(inside_x) > 0 and len(inside_y) > 0:
        pinpoint = common (inside_x, inside_y)
        return pinpoint

    else:
        return []
######################################################################
######################################################################
def make_av10(data, dates, intervals):
    av10 = []
    for i in range(len(data)):
        A = []
        for j in range(5, len(data[i])-5):
            A.append(np.mean(data[i,j-5:j+5]))
        av10.append(A)


    av10 = np.asarray(av10)
    dates_av10 = dates[5:-5]
    intervals_av10 = intervals[5:-5]
    return av10, dates_av10, intervals_av10

######################################################################
######################################################################
def calculate_av10_dva(s, s_intervals_yr, ewv, ewv_intervals_yr):
    # This is cumulative displacement in mm
    cumdisp = np.array(s[datecols])[0]

    # This is a 10-measurement moving average of cumul. displacement to denoise the time series
    av10_cumdisp = make_av10(cumdisp)
    intervals_yr_av10 = intervals_yr[5:-5]

    # Calculate "instantaneous" ABSOLUTE velocity in the LoS direction at all dates
    inst_vel_av10 = abs((av10_cumdisp[1:] - av10_cumdisp[:-1]) / intervals_yr_av10)
    inst_vel = abs((cumdisp[1:] - cumdisp[:-1]) / intervals_yr)

    # Calculate "instantaneous" acceleration in the LoS direction at all dates
    inst_acc_av10 = (inst_vel_av10[1:] - inst_vel_av10[:-1]) / intervals_yr_av10[1:]
    inst_acc = (inst_vel[1:] - inst_vel[:-1]) / intervals_yr[1:]

    # Only keep positie acceleration as we are not interested in stuff slowing down
    inst_acc_av10[inst_acc_av10 < 0] = 0
    inst_acc[inst_acc < 0] = 0

    return cumdisp, av10_cumdisp, inst_vel, inst_vel_av10, inst_acc, inst_acc_av10


######################################################################
######################################################################
def signal2noiseratio (a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

######################################################################
######################################################################
def derivate_per_yr(arr, intervals_yr):

    darr = []
    for i in range(len(arr)):
        D = (arr[i,1:] - arr[i,:-1]) / intervals_yr
        darr.append(D)

    darr = np.asarray(darr)

    return darr

######################################################################
######################################################################

def piecewise(x,x0,x1,y0,y1,k0,k1,k2):
    return np.piecewise(x , [x <= x0, np.logical_and(x0<x, x<= x1),x>x1] , [lambda x:k0*x + y0, lambda x:k1*(x-x0)+y1+k0*x0,
                                                                            lambda x:k2*(x-x1) + y0+y1+k0*x0+k1*(x1-x0)])



######################################################################
######################################################################
def SegmentedLinearReg( X, Y, breakpoints, nIterationMax = 10):

    ramp = lambda u: np.maximum( u, 0 )
    step = lambda u: ( u > 0 ).astype(float)

    breakpoints = np.sort( np.array(breakpoints) )

    dt = np.min( np.diff(X) )
    ones = np.ones_like(X)

    for i in range( nIterationMax ):
        # Linear regression:  solve A*p = Y
        Rk = [ramp( X - xk ) for xk in breakpoints ]
        Sk = [step( X - xk ) for xk in breakpoints ]
        A = np.array([ ones, X ] + Rk + Sk )
        p =  lstsq(A.transpose(), Y, rcond=None)[0]

        # Parameters identification:
        a, b = p[0:2]
        ck = p[ 2:2+len(breakpoints) ]
        dk = p[ 2+len(breakpoints): ]

        # Estimation of the next break-points:
        newBreakpoints = breakpoints - dk/ck

        # Stop condition
        if np.max(np.abs(newBreakpoints - breakpoints)) < dt/5:
            break

        breakpoints = newBreakpoints
    else:
        print( 'maximum iteration reached' )

    # Compute the final segmented fit:
    Xsolution = np.insert( np.append( breakpoints, max(X) ), 0, min(X) )
    ones =  np.ones_like(Xsolution)
    Rk = [ c*ramp( Xsolution - x0 ) for x0, c in zip(breakpoints, ck) ]

    Ysolution = a*ones + b*Xsolution + np.sum( Rk, axis=0 )

    return Xsolution, Ysolution


######################################################################
######################################################################
def find_failure_indices(arr_av10, dates_av10, startdate, enddate, datacounter):

    failure_indices = []

    if datacounter == 0:
        threshold = 50
    if datacounter == 1:
        threshold = 10

    # USE THIS:
    # https://www.nature.com/articles/s41598-018-25369-w.pdf

    # 1. slope of linear regression over the whole displacement timeseries
    for l in range(len(arr_av10)):
        y = list(arr_av10[l]); x = np.copy(dates_av10)
        for t in range(len(x)):
            x[t] = (x[t] - startdate).total_seconds()
        x = list(x)

        fit_slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        # 2. slope of linear regression over a 150 days moving window
        hi_diff_indice = []

        for t in range(len(x)):
            limit = x[t] + 150*24*3600
            # don't exceed the end of your dataset
            if limit < x[-1]:
                end_t = np.where(np.asarray(x) < limit)[0][-1]
                Y = y[t:end_t]
                X = x[t:end_t]
                window_slope, window_intercept, r_value, p_value, std_err = stats.linregress(X,Y)
                Diff = (window_slope - fit_slope)*3600*24*365 # that's in mm/yr
                # let's say anything above 50mm/yr difference is a failure
                if abs(Diff) > threshold:
                    hi_diff_indice.append(t)
            else:
                break

        # 3. Find the consecutive indices
        # If you can string 5 together, then it's a failure
        consec = np.copy(hi_diff_indice)
        counting = 0
        for k in range(len(hi_diff_indice)-1):
            if hi_diff_indice[k+1] == hi_diff_indice[k]+1:
                consec[k] = counting
                counting +=1
                if consec[k] == 4:
                    failure_indices.append(hi_diff_indice[k-consec[k]])
            else:
                counting = 0
                consec[k] = counting

    return failure_indices

######################################################################
######################################################################
def plot_disp_failure(all_movement, all_movement_dates, all_failures, rain, slope, startdate, enddate,i,j, out_dir_csv, datasource):

    fig=plt.figure(1, facecolor='White',figsize=[7, 7])
    ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
    ax11 = ax1.twinx()

    # print the pixel slope
    ax1.annotate('pixel slope:'+str(np.round(slope,2)), xy = [0.05, 0.9], xycoords = 'axes fraction')

    #plot the rain
    ax11.fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)
    print(np.shape(all_movement))
    print(np.shape(all_failures))
    for source in range(len(datasource)):
        movement = all_movement[source]
        movement_dates = all_movement_dates[source]
        failures = all_failures[source]
        print("source: {}".format(source))
        for k in range(len(movement)):
            print(k)
            if len(movement[k]) > 0:
                # red is cosmo sky med and blue is sentinel
                ax1.plot(movement_dates, movement[k], '-', c = plt.cm.jet(255*source), lw = 0.8)
                print(movement_dates)
                print(movement[k])
                print(i,j)

        if len(failures) > 0:
            for f in failures:
                ax11.scatter(movement_dates[f], 0, marker = 'o', facecolor = plt.cm.jet(255*source))



    if source == 0:
        ax1.set_xlim(left = startdate, right = enddate)
        ax1.set_xlabel('Time (yr)')
        ax1.set_ylabel('Precipitation (mm)')

    ax1.set_ylabel('Cumulative Displacement (mm)')

    plt.savefig(out_dir_csv+'TESTGroundMotion_pixel'+str(i)+'_'+str(j)+'_failure.png')

######################################################################
# saves the ground motion timeseries for each pixel
######################################################################
def save_disp_failure_csv(all_movement, all_movement_dates, slope, i,j, out_dir_csv, datasource):
    ground_motion_df= pd.DataFrame(columns=['ground_motion','time_of_motion','slope','row','col'])

    for source in range(len(datasource)):
        movement = all_movement[source]
        movement = movement.squeeze()
        movement_dates = all_movement_dates[source]

        for k in range(len(movement)):
            ground_motion_df = ground_motion_df.append({'ground_motion':movement[k], 'time_of_motion':movement_dates[k],'slope':slope,'row':i,'col':j, 'datasource':datasource}, ignore_index=True)
            ground_motion_df.to_csv(out_dir_csv + 'TEST_Timeseries_GroundMotion_pixel'+str(i)+'_'+str(j)+'_failure.csv', index=False)

def save_disp_failure_csv_updated(all_movement, all_movement_dates, slope, curvature, aspect, i,j, out_dir_csv, datasource):
    ground_motion_df= pd.DataFrame(columns=['ground_motion','time_of_motion','slope','curvature','aspect','row','col'])

    for source in range(len(datasource)):
        movement = all_movement[source]
        movement_dates = all_movement_dates[source]

        for k in range(len(movement)):
            movement = np.array(movement)
            movement_dates = np.array(movement_dates)
            movement_dates = movement_dates.squeeze()
            if len(movement[k]) > 0:
                for m in range(np.shape(movement)[1]):

                    ground_motion_df = ground_motion_df.append({'ground_motion':movement[k,m], 'time_of_motion':movement_dates[m],'slope':slope, 'curvature':curvature, 'aspect':aspect,'row':i,'col':j, 'datasource':source}, ignore_index=True)
                    ground_motion_df.to_csv(out_dir_csv + '10mDEM_Timeseries_GroundMotion_pixel'+str(i)+'_'+str(j)+'_failure.csv', index=False)




######################################################################
######################################################################
def plot_dva(all_movement, all_movement_dates, rain, slope, startdate, enddate,i,j, out_dir, datasource):

    fig=plt.figure(1, facecolor='White',figsize=[7, 7])
    ax1 =  plt.subplot2grid((3,1),(0,0),colspan=1, rowspan=1)
    ax2 =  plt.subplot2grid((3,1),(1,0),colspan=1, rowspan=1)
    ax3 =  plt.subplot2grid((3,1),(2,0),colspan=1, rowspan=1)
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    ax33 = ax3.twinx()

    axes = [ax1, ax2, ax3]
    axess = [ax11, ax22, ax33]

    # print the pixel slope
    ax1.annotate('pixel slope:'+str(np.round(slope,2)), xy = [0.05, 0.9], xycoords = 'axes fraction')

    for source in range(len(datasource)):
        movement = all_movement[source]
        movement_dates = all_movement_dates[source]

        for k in range(len(movement)):

            #plot the rain
            axess[k].fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)

            #plot movement
            movk = movement[k]
            for l in range(len(movk)):
                if len(movk[l]) > 0:
                    axes[k].plot(movement_dates[k], movk[l], '-', c = plt.cm.jet(255*source), lw = 0.8)

            if source == 0:
                axes[k].set_xlim(left = startdate, right = enddate)
                axes[k].set_xlabel('Time (yr)')
                axess[k].set_ylabel('Precipitation (mm)')


    ax1.set_ylabel('Cumulative Displacement (mm)')
    ax2.set_ylabel('Velocity (mm/yr)')
    ax3.set_ylabel('Acceleration (mm/yr2)')

    plt.savefig(out_dir+'GroundMotion_pixel'+str(i)+'_'+str(j)+'.png')



######################################################################
######################################################################
def plot_dva_linfit(all_movement, all_movement_dates, fit, rain, slope, startdate, enddate,i,j, out_dir, datasource):

    fig=plt.figure(1, facecolor='White',figsize=[7, 7])
    ax1 =  plt.subplot2grid((3,1),(0,0),colspan=1, rowspan=1)
    ax2 =  plt.subplot2grid((3,1),(1,0),colspan=1, rowspan=1)
    ax3 =  plt.subplot2grid((3,1),(2,0),colspan=1, rowspan=1)
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    ax33 = ax3.twinx()

    axes = [ax1, ax2, ax3]
    axess = [ax11, ax22, ax33]

    # print the pixel slope
    ax1.annotate('pixel slope:'+str(np.round(slope,2)), xy = [0.05, 0.9], xycoords = 'axes fraction')

    for source in range(len(datasource)):
        movement = all_movement[source]
        movement_dates = all_movement_dates[source]

        for k in range(len(movement)):

            #plot the rain
            axess[k].fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)

            #plot movement
            movk = movement[k]
            for l in range(len(movk)):
                if len(movk[l]) > 0:
                    axes[k].plot(movement_dates[k], movk[l], '-', c = plt.cm.jet(255*source), lw = 0.8)

            if source == 0:
                axes[k].set_xlim(left = startdate, right = enddate)
                axes[k].set_xlabel('Time (yr)')
                axess[k].set_ylabel('Precipitation (mm)')

    # plot linear fit on dsplacement axis
    A = (startdate - startdate).total_seconds() * fit[0] + fit[1]
    B = (enddate - startdate).total_seconds() * fit[0] + fit[1]

    ax1.plot([startdate, enddate], [A,B], '-k', lw = 2)


    ax1.set_ylabel('Cumulative Displacement (mm)')
    ax2.set_ylabel('Velocity (mm/yr)')
    ax3.set_ylabel('Acceleration (mm/yr2)')

    plt.savefig(out_dir+'GroundMotion_pixel'+str(i)+'_'+str(j)+'.png')


######################################################################
######################################################################
def plot_movement(s_av10, dates_s_av10, ewv_av10, dates_ewv_av10, rain, slope, startdate, enddate,i,j, out_dir, movement_type):
    fig=plt.figure(1, facecolor='White',figsize=[7, 7])
    ax1 =  plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
    ax11 = ax1.twinx()

    fontred = {'family': 'serif', 'color':  'red', 'weight': 'normal', 'size': 12 }

    fontblue = {'family': 'serif', 'color':  'blue', 'weight': 'normal', 'size': 12 }

    # print the pixel slope
    ax1.annotate('pixel slope:'+str(np.round(slope,2)), xy = [0.05, 0.95], xycoords = 'axes fraction')
    # ax1.annotate(r'$Sentinel$', xy = [0.55, 0.95], xycoords = 'axes fraction', fontdict = fontred)
    #ax1.annotate(r'$CosmoSkyMed$', xy = [0.75, 0.95], xycoords = 'axes fraction', facecolor = 'r')
    ax1.text(0.55, 0.95, 'Sentinel', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color="blue")
    ax1.text(0.75, 0.95, 'Cosmo-SkyMed', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color="red")

    #plot the rain
    ax11.fill_between(rain['time'], 0, rain['rainfall_mm'], facecolor = 'k', lw = 0.1, alpha = 0.8)

    print (len(s_av10), len(ewv_av10))

    if len(s_av10) > 0:
        for k in range(len(s_av10)):
            ax1.plot(dates_s_av10, s_av10[k], '-', c = plt.cm.jet(0), lw = 0.8)

    if len(ewv_av10) > 0:
        for k in range(len(ewv_av10)):
            ax1.plot(dates_ewv_av10, ewv_av10[k], '-+', c = plt.cm.jet(255), lw = 0.8)

    ax1.set_xlim(left = startdate, right = enddate)

    ax1.set_xlabel('Time (yr)')
    if movement_type == "Displacement":
        ax1.set_ylabel('Cumulative Displacement (mm)')
    elif movement_type == 'Velocity':
        ax1.set_ylabel('Veloicty (mm/yr)')
    elif movement_type == "Acceleration":
        ax1.set_ylabel('Acceleration (mm/yr2)')
    ax11.set_ylabel('Precipitation (mm)')



    plt.savefig(out_dir+'GroundMotion_pixel'+str(i)+'_'+str(j)+'_'+movement_type+'.png')



######################################################################
######################################################################

def maps_to_timeseries(working_dir, arglist, output_dir):
    # List the .bil files
    files = sorted(os.listdir(working_dir)); bilfiles = []
    for i in range(len(files)):
        ending = files[i][-4:]
    if ending == '.bil':
        bilfiles.append(files[i])

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
    # Save it all together
    Full_list.append([30*60, Intensity])

    # Now save it
    with open(output_dir+'/'+arglist[1]+"_to_"+arglist[2]+"_Intensity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['duration_s','intensity_mm_sec'])
        writer.writerows(Full_list)
    print ('saved file:', output_dir+arglist[1]+"_to_"+arglist[2]+"_Intensity.csv")




################################################################################
################################################################################

def plot_dva_failure(i, rain, dates, disp, av10_disp, inst_vel, inst_vel_av10, inst_acc, inst_acc_av10, S_startdate, S_enddate, Tfail, Tprefail, th, out_dir):
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

    ax1.set_ylabel('Cumulative Displacement (mm)')
    ax2.set_ylabel('Abs. velocity (m/yr)')
    ax3.set_ylabel('Displacement acceleration (m/yr2)')

    plt.savefig(out_dir+str(i)+'.png')
