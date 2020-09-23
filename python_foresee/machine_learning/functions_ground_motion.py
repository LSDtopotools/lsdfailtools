################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import os
import re
import json
import datetime
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt

import sys
import argparse
import time
import shutil
import tkinter
from tkinter import filedialog
import platform
import argparse
import datetime
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import csv

import lsdfailtools.iverson2000 as iverson

import matplotlib.lines as mlines
from itertools import product

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


def make_pxl_csv(ground_motion_dir):

    pxl_df = pd.DataFrame(columns=['rows','cols'])
    # Get the pixel coordinates from the file names
    regex_pxl_values = re.compile(r'[0-9]+_[0-9]+')

    directory = os.fsencode(ground_motion_dir)

    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         if filename.endswith(".csv"):
             pixel_values = regex_pxl_values.findall(filename)
             for item in pixel_values:
                 # split string by _
                pixels = item.split('_')
                pxl_df = pxl_df.append({'rows': pixels[0],'cols': pixels[1]}, ignore_index=True)

             continue
         else:
             continue

    pxl_values_file = pxl_df.to_csv(ground_motion_dir + "pixel_values.csv")
