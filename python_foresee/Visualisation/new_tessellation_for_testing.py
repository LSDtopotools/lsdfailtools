# Importing the model
#import lsdfailtools.iverson2000 as iverson

# I'll need that to process the outputs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import product
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import pandas as pd
import numpy as np
import shapefile
import itertools
import json
import os
import rasterio
from rasterio.features import shapes

#MR: i´m assuming for now this is InSAR Insar_functions
import sys
sys.path.insert(0,'../Alldata_processing/InSAR')
import Insar_functions as fn
 #import functions as fn

import Figure_functions as ff


######################################################
######################################################
# set up stuff
######################################################
######################################################

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)


# Model directory
rundir = FILE_PATHS["rain_intensity_caliv_valid"]

# Setting the depth resolution vector
depths = np.arange(0.2,3.1,0.1)

#failure threshold
threshold = 80 # mm/yr




# failure data files
faildir = FILE_PATHS["interferometry_out_dir"]
failfile = faildir + "All_1st_failtime__threshold"+str(threshold)+"mmyr.bil"
prefailfile = faildir + "All_1st_prefailtime__threshold"+str(threshold)+"mmyr.bil"

# topography files
topodir = FILE_PATHS["topo_dir"]
demfile = topodir + "eu_dem_AoI_epsg32633.bil"
slopefile = topodir + "eu_dem_AoI_epsg32633_SLOPE.bil"

# road files
roaddir = FILE_PATHS["road_dir"]
roadfile = roaddir + "Road_line.shp"

# calibrated points files
# need to check if this is the right directory
calibdir = FILE_PATHS["rain_intensity_caliv_valid"]
calibfile = calibdir + "Calibrated_all.csv"

rainfile = FILE_PATHS["rain_dir"]
fig_out_dir = FILE_PATHS["figures_dir"]

validdir = FILE_PATHS["rain_intensity_caliv_valid"]
validfile = validdir +"Validated_updated.csv"

Cal_params_dir = FILE_PATHS["rain_intensity_caliv_valid"]
Cal_params_file = Cal_params_dir+"Calibration_parameters.csv"

######################################################
######################################################
# See which points were calibrated
######################################################
######################################################

# 0. Load rasters into arrays for DEM, slope, failtimes for a given failure threshold. Let's use 80mm/yr for now.
demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(demfile)
slopearr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(slopefile)
failarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(failfile)

#y_obs_gm = pd.read_csv(y_obs_gm, header = None)

# need to convert the csv into a geodataframe
valid_df = pd.read_csv(validfile)
print(valid_df.head(5))

valid_array = 0*demarr
for i in range (len(valid_df)):
#for i in range(5):
    x = int(valid_df.iloc[i,-3])
    y = int(valid_df.iloc[i,-2])
    #print(y)
    valid_array[x,y] = valid_df.iloc[i,-6]
    #print(valid_array[x,y])
valid_array[valid_array == 0] = 'nan'
print("done")
print(valid_array[0,0])
print(np.shape(valid_array))

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





# convert the csv files with the x,y pixel coordinates into x,y coordinates

#new_geotransform,new_projection,file_out = ENVI_raster_binary_from_2d_array( (geotransform, inDs), faildir+"test_convert_array_to_raster.bil", pixelWidth, valid_array)
'''
image_array, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_2d_array(file_out)

flattened_image = image_array.flatten()
print(np.shape(flattened_image))
print(flattened_image[:5])
'''
'''
filename = faildir+'test_convert_array_to_raster'
inDs = gdal.Open('{}.bil'.format(filename))
outDs = gdal.Translate('{}.xyz'.format(filename), inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"], noData = np.nan)
outDs = None
try:
    os.remove('{}.csv'.format(filename))
except OSError:
    pass
os.rename('{}.xyz'.format(filename), '{}.csv'.format(filename))

os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=YES {0}.shp {0}.csv'.format(filename))

'''

shp_file = gpd.read_file(faildir+"test_convert_array_to_raster.shp")
print(shp_file.head(5))
#shp_file = shp_file[shp_file.Z != np.nan]

#shp_file = shp_file.dropna(inplace = True)
shp_file['Z'] = shp_file['Z'].astype('float64')
print(shp_file.dtypes)
shp_file = shp_file[shp_file.Z.notnull()]
shp_file = shp_file.dropna()
#shp_file.to_file(driver = 'ESRI Shapefile', filename= faildir+"result.shp")
print(shp_file.head(5))
'''
topodir = FILE_PATHS["topo_dir"]
AoI = topodir + "AoI.shp"

# Import area of interest in Italy for region clipping
AoI = gpd.read_file(topodir + "AoI.shp")
'''
