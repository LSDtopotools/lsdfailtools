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

print("done")


# convert the numpy array with the pixel locations into a georeferenced raster
# we need the locations of the ground motion instances in terms of lat/lot
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

# convert the csv files with the x,y pixel coordinates into x,y coordinates
#
#ENVI_raster_binary_from_2d_array( (geotransform, inDs), faildir+"test_convert_array_to_raster.bil", pixelWidth, valid_array)

'''
filename = faildir+'points_validation_raster'
inDs = gdal.Open('{}.tif'.format(filename))
outDs = gdal.Translate('{}.xyz'.format(filename), inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
outDs = None
try:
    os.remove('{}.csv'.format(filename))
except OSError:
    pass
os.rename('{}.xyz'.format(filename), '{}.csv'.format(filename))
#os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=NO {0}.shp {0}.csv'.format(filename))
'''

points_csv_file = faildir+'points_validation_raster.csv'
points_csv_df = pd.read_csv(points_csv_file)
print(points_csv_df.head(5))
#points_csv_df["XYZ"] = points_csv_df["XYZ"].str.replace(" ",",")
points_csv_df[['X', 'Y', 'Z']] = points_csv_df['X Y Z'].str.split(' ', 2, expand=True)
points_csv_df = points_csv_df.drop('X Y Z', 1)
points_csv_df = points_csv_df.drop('Z', 1)
#points_csv_df = pd.to_numeric(points_csv_df)
points_csv_df["X"] = points_csv_df["X"].astype(str).astype(float)
points_csv_df["Y"] = points_csv_df["Y"].astype(str).astype(float)
print(points_csv_df.dtypes)



#geometry = [Point(xy) for xy in zip(points_csv_df.X, points_csv_df.Y)]
# add time of failure associated to each of the points
df = points_csv_df.drop(['X', 'Y'], axis=1)
points_csv_df['time_of_failure'] = valid_df['time_of_failure']
#gdf = GeoDataFrame(df, crs="EPSG:32633", geometry=geometry)


gdf = gpd.GeoDataFrame(points_csv_df, crs="EPSG:32633",geometry=gpd.points_from_xy(points_csv_df.X, points_csv_df.Y))
print(gdf.head(5))
# save the GeoDataFrame
#gdf.to_file(driver = 'ESRI Shapefile', filename= faildir+"validation_point_locations.shp")


lsoas_link = faildir+"validation_point_locations.shp"
lsoas = gpd.read_file(lsoas_link)
plt.plot(lsoas)

#points_csv_df = points_csv_df[0].split(' ')
#points_csv_df.head(5)
quit()




#######################
# Map calibrated points
#ff.map_calibrated (demarr, calibrated, line, 10, 15, fig_out_dir + 'Map_calibrated_pixels.png')
