#####
# File to take in a list of points and find where the closest calibrated point is.
# Find the parameters of the closest calibrated point.

# Author: Marina Ruiz Sanchez-Oro
# Date: 10/11/2021
#####

import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import pandas as pd
import os
import sys
import image_functions as fn
from shapely.ops import nearest_points
from shapely import wkt
import pyproj

from shapely.geometry import Point
from shapely.ops import transform


### IMPORTANT NOTE ### REMEMBER THAT SOME POINTS HAVE TWO VALUES, THIS IS WHY
### THE NUMBER OF ROWS IN THE CALIBRATION FILE DOESN'T MATCH THE NUMBER OF
### COORDINATE POINTS
def convert_calib_to_lat_lon(dem_raster, calib_csv, output_calib_raster):
    dem_file = dem_raster

    # 0. Load rasters into arrays
    demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(dem_file)

    calibration_file = calib_csv

    # need to convert the csv into a geodataframe
    input_df = pd.read_csv(calibration_file)
    input_array = 0*demarr
    for i in range (len(input_df)):
        x = int(input_df['row'].iloc[i])
        y = int(input_df['col'].iloc[i])
        input_array[x,y] = input_df.index[i] + 1

    input_array[input_array == 0] = 'nan'

    # convert the row,col of the calibration file into lat, lon coordinates
    new_geotransform,new_projection,file_out = fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), output_calib_raster, pixelWidth, input_array)
    print('I am done. The calibration file now has lon, lat coordinates.')

# The following is useful code. KEEP IT!! - just need to uncomment it
# Convert the raster file with the calibrated points to a shapefile + csv file

# get the name of the file from the full filepath and extension
def convert_calib_raster_to_csv_shp(calib_raster):
    filename_full = os.path.basename(calib_raster)
    filename = filename_full.split('.')
    filename = filename[0]

    inDs = gdal.Open(calib_raster)
    outDs = gdal.Translate('{}.xyz'.format(filename), inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
    outDs = None
    try:
        os.remove('{}.csv'.format(filename))
    except OSError:
        pass
    os.rename('{}.xyz'.format(filename), '{}.csv'.format(filename))
    os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=NO {0}.shp {0}.csv'.format(filename))


# This has converted the raster to a csv with the xy coordinates and the Z values at the calibrated points.
## We want to split the data now so that we have a lat, lon, Z column and it's easier to sort the data that is not NaN.

def create_calib_multipoint(calib_csv):
    # Load the .csv file with all the calibrated points in point geometry
    file_path = os.path.dirname(calib_csv)
    filename_full = os.path.basename(calib_csv)

    #input_df = pd.read_csv('./test_transform.csv', ' ')
    input_df = pd.read_csv(file_path + '/' + filename_full, ' ')

    # remove the rows that are NaN
    selected_rows = input_df[~input_df['Z'].isnull()]
    selected_rows_indices = selected_rows.index
    print(selected_rows.index)

    # convert the columns into shapely Point Objects
    geometry = [Point(xy) for xy in zip(selected_rows.X, selected_rows.Y)]
    geo_df = gpd.GeoDataFrame(selected_rows, geometry=geometry)
    print(geo_df.head)

    geo_df = geo_df.drop(columns=['X', 'Y', 'Z'])
    # create a multipoint object - need this to use the nearest_points function
    multipoint = geo_df.geometry.unary_union
    return multipoint, selected_rows


# test point - need to change this to be the output from the lat_lon_area_check.py script.
# need to also be able to take a list of points instead of just one.

def calib_params_closest_point(lat_lon_file, calib_file):
    points_df = pd.read_csv(lat_lon_file, index_col = None)
    #point_one = points_df['geometry'][0]

    full_calibration_df = pd.read_csv(calib_file, index_col = None)

    closest_points_df = pd.DataFrame(columns=full_calibration_df.columns[1:])
    # add extra lat lon columns
    closest_points_df['lat_test'] = ""
    closest_points_df['lon_test'] = ""
    points_df['geometry'] = points_df['geometry'].apply(wkt.loads)
    points_gdf = gpd.GeoDataFrame(points_df, crs='epsg:4326')

    for i in range(len(points_df)):
        #point = Point(515854, 4.551284e+06)
        wgs84_pt = points_gdf['geometry'][i]
        print(wgs84_pt.x, wgs84_pt.y)
        # reproject point - MAYBE PUT THIS IN A SEPARATE FUNCTION
        AoI_proj = pyproj.CRS('EPSG:32633')
        wgs84 = pyproj.CRS('EPSG:4326')
        project =  pyproj.Transformer.from_crs(wgs84, AoI_proj, always_xy=True).transform
        AoI_point = transform(project, wgs84_pt)
        print(AoI_point.x, AoI_point.y)


        # this prints out the point closest to our list of points
        closest_cal_point = nearest_points(multipoint, AoI_point)[0]
        print(closest_cal_point, nearest_points(multipoint, AoI_point)[0])
        #print(closest_cal_point.x)

        # note that Z is not the altitude but the row number in the initial dataframe... need to figure out a better way to factor this in.
        calibration_params_index = list(selected_rows[selected_rows['X']==closest_cal_point.x].index.values)
        calibration_parameters_selection = int(selected_rows['Z'][calibration_params_index])

        calibration_parameters = full_calibration_df.iloc[[calibration_parameters_selection]]

        # save the parameters to a new file with only the point we want
        #calibration_parameters = calibration_parameters.drop[0]
        calibration_parameters = calibration_parameters.drop(calibration_parameters.columns[[0]], axis=1)

        #calibration_parameters[0]
        print(calibration_parameters.iloc[0])
        closest_points_df.loc[i] = calibration_parameters.iloc[0]
        print(closest_points_df)
        closest_points_df['lat_test'].loc[i]= AoI_point.y
        closest_points_df['lon_test'].loc[i] = AoI_point.x
    closest_points_df.to_csv('./test_closest_calibration_points.csv',index=False)


# convert_calib_to_lat_lon('/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_AoI_epsg32633.bil',\
# '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv',\
# './_test_transform.bil')
#
# convert_calib_raster_to_csv_shp('./_test_transform.bil')
#
multipoint, selected_rows = create_calib_multipoint('./test_transform.csv')


calib_params_closest_point('./bool_lat_lon.csv', '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv')
