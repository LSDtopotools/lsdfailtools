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

from .image_functions import ENVI_raster_binary_to_2d_array, ENVI_raster_binary_from_2d_array

from shapely.ops import nearest_points
from shapely import wkt
import pyproj

from shapely.geometry import Point
from shapely.ops import transform
from .run_json import *


### IMPORTANT NOTE ### REMEMBER THAT SOME POINTS HAVE TWO VALUES, THIS IS WHY
### THE NUMBER OF ROWS IN THE CALIBRATION FILE DOESN'T MATCH THE NUMBER OF
### COORDINATE POINTS
def convert_calib_to_lat_lon(dem_raster, calib_csv, output_calib_raster):
    """
    convert_calib_to_lat_lon creates a raster image from the points of a numpy
    array corresponding to lat, lon coordinates of the calibrated points. Non-calibrated
    points have nan values.
    :dem_raster: raster used to extract parameters and coordinates for transformation
    and projection of array points
    :calib_csv: csv with the parameters for the calibrated points. Has a column with the x,y array
    position of the points.
    :output_calib_raster: raster with the calibrated points projected onto the
    dem_raster. Non-calibrated points have nan values.
    """

    dem_file = dem_raster

    # 0. Load rasters into arrays
    demarr, pixelWidth, (geotransform, inDs) = ENVI_raster_binary_to_2d_array(dem_file)

    calibration_file = calib_csv

    # need to convert the csv into a geodataframe
    input_df = pd.read_csv(calibration_file)
    input_array = 0*demarr
    input_array[:] = np.NaN
    for i in range (len(input_df)):
        x = int(input_df['row'].iloc[i])
        y = int(input_df['col'].iloc[i])
        input_array[x,y] = input_df.index[i]

    #input_array[input_array == 0] = 'nan'

    # convert the row,col of the calibration file into lat, lon coordinates
    new_geotransform,new_projection,file_out = ENVI_raster_binary_from_2d_array( (geotransform, inDs), output_calib_raster, pixelWidth, input_array)
    print('I am done. The calibration file now has lon, lat coordinates.')

# Convert the raster file with the calibrated points to a shapefile + csv file

# get the name of the file from the full filepath and extension
def convert_calib_raster_to_csv_shp(calib_raster):
    """
    convert_calib_raster_to_csv_shp converts the raster to a XYZ point geometry
    shapefile/csv object
    :calib_raster: raster file to be converted
    """
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

def create_calib_multipoint(calib_csv_point_transformed):
    """
    create_calib_multipoint converts XYZ point geometry object into a multipoint
    object with only calibrated points
    :param calib_csv_point_transformed: csv with the point geometry geometry objects
    of the raster file containing the calibrated points
    :returns:
        - multipoint - shapely multipoint object of the calibrated points
        - selected_rows - row numbers of calibrated points (based on input csv file)
    """

    # Load the .csv file with all the calibrated points in point geometry
    #file_path = os.path.dirname(calib_csv_point_transformed)
    #filename_full = os.path.basename(calib_csv_point_transformed)

    input_df = pd.read_csv(calib_csv_point_transformed, ' ')
    #input_df = pd.read_csv(file_path + '/' + filename_full, ' ')

    # remove the rows that are NaN
    selected_rows = input_df[~input_df['Z'].isnull()]
    selected_rows_indices = selected_rows.index

    # convert the columns into shapely Point Objects
    geometry = [Point(xy) for xy in zip(selected_rows.X, selected_rows.Y)]
    geo_df = gpd.GeoDataFrame(selected_rows, geometry=geometry)

    geo_df = geo_df.drop(columns=['X', 'Y', 'Z'])
    # create a multipoint object - need this to use the nearest_points function
    multipoint = geo_df.geometry.unary_union
    return multipoint, selected_rows


def calib_params_closest_point(multipoint, selected_rows, lat_lon_file, calib_file, outfile):
    """
    calib_params_closest_point finds the closest calibration points to the lat,lon
    test points in the area of interest.
    :param multipoint: shapely multipoint object of the calibrated points
    :param selected_rows: row numbers of calibrated points (based on input csv file)
    :param lat_lon_file: csv file with the point geometry objects of the test points in
    the area of interest
    :param calib_file: csv file with the parameters for each calibrated point
    :param outfile: name of csv file with the parameters of the closest calibration points
    to the test points
    """
    points_df = pd.read_csv(lat_lon_file, index_col = None)

    full_calibration_df = pd.read_csv(calib_file, index_col = None)

    closest_points_df = pd.DataFrame(columns=full_calibration_df.columns[1:])
    # add extra lat lon columns
    closest_points_df['lat_test'] = ""
    closest_points_df['lon_test'] = ""
    closest_points_df['lat_calib'] = ""
    closest_points_df['lon_calib'] = ""
    points_df['geometry'] = points_df['geometry'].apply(wkt.loads)
    points_gdf = gpd.GeoDataFrame(points_df, crs='epsg:4326')

    for i in range(len(points_df)):
        #point = Point(515854, 4.551284e+06)
        wgs84_pt = points_gdf['geometry'][i]
        # reproject point
        AoI_proj = pyproj.CRS('EPSG:32633')
        wgs84 = pyproj.CRS('EPSG:4326')
        project =  pyproj.Transformer.from_crs(wgs84, AoI_proj, always_xy=True).transform
        AoI_point = transform(project, wgs84_pt)

        # this prints out the point closest to our list of points
        closest_cal_point = nearest_points(multipoint, AoI_point)[0]

        # note that Z is not the altitude but the row number in the initial dataframe... need to figure out a better way to factor this in.
        calibration_params_index = list(selected_rows[selected_rows['X']==closest_cal_point.x].index.values)
        calibration_parameters_selection = int(selected_rows['Z'][calibration_params_index])

        calibration_parameters = full_calibration_df.iloc[[calibration_parameters_selection]]

        # save the parameters to a new file with only the point we want
        calibration_parameters = calibration_parameters.drop(calibration_parameters.columns[[0]], axis=1)

        closest_points_df.loc[i] = calibration_parameters.iloc[0]
        closest_points_df['lat_test'].loc[i]= AoI_point.y
        closest_points_df['lon_test'].loc[i] = AoI_point.x
        closest_points_df['lat_calib'].loc[i] = closest_cal_point.y
        closest_points_df['lon_calib'].loc[i] = closest_cal_point.x
    closest_points_df.to_csv(outfile,index=False)

def convert_crs_point(point_x, point_y, in_proj, out_proj):
    """
    convert_crs_point reprojects a point into a different coordinate system
    :param point_x: latitude value of the point
    :param point_y: longitude value of the point
    :param in_proj: input coordinate system
    :param out_proj: output coordinate system
    :returns:
        - AoI_point - point object in the new coordinate system
    """
    in_pt = Point(point_x, point_y)
    #wgs84_pt = points_gdf['geometry'][i]
    # reproject point
    in_proj = pyproj.CRS(in_proj)
    out_proj = pyproj.CRS(out_proj)
    project =  pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True).transform
    AoI_point = transform(project, in_pt)
    return AoI_point

def how_close_is_calibrated_point(point_1, point_2, buffer_distance, point_number):
    """
    how_close_is_calibrated_point calculates whether 2 points are within a
    certain distance of each other
    :param point_1: first point object
    :param point_2: second point object
    :param buffer_distance: distance (meters) for the buffer area starting from
    point 1
    :param point_number: number of the point to evaluate
    :returns:
        - bool - True if point_2 is within buffer distance of point_1. False otherwise.
        - meters_distance - distance between point_1 and point_2.
    """
    # create your circle buffer from one of the points - is this in km?
    distance = buffer_distance
    circle_buffer = point_1.buffer(distance)
    # but a simpler method is to simply check the distance
    meters_distance = point_1.distance(point_2)
    if point_1.distance(point_2) < distance:
        print(f'Test Point {point_number} is within {distance} m of calibrated Point {point_number}')
        return True, meters_distance
    else:
        print(f'Test Point {point_number} is not within {distance} m of calibrated Point {point_number}')
        return False, meters_distance

def find_points_within_buffer_distance(point_to_test, calib_multipoints, buffer_distance_meters):
    """
    find_points_within_buffer_distance calculates which points from a multipoint
    object are within a certain buffer distance of a given point
    :param point_to_test: point to take the buffer distance from
    :param calib_multipoints: multipoint object with the location of the points to test
    :param buffer_distance_meters: distance (meters) for the buffer area starting from point_to_test
    :returns:
        - points_in_buffer - points with from the calib_multipoints object that lie within
        the buffer distance of point_to_test
    """
    circle = point_to_test.buffer(buffer_distance_meters)
    points_in_buffer = []
    for p in calib_multipoints:
        if circle.covers(p):
            point_in_buffer = Point(p.x,p.y)
            points_in_buffer.append(point_in_buffer)
    return points_in_buffer


# convert calib_points to the same coord system as the test points
def create_list_test_calib_test_points(calib_df, test_df):
    """
    create_list_test_calib_test_points convert the dataframes of points to list
    shapely point objects
    :param calib_df: pandas dataframe with the parameters of the closest calibration points
    to the test points
    :param test_df: pandas geodataframe with the point locations of the test points in the
    area of interest
    :returns:
        - test_points_list - list of point objects corresponding to the test points
        - calib_points - list of point objects corresponding to the calibrated points

    """
    calib_points = []
    test_points_list = []
    for i in range(len(calib_df)):
        calib_y = int(calib_df['lat_calib'][i])
        calib_x = int(calib_df['lon_calib'][i])
        calib_point = convert_crs_point(calib_x, calib_y, 'epsg:32633', 'epsg:32633')
        test_y = test_df['geometry'][i].y
        test_x = test_df['geometry'][i].x
        test_point = convert_crs_point(test_x, test_y, 'epsg:4326', 'epsg:32633')
        calib_points.append(calib_point)
        test_points_list.append(test_point)
    return test_points_list, calib_points

def get_points_in_buffer_distance(in_file_calib, out_file, boolean_file_test):
    """
    get_points_in_buffer_distance creates a csv file with the distances between the
    test point and the calibrated points **UNFINISHED**
    :param in_file_calib: csv file with the parameters of the closest calibration points
    to the test points
    :param out_file: name of csv file to save the distance (m) from the calibrated
    to the test point
    :param boolean_file_test: point object csv file for the test points in the area of interest
    """
    #########
    ##### UNFINISHED????? #####
    # need the following conversion as the input of the create_list_test_calib_test_points function
    # the x,y test points are taken from the bool_lat_lon file which has the
    # geometry column in POINT geometry form.
    test_points = pd.read_csv(boolean_file_test)
    test_points['geometry'] = test_points['geometry'].apply(wkt.loads)
    test_points_gdf = gpd.GeoDataFrame(test_points, crs='epsg:4326')

    calib_points_df = pd.read_csv(in_file_calib)#, index=None)
    #########

    test_points_list, calib_points = create_list_test_calib_test_points(calib_points_df, test_points_gdf)

    # checks if the closest point is within the buffer distance
    # if if isnt within the buffer distance, we don't care about that point
    # if it is within the buffer distance, then we calculate if there are other points
    # within that buffer distance

    is_it_within_buffer_distance = []
    meters_distance_buffer = []
    for i in range(len(test_points_gdf)):
        true_or_false, meters_distance = how_close_is_calibrated_point(test_points_list[i], calib_points[i], 1000, i)
        is_it_within_buffer_distance.append(true_or_false)
        meters_distance_buffer.append(meters_distance)

    meters_distance_buffer = np.asarray(meters_distance_buffer)

    pd.DataFrame(meters_distance_buffer, columns=['distance_m_from_calib_to_test_point']).to_csv(out_file, index = None)
