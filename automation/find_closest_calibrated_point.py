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
    input_array[:] = np.NaN
    for i in range (len(input_df)):
        x = int(input_df['row'].iloc[i])
        y = int(input_df['col'].iloc[i])
        input_array[x,y] = input_df.index[i]

    #input_array[input_array == 0] = 'nan'

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

    # convert the columns into shapely Point Objects
    geometry = [Point(xy) for xy in zip(selected_rows.X, selected_rows.Y)]
    geo_df = gpd.GeoDataFrame(selected_rows, geometry=geometry)

    geo_df = geo_df.drop(columns=['X', 'Y', 'Z'])
    # create a multipoint object - need this to use the nearest_points function
    multipoint = geo_df.geometry.unary_union
    return multipoint, selected_rows


# test point - need to change this to be the output from the lat_lon_area_check.py script.
# need to also be able to take a list of points instead of just one.

def calib_params_closest_point(multipoint, selected_rows, lat_lon_file, calib_file, outfile):
    points_df = pd.read_csv(lat_lon_file, index_col = None)
    #point_one = points_df['geometry'][0]

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
        #print(wgs84_pt.x, wgs84_pt.y)
        # reproject point - MAYBE PUT THIS IN A SEPARATE FUNCTION
        AoI_proj = pyproj.CRS('EPSG:32633')
        wgs84 = pyproj.CRS('EPSG:4326')
        project =  pyproj.Transformer.from_crs(wgs84, AoI_proj, always_xy=True).transform
        AoI_point = transform(project, wgs84_pt)
        #print(AoI_point.x, AoI_point.y)


        # this prints out the point closest to our list of points
        closest_cal_point = nearest_points(multipoint, AoI_point)[0]
        #print(closest_cal_point, nearest_points(multipoint, AoI_point)[0])
        #print(closest_cal_point.x)

        # note that Z is not the altitude but the row number in the initial dataframe... need to figure out a better way to factor this in.
        calibration_params_index = list(selected_rows[selected_rows['X']==closest_cal_point.x].index.values)
        calibration_parameters_selection = int(selected_rows['Z'][calibration_params_index])

        calibration_parameters = full_calibration_df.iloc[[calibration_parameters_selection]]

        # save the parameters to a new file with only the point we want
        #calibration_parameters = calibration_parameters.drop[0]
        calibration_parameters = calibration_parameters.drop(calibration_parameters.columns[[0]], axis=1)

        #calibration_parameters[0]
        #print(calibration_parameters.iloc[0])
        closest_points_df.loc[i] = calibration_parameters.iloc[0]
        #print(closest_points_df)
        closest_points_df['lat_test'].loc[i]= AoI_point.y
        closest_points_df['lon_test'].loc[i] = AoI_point.x
        closest_points_df['lat_calib'].loc[i] = closest_cal_point.y
        closest_points_df['lon_calib'].loc[i] = closest_cal_point.x
    closest_points_df.to_csv(outfile,index=False)

def convert_crs_point(point_x, point_y, in_proj, out_proj):
    in_pt = Point(point_x, point_y)
    #wgs84_pt = points_gdf['geometry'][i]
    #print(wgs84_pt.x, wgs84_pt.y)
    # reproject point - MAYBE PUT THIS IN A SEPARATE FUNCTION
    in_proj = pyproj.CRS(in_proj)
    out_proj = pyproj.CRS(out_proj)
    project =  pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True).transform
    AoI_point = transform(project, in_pt)
    return AoI_point

def how_close_is_calibrated_point(point_1, point_2, buffer_distance, point_number):
    #rint(point_1, point_2)
    # create your circle buffer from one of the points - is this in km?
    distance = buffer_distance
    circle_buffer = point_1.buffer(distance)
    # but a simpler method is to simply check the distance
    meters_distance = point_1.distance(point_2)
    #print(meters_distance)
    if point_1.distance(point_2) < distance:
        print(f'Test Point {point_number} is within {distance} m of calibrated Point {point_number}')
        return True, meters_distance
    else:
        print(f'Test Point {point_number} is not within {distance} m of calibrated Point {point_number}')
        return False, meters_distance

def find_points_within_buffer_distance(point_to_test, calib_multipoints, buffer_distance_meters):
    circle = point_to_test.buffer(buffer_distance_meters)
    points_in_buffer = []
    for p in calib_multipoints:
        if circle.covers(p):
            #print(p.x, p.y)
            point_in_buffer = Point(p.x,p.y)
            points_in_buffer.append(point_in_buffer)
    return points_in_buffer


# convert calib_points to the same coord system as the test points
def create_list_test_calib_test_points(calib_df, test_df):
    calib_points = []
    test_points_list = []
    for i in range(len(calib_df)):
        calib_y = int(calib_df['lat_calib'][i])
        calib_x = int(calib_df['lon_calib'][i])
        calib_point = convert_crs_point(calib_x, calib_y, 'epsg:32633', 'epsg:32633')
        test_y = test_df['geometry'][i].y
        test_x = test_df['geometry'][i].x
        #print(test_x,test_y)
        test_point = convert_crs_point(test_x, test_y, 'epsg:4326', 'epsg:32633')
        calib_points.append(calib_point)
        test_points_list.append(test_point)
    return test_points_list, calib_points

def get_points_in_buffer_distance(in_file, out_file):
    #########
    # need the following conversion as the input of the create_list_test_calib_test_points function
    # the x,y test points are taken from the bool_lat_lon file which has the
    # geometry column in POINT geometry form.
    test_points = pd.read_csv('./bool_lat_lon.csv')
    test_points['geometry'] = test_points['geometry'].apply(wkt.loads)
    test_points_gdf = gpd.GeoDataFrame(test_points, crs='epsg:4326')
    #print(test_points_gdf['geometry'][0])

    calib_points_df = pd.read_csv(in_file)#, index=None)
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



# convert_calib_to_lat_lon('/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_AoI_epsg32633.bil',\
# '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv',\
# './test_transform.bil')
#
# convert_calib_raster_to_csv_shp('./test_transform.bil')
#
# multipoint, selected_rows = create_calib_multipoint('./test_transform.csv')
#
# calib_params_closest_point('./bool_lat_lon.csv', '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv',\
# './test_closest_calibration_points_add_coords.csv')
#
# get_points_in_buffer_distance('./test_closest_calibration_points_add_coords.csv', './test_points_within_buffer_distance.csv')

'''
# All of the code below is for the development of a system to calculate error on
# the values taken for the calibration.
# I will finish it eventually but now it is in development.

all_calib_points_in_extended_buffer = []
# add this boolean list with buffer distances to the test_points dataframe

for i in range(len(test_points_list)):
    # find if there are other points within the buffer distance
    # these points may not be the closest ones to the point of interest
    points_in_buffer = find_points_within_buffer_distance(test_points_list[i], multipoint, 1000)
    print(test_points_list[i])
    #print(points_in_buffer[])
    #all_calib_points_in_extended_buffer[i] = Point(test_points_list[i].x, test_points_list[i].y)
    all_calib_points_in_extended_buffer.append(points_in_buffer)


print(all_calib_points_in_extended_buffer[0][0].x)
print(all_calib_points_in_extended_buffer)
full_calibration_df = pd.read_csv('/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv', index_col = None)
print(selected_rows)

columns_df = ['alpha', 'D_0', 'K_sat', 'd', 'Iz_over_K_steady', 'friction_angle','cohesion', 'weight_of_water', 'weight_of_soil']
avg_calib_params = pd.DataFrame(columns = columns_df)

for i in range(len(test_points_list)):
    indices_to_average = []
    for j in range(len(all_calib_points_in_extended_buffer[i])):
        calibration_params_index = list(selected_rows[selected_rows['X']==all_calib_points_in_extended_buffer[i][j].x].index.values)
        calibration_parameters_selection = int(selected_rows['Z'][calibration_params_index])
        calibration_parameters = full_calibration_df.iloc[[calibration_parameters_selection]]
        index_to_avg = full_calibration_df.iloc[[calibration_parameters_selection]].index[0]
        indices_to_average.append(index_to_avg)
    calibration_parameters = full_calibration_df.iloc[[calibration_parameters_selection]]
    print(calibration_parameters)
    #print(full_calibration_df[['alpha', 'D_0', 'K_sat', 'd', 'Iz_over_K_steady', 'friction_angle','cohesion', 'weight_of_water', 'weight_of_soil']].iloc[indices_to_average].mean(axis=1))
    avg_calib_params.append(full_calibration_df[['alpha', 'D_0', 'K_sat', 'd', 'Iz_over_K_steady', 'friction_angle','cohesion', 'weight_of_water', 'weight_of_soil']].iloc[indices_to_average].mean(axis=0), ignore_index=True)

# need to find the location of these points in the calibration array so that we can average the values.
print(avg_calib_params)

'''
##################

#find_points_within_buffer_distance(Point(502133, 4547372), multipoint, 1000)
#how_close_is_calibrated_point(Point(502133, 4547372), Point(502386, 4547119))
