import numpy as np

from itertools import product
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import pandas as pd
import itertools
import json
import os
import rasterio
from rasterio.features import shapes
import sys
import fiona
import seaborn as sns
import image_functions as fn

#calib_file = '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/Calibrated_FoS_depth.csv'
print('Read in the file with the points that have failed too early')
all_precip_early_file = './Validated_updated_FoS_depth_all_days_early.csv'
print('Convert this file to a dataframe')
all_precip_early = pd.read_csv(all_precip_early_file)
print(len(all_precip_early))

#calib = pd.read_csv(calib_file, index_col = 0)
print('Read in the dem file that I will take the coordinates from. I need this to convert row,col to lat,lon')
dem_file = '/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/eu_dem_AoI_epsg32633.bil'

# need to convert the row/col points to lat/lon
# should probably change the names inside the function to make it more general
# it is not always a dem that we want to convert - it can be any raster
def convert_array_to_lat_lon(dem_raster, row_col_csv, output_calib_raster):
    dem_file = dem_raster

    # 0. Load rasters into arrays - this must be a raster where the array that we want to convert lies
    demarr, pixelWidth, (geotransform, inDs) = fn.ENVI_raster_binary_to_2d_array(dem_file)

    row_col_file = row_col_csv

    # need to convert the csv into a geodataframe
    input_df = pd.read_csv(row_col_file)
    input_array = 0*demarr
    for i in range (len(input_df)):
        x = int(input_df['row'].iloc[i])
        y = int(input_df['col'].iloc[i])
        # need to add +1 here because otherwise it thinks it's a 0 and it's identified as Nan
        input_array[x,y] = input_df.index[i] + 1

    input_array[input_array == 0] = 'nan'

    # convert the row,col of the calibration file into lat, lon coordinates
    new_geotransform,new_projection,file_out = fn.ENVI_raster_binary_from_2d_array( (geotransform, inDs), output_calib_raster, pixelWidth, input_array)
    print('I am done. The row, col file now has lon, lat coordinates.')

# get the name of the file from the full filepath and extension
def convert_raster_to_csv_shp(raster):
    filename_full = os.path.basename(raster)
    filename = filename_full.split('.')
    filename = filename[0]

    inDs = gdal.Open(raster)
    outDs = gdal.Translate('{}.xyz'.format(filename), inDs, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
    outDs = None
    try:
        os.remove('{}.csv'.format(filename))
    except OSError:
        pass
    os.rename('{}.xyz'.format(filename), '{}.csv'.format(filename))
    os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=NO {0}.shp {0}.csv'.format(filename))



print('I am going to create a raster file from the row, col fil that you provided.')
#convert_array_to_lat_lon(dem_file, all_precip_early_file, './failure_anomalies.bil')

print('I am now going to convert the lat,lon file that I just created to a csv file ')
#convert_raster_to_csv_shp('./failure_anomalies.bil')



# Need to add the lat,lon from failure_anomalies into the csv file/dataframe
# with the info from the anomalous precipitation
#quit()
# extract the lat, lon values of ALL POINTS IN THE AOI!!!!!
all_coords = pd.read_csv('./failure_anomalies.csv', ' ', index_col=None)
# when the 'Z' values are not Nan, this corresponds to one of the points which
# are in the original array of values we are trying to convert.
print(all_coords.iloc[20440])
print(len(all_coords))
#df[~np.isnan(df)]
failures_all_coords = all_coords['Z'].notnull()
indices_failures_all_coords = [i for i, x in enumerate(failures_all_coords) if x]
print(indices_failures_all_coords[:10])

# when the 'Z' values are not Nan, this corresponds to one of the points which
# are in the original array of values we are trying to convert.
failures_all_coords = all_coords[~all_coords['Z'].isnull()]
failures_all_coords = failures_all_coords.astype(int)

indices_failures_all_coords = failures_all_coords.index
print(len(failures_all_coords))


#all_coords = all_coords.drop('Z', 1)
#all_coords = all_coords.astype(int)

Y = pd.DataFrame(failures_all_coords['Y'])
X = pd.DataFrame(failures_all_coords['X'])

X.reset_index(inplace=True,drop=True)
Y.reset_index(inplace=True,drop=True)

all_precip_early['X'] = X
all_precip_early['Y'] = Y

#all_precip_early.to_csv('./failure_anomalies_with_coords.csv')

# All precipitation timeseries - length 1826 - early peak


# select the indices of the rows with non-zero - these are the failures that happen too early and are anomalous
non_zero_indexes_all = all_precip_early.index[all_precip_early['factor_of_safety'] != 0].tolist()
# add a boolean column
all_precip_early['is_it_failure'] = np.where(all_precip_early['time_of_failure']!= 0, True, False)
# convert the time of failure into days
all_precip_early['time_of_failure'] = all_precip_early['time_of_failure'].apply(lambda x: x/86400)
# dataframe with only the rows of the points that have failed early - anomalies
anomaly_failures=all_precip_early.loc[all_precip_early['is_it_failure'] == True]

print(anomaly_failures)
anomaly_failures.to_csv('anomaly_failures.csv', index=False)
