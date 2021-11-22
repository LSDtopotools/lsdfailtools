import numpy as np
#import system
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
#import shapefile
import itertools
import json
import os
import rasterio
from rasterio.features import shapes

import sys
import fiona
#import functions as fn

# input the data file
## must have a list of lat lon points

file = gpd.read_file('/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Topography/AoI.shp', index=False)
file = gpd.GeoDataFrame(file)
file = file.drop(columns=['id'])


csv_file_path = './test_lat_lon.csv'
csv_df = pd.read_csv(csv_file_path)

geometry = [Point(xy) for xy in zip(csv_df.lon, csv_df.lat)]

# assume the projection is the same as the one in the AoI shapefile.
geo_df = gpd.GeoDataFrame(csv_df, geometry=geometry)
# NEED TO POSSIBLY CHANGE THIS SO THAT IT TAKES THE PROJECTION FROM THE AREA OF INTEREST
geo_df = geo_df.set_crs('epsg:4326')

geo_df = geo_df.drop(columns=['lat', 'lon'])

print(geo_df.head(5))

# aoi_polygon = file.iloc[0]
## In first instance we will assume there is aonly one data point in the area of interest.
## later we will adapt for more than 1 datapoint.
for i in range(len(geo_df)):
    test_point = geo_df['geometry'].iloc[i]
    print(test_point)
    print(file.contains(test_point))
    is_in_area = file.contains(test_point)[0]
    geo_df.at[i,'is_in_area'] = is_in_area

# keep only the columns with the points in the AoI
geo_df = geo_df[geo_df.is_in_area]
print(geo_df.head())


################################################################################
################################################################################
################################################################################
################################################################################
