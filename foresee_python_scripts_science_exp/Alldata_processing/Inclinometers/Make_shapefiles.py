"""
Make_shapefiles.py

This is a file to make these pesky .csv files behave
"""



################################################################################
################################################################################
#Import packages
################################################################################
################################################################################

import os
import csv
import json
import numpy as np
import pandas as bb
import geopandas as gpd
import functions as fn

with open("../../../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

################################################################################
################################################################################
# Set the inclino_dir variables
################################################################################
################################################################################
# inclino_dir
inclino_dir = FILE_PATHS["inclino_dir"]
inclino_data = "data_inclinometer.csv"
inclino_loc = "coo_inclinometer.csv"


################################################################################
################################################################################
# IMPORTANT! Set the EPSG
################################################################################
################################################################################

world_epsg = '4326' # for WGS84
ita_epsg = '32633' # for Italy

################################################################################
################################################################################
# Load the .csv data (instrument location and readings)
################################################################################
################################################################################

# Load inclino locations
Inclino_loc = bb.read_csv(inclino_dir+inclino_loc)
print("hello")
Inclino_loc.drop('AZIMUTH OF CUMULATIVE DISPLACEMENT - FIGURE', axis = 1, inplace = True)
Inclino_loc.drop('CUMULATIVE DISPLACEMENT - FIGURE', axis = 1, inplace = True)

# Load Inclino data
Inclino_data = bb.read_csv(inclino_dir+inclino_data)
DF = fn.inclino_to_one_df(Inclino_loc, Inclino_data)
fn.inclino_to_velocity_shp(DF, inclino_dir, 'test_', int(world_epsg))

# read and reproject all shapefile
for file in os.listdir(inclino_dir):
    if file.endswith(".shp"):
        shapes = gpd.read_file(inclino_dir + file)
        shapes.crs = 'epsg:'+world_epsg
        shapes_ita = shapes.to_crs({'init': 'epsg:'+ita_epsg})
        shapes_ita.to_file(driver = 'ESRI Shapefile', filename= inclino_dir+file[:-4]+"_epsg"+ita_epsg+".shp")


"""
Reprojecting stuff automatically can be annoying. This stuff is helpful
https://glenbambrick.com/2016/01/24/reproject-shapefile/
"""
