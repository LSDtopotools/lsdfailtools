"""
Make_shapefiles.py

This is a file to convert .csv files into shapefiles
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

with open("file_paths_inclinometer.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

################################################################################
################################################################################
# Set the inclinometer variables
################################################################################
################################################################################

inclino_dir = FILE_PATHS["inclino_dir"]
inclino_data = FILE_PATHS["inclino_data"]
inclino_loc = FILE_PATHS["inclino_location"]


################################################################################
################################################################################
# IMPORTANT! Set the EPSG
################################################################################
################################################################################

world_epsg = FILE_PATHS["world_epsg_initial"] # for WGS84
ita_epsg = FILE_PATHS["italy_epsg_final"] # for Italy

################################################################################
################################################################################
# Load the .csv data (instrument location and readings)
################################################################################
################################################################################

# Load inclino locations
Inclino_loc = bb.read_csv(inclino_loc)
Inclino_loc.drop('AZIMUTH OF CUMULATIVE DISPLACEMENT - FIGURE', axis = 1, inplace = True)
Inclino_loc.drop('CUMULATIVE DISPLACEMENT - FIGURE', axis = 1, inplace = True)

# Load Inclino data
Inclino_data = bb.read_csv(inclino_data)
DF = fn.inclino_to_one_df(Inclino_loc, Inclino_data)
fn.inclino_to_velocity_shp(DF, inclino_dir, 'test_', int(world_epsg))

# read and reproject all shapefile
for file in os.listdir(inclino_dir):
    if file.endswith(".shp"):
        shapes = gpd.read_file(inclino_dir + file)
        shapes.crs = 'epsg:'+world_epsg
        shapes_ita = shapes.to_crs({'init': 'epsg:'+ita_epsg})
        shapes_ita.to_file(driver = 'ESRI Shapefile', filename= inclino_dir+file[:-4]+"_epsg"+ita_epsg+".shp")
