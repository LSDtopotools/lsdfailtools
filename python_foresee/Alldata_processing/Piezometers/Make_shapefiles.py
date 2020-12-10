
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
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import csv

import functions as fn


################################################################################
################################################################################
# Set the piezometer directory variables
################################################################################
################################################################################

with open("../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["out_failure_dir"]))

piezo_dir = FILE_PATHS["piezo_dir"]
piezo_data = FILE_PATHS["piezo_data"]
piezo_loc = FILE_PATHS["piezo_location"]


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

# Load piezo locations
Piezo_loc = pd.read_csv(piezo_loc)

# Load piezo data
Piezo_data = pd.read_csv(piezo_data)
DF = fn.inclino_to_one_df(Piezo_loc, Piezo_data)
fn.piezo_to_shp(DF, piezo_dir, 'test_')

# read and reproject all shapefile
for file in os.listdir(piezo_dir):
    if file.endswith(".shp"):
        shapes = gpd.read_file(piezo_dir + file)
        shapes.crs = 'epsg:'+world_epsg
        shapes_ita = shapes.to_crs({'init': 'epsg:'+ita_epsg})
        shapes_ita.to_file(driver = 'ESRI Shapefile', filename= piezo_dir+file[:-4]+"_epsg"+ita_epsg+".shp")
