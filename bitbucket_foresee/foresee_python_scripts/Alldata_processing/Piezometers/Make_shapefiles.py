
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
import json
import numpy as np
import pandas as bb
import geopandas as gpd
import csv

import functions as fn


################################################################################
################################################################################
# Set the piezo_dir variables
################################################################################
################################################################################

with open("../../../../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)
#piezo_data, piezo_loc

#piezo_dir = "/home/willgoodwin/PostDoc/Foresee/Data/Terrestrial/"
piezo_dir = FILE_PATHS["piezo_dir"]
piezo_data = piezo_dir + "data_piezometer.csv"
piezo_loc = piezo_dir + "coo_piezometer.csv"


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

# Load piezo locations
Piezo_loc = bb.read_csv(piezo_dir+piezo_loc)

# Load piezo data
Piezo_data = bb.read_csv(piezo_dir+piezo_data)
DF = fn.inclino_to_one_df(Piezo_loc, Piezo_data)
fn.piezo_to_shp(DF, piezo_dir, 'test_')

# read and reproject all shapefile
for file in os.listdir(piezo_dir):
    if file.endswith(".shp"):
        shapes = gpd.read_file(piezo_dir + file)
        shapes.crs = 'epsg:'+world_epsg
        shapes_ita = shapes.to_crs({'init': 'epsg:'+ita_epsg})
        shapes_ita.to_file(driver = 'ESRI Shapefile', filename= piezo_dir+file[:-4]+"_epsg"+ita_epsg+".shp")


"""
Reprojecting stuff automatically can be annoying. This stuff is helpful
https://glenbambrick.com/2016/01/24/reproject-shapefile/
"""