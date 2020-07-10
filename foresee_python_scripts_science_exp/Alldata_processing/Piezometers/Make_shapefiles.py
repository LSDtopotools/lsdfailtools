
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
import numpy as np
import pandas as bb
import geopandas as gpd
import csv

import functions as fn


################################################################################
################################################################################
# Set the path variables
################################################################################
################################################################################

path = "/home/willgoodwin/PostDoc/Foresee/Data/Terrestrial/"
piezo_data = "Piezometer/data_piezometer.csv"
piezo_loc = "Piezometer/coo_piezometer.csv"


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
Piezo_loc = bb.read_csv(path+piezo_loc)

# Load piezo data
Piezo_data = bb.read_csv(path+piezo_data)
DF = fn.inclino_to_one_df(Piezo_loc, Piezo_data)
fn.piezo_to_shp(DF, path, 'test_')

# read and reproject all shapefile
for file in os.listdir(path):
    if file.endswith(".shp"):
        shapes = gpd.read_file(path + file)
        shapes.crs = 'epsg:'+world_epsg
        shapes_ita = shapes.to_crs({'init': 'epsg:'+ita_epsg})
        shapes_ita.to_file(driver = 'ESRI Shapefile', filename= path+file[:-4]+"_epsg"+ita_epsg+".shp")


"""
Reprojecting stuff automatically can be annoying. This stuff is helpful
https://glenbambrick.com/2016/01/24/reproject-shapefile/
"""

