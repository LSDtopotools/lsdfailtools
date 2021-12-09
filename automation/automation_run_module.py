import json
import os
import sys

sys.path.append( '/exports/csce/datastore/geos/users/s1440040/projects/lsdfailtools/automation' )

#####################################
from get_lat_lon_check import *
from get_closest_calib_point import *
from get_output_files import *
from remove_files import *
##########################################################################
def run_all_scripts(input_dir, input_file_lat_lon, rain_file):
    #input_dir = sys.argv[1]
    #input_file = sys.argv[2]
    run_lat_lon_area_check(input_dir, input_file_lat_lon)
    run_find_closest_calibrated_point()
    run_single_point_validation(rain_file)
    remove_unwanted_files()
