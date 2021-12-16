import json
import os
import sys
from run_json import *

FILE_PATHS = read_paths_file()
path_to_module = FILE_PATHS["path_to_simulation_repo"]

sys.path.append(path_to_module)

#####################################
from get_lat_lon_check import *
from get_closest_calib_point import *
from get_output_files import *
from remove_files import *
##########################################################################
def run_landslide_simulation(input_dir, input_file_lat_lon, input_file_rain):
    run_lat_lon_area_check(input_dir, input_file_lat_lon)
    run_find_closest_calibrated_point()
    landslide_output_from_rain(input_file_rain, input_dir)
    remove_unwanted_files()
