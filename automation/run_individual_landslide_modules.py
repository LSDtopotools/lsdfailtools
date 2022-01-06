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
from zip_outputs import *
##########################################################################
def run_landslide_simulation(input_dir, input_file_lat_lon, input_file_rain):
    """
    run_landslide_simulation calculates the factor of safety timeseries for the test points that record a failure. Removes the unwated temporary files.
    :param input_dir: path to the directory with the test data points
    :param input_file_lat_lon: name of the file with the test data points
    :param input_file_rain: full path and name of the rainfall timeseries data
    """
    run_lat_lon_area_check(input_dir, input_file_lat_lon)
    run_find_closest_calibrated_point()
    landslide_output_from_rain(input_file_rain, input_dir)
    files_to_zip = zip_output_files(input_dir)
    remove_zipped_files(files_to_zip)
    remove_unwanted_files()
