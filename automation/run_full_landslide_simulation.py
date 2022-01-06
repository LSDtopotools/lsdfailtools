import json
import os
import sys

from run_json import *
# path to where the python code is in the server
FILE_PATHS = read_paths_file()
path_to_module = FILE_PATHS["path_to_simulation_repo"]

sys.path.append(path_to_module)

from run_individual_landslide_modules import *
# sys.argv1 is the directory where the data will be stored
# sys.argv2 is the name of the coordinate input file
# sys argv3 is the name of the precipitation file

run_landslide_simulation(sys.argv[1], sys.argv[2], sys.argv[3])
