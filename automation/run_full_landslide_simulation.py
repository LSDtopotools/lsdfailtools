import json
import os
import sys
# path to where the python code is in the server
sys.path.append( '/exports/csce/datastore/geos/users/s1440040/projects/lsdfailtools/automation' )

from run_individual_landslide_modules import *
# sys.argv1 is the directory where the data will be stored
# sys.argv2 is the name of the coordinate input file
# sys argv3 is the name of the precipitation file

run_landslide_simulation(sys.argv[1], sys.argv[2], sys.argv[3])
