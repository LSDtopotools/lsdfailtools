import json
import os
import sys
# path to where the python code is in the server
sys.path.append( '/exports/csce/datastore/geos/users/s1440040/projects/lsdfailtools/automation' )

from run_individual_landslide_modules import *

run_landslide_simulation(sys.argv[1], sys.argv[2], sys.argv[3])
