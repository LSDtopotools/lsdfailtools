import json
import os
import sys

from .run_individual_landslide_modules import *
# sys.argv1 is the directory where the data will be stored
# sys.argv2 is the name of the coordinate input file
# sys argv3 is the name of the precipitation file

def main():
    run_landslide_simulation(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == '__main__':
    main()
