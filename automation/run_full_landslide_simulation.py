import argparse

from .run_individual_landslide_modules import *
from .run_json import read_paths_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rundir',help='directory where the data will be stored')
    parser.add_argument('coordinates', help='name of the coordinate input file')
    parser.add_argument('precipitation', help='name of the precipitation file')
    parser.add_argument('-c', '--config', default='file_paths_landslide_automation.json',
                        help='configuration file')
    args = parser.parse_args()

    file_paths = read_paths_file(args.config)
    
    run_landslide_simulation(args.rundir, args.coordinates, args.precipitation, file_paths)


if __name__ == '__main__':
    main()
