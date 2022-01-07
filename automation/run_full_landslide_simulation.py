import argparse

from .run_individual_landslide_modules import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rundir',help='directory where the data will be stored')
    parser.add_argument('coordinates', help='name of the coordinate input file')
    parser.add_argument('precipitation', help='name of the precipitation file')
    args = parser.parse_args()
    run_landslide_simulation(args.rundir, args.coordinates, args.precipitation)


if __name__ == '__main__':
    main()
