# Checks if the given points are in the area of interest
# Creates a file with the lat, lon converted to points. 
# usage: Script name, input/output file path, input file name
python lat_lon_area_check.py ./ test_lat_lon.csv
# 
# usage: script name, directory with the file outputs from he previous script 
python find_closest_calibrated_point.py ./
rm ./test_transform.bil
# script name, rainfall file path, 
python single_point_validation.py /exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/2014-01-01_to_2019-12-31_Intensity.csv ./

