# Marina Ruiz Sanchez-Oro
# 16/11/2021

# File to explore the .csv file that contains all the parameters about the
# failures and how they relate to rainfal values.
# The aim is to check what are the points the immediately fail given any amount
# of rainfall at the start of the timeseries.

import pandas as pd
import numpy as np

precip_file = "/exports/csce/datastore/geos/groups/LSDTopoData/FORESEE/Data/Calibration/2014-01-01_to_2019-12-31_Intensity.csv"
precip = pd.read_csv(precip_file)

# extract the times of failure
