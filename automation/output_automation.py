# Output for the given points of interest

# take in the validation file that we need
import pandas as pd
import numpy as np
import seaborn as sns

valid_file = "./Validated_updated_test_one_point.csv"
valid = pd.read_csv(valid_file)

# import data from the file that has the data with the points that always
# collapse very early on in the time series regardless of rainfall.
rain_file = "./early_Validated_updated_FoS_depth_100_0.001.csv"
rain = pd.read_csv(rain_file)

# create a column with booleans values for the points which always fail
rain['is_it_failure'] = np.where(rain['time_of_failure']!= 0, True, False)


# create a column with booleans values for
valid['time_of_failure'] = valid['time_of_failure'].apply(lambda x: x/86400)
