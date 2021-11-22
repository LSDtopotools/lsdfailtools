import numpy as np
import pandas as pd

###
# File to create fake precipitation timeseries to test when points fail based
# on the precipitation response.
###


##### Generate early rainfall data ######################
early_precip = np.zeros((100))
early_precip[:] = 0
##### Generate late rainfall data #######################

late_precip = np.zeros((100))
late_precip[:] = 0
# save to csv files

duration = [84600]*100
#### early precipitation ####
# dictionary of lists
dict = {'duration_s': duration, 'intensity_mm_sec': early_precip}

df = pd.DataFrame(dict)

# saving the dataframe
df.to_csv('early_precip_100_0.csv', index=False)

#### late precipitation ####
dict = {'duration_s': duration, 'intensity_mm_sec': late_precip}

df = pd.DataFrame(dict)

# saving the dataframe
df.to_csv('late_precip_100_0.csv', index=False)
