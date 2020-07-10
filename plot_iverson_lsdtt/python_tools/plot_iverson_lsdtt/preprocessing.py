"""
This file contains useful functions to preprocess the input data before ingestion.
First test, subjected to evolve quite a lot then.
B.G. - 01/02/2019
"""

import matplotlib;matplotlib.use("Agg");from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import numba

@numba.jit(nopython = True)
def collapse_pandas_col(A,B):
	"""
	This function takes 2 arrays with the same size and merge all the row of A that have the same value while summing the corresponding B ones. The I one (added later) is the index
	There is probably a Pandas way to do so but groupby was a bit buggy so let's hard code it. Not a big deal.
	Returns two numpy arrays: the collapsed one and the summed one
	"""

	C = []
	D = []
	I = []
	last_element = A[0]
	tempsum = B[0]
	for i in range(1,A.shape[0]):
		# In this case, the previous intensity was the same
		if(A[i] == last_element):
			tempsum += B[i]
			if(i == A.shape[0]-1):
				C.append(tempsum)
				D.append(A[i-1])
		else:
			C.append(tempsum)
			D.append(A[i-1])
			I.append(i)
			tempsum = B[i]

		last_element = A[i]

	return np.array(D), np.array(C), np.array(I) # yes inverting C and D is confusing but too late.



def reformat_USGS_data(input_csv, output_name = "preprocessed_data.csv", prec_col = "", time_col = ""):
	"""
	Takes an input csv from the usgs survey and reformat it for feeding the model

	B.G.
	"""

	# Loading the csv file
	dfi = pd.read_csv(input_csv)
	dfi[prec_col][pd.isnull(dfi[prec_col])] = -9999

	# Dictionnary of output
	outD = {}
	outD["Time_step"]= np.abs(np.array(dfi[time_col].diff(periods=-1))) # getting the diff -> i - (i+1)
	outD["Time_step"][0] = 0
	outD["Time_seconds"] = 24*3600*outD["Time_step"] # Converting to minuts

	intensity_mm, duration_s, indices = collapse_pandas_col(dfi[prec_col].values, outD["Time_seconds"])

	# VERY IMPORTANT
	# CONVERTING INTENSITY TO mm/seconds
	intensity_mm = (intensity_mm*duration_s)/(24*3600)

	odf = pd.DataFrame({"duration_s":duration_s, "intensity_mm_sec": intensity_mm})

	odf.to_csv(output_name, index = False)

	# plt.plot(np.cumsum(odf["duration_s"]), odf["intensity_mm"], lw =1, c = "b")
	# plt.ylim(0, odf["intensity_mm"].max())
	# plt.xlabel("Duration in seconds")
	# plt.ylabel("Precipitation intensity in mm")
	# plt.savefig("check_input_prec.png")
	# plt.clf()



# reformat_USGS_data("20150711_20160809_filtered.csv")