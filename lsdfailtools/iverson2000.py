# Importing the c++ code here
import lsdiverson as lsdi
import numpy as numpy
import pandas as pd


class iverson_model(object):
	"""docstring for iverson_model"""
	def __init__(self, alpha = 0.1, D_0 = 5e-6,K_sat = 5e-8, d = 2,Iz_over_K_steady = 0.2,
      friction_angle = 0.38, cohesion = 12000, weight_of_water = 9800, 
      weight_of_soil = 19000, depths = None):

		super(iverson_model, self).__init__()
		self.alpha  = alpha
		self.D_0  = D_0
		self.K_sat  = K_sat
		self.d = d
		self.Iz_over_K_steady  = Iz_over_K_steady
		self.friction_angle  = friction_angle
		self.cohesion  = cohesion
		self.weight_of_water  = weight_of_water
		self.weight_of_soil  = weight_of_soil
		if(depths is not None):
			self.depths  = np.array(depths)
		else:
			# Default array
			self.depths  = np.arange(0.1,5,0.2)
		
		# creating the c++ object
		self.cppmodel = lsdi(alpha,D_0,K_sat,d,Iz_over_K_steady,friction_angle,cohesion,weight_of_water,weight_of_soil,depths.min())
		# Initialising the depths vector
		self.cppmodel.set_depths_vector(self.depths)

	def run(self, durations_of_prec, intensities_of_prec):

		# first I need to set the duration and intensities of precipitations
		self.cppmodel.set_duration_intensity(durations_of_prec,intensities_of_prec)
		
		# Then I can run the model
		self.cppmodel.ScanTimeseriesForFailure()

		# to get outputs:
		# self.cppmodel.output_times
		# self.cppmodel.output_depthsFS
		# self.cppmodel.output_minFS
		# self.cppmodel.output_PsiFS
		# self.cppmodel.output_durationFS
		# self.cppmodel.output_intensityFS
		# self.cppmodel.output_failure_times
		# self.cppmodel.output_failure_mindepths
		# self.cppmodel.output_failure_maxdepths
		# self.cppmodel.output_Psi_timedepth
		# self.cppmodel.output_FS_timedepth

