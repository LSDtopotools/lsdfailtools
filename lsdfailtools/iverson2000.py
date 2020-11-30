# Importing the c++ code here
from lsdfailtools_cpp import lsdiverson as lsdi
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import random
import os
import gc

class iverson_model(object):
  """docstring for iverson_model"""
  def __init__(self, alpha = 0.1, D_0 = 5e-6,K_sat = 5e-8, d = 2,Iz_over_K_steady = 0.2,
      friction_angle = 0.38, cohesion = 12000, weight_of_water = 9800,
      weight_of_soil = 19000, depths = "default", **kwargs):

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
    if(isinstance(depths,str) == False):
      self.depths  = np.array(depths)
    else:
      # Default array
      self.depths  = np.arange(0.1,5,0.2)

    # creating the c++ object
    self.min_depth = self.depths.min()
    self.cppmodel = lsdi(alpha,D_0,K_sat,d,Iz_over_K_steady,friction_angle,cohesion,weight_of_water,weight_of_soil,self.min_depth)
    # Initialising the depths vector
    self.cppmodel.set_depths_vector(self.depths)

  def run(self, durations_of_prec, intensities_of_prec):

    # first I need to set the duration and intensities of precipitations
    self.cppmodel.set_duration_intensity(durations_of_prec,intensities_of_prec)

    # Then I can run the model
    self.cppmodel.ScanTimeseriesForFailure()

    # to get outputs:
    # self.cppmodel.output_times:
    ## 1d array of time for each outputs

    # self.cppmodel.output_depthsFS
    ## 1D array (dim=time) of depths where the minimum factor of safety is for each timestep
    ## Might always be the surface that ones!!!

    # self.cppmodel.output_minFS
    ## 1D array (dim=time) giving the minimum FS of the whole depths column (its depths is given by self.cppmodel.output_depthsFS)

    # self.cppmodel.output_PsiFS
    ## 1darray (dim = ?) not sure how it gets PSI here ...

    # self.cppmodel.output_durationFS
    ## 1D array of dureation of each rain events (in seconds ?)

    # self.cppmodel.output_intensityFS
    ## !d array (dim = time) containing the corresponding intensities of rain events

    # self.cppmodel.output_failure_times
    ## 1darray (dim = time) erm the time corresponding to the output_failure_bool

    # self.cppmodel.output_failure_bool
    ## 1darray (dim = time) 1 if there is a failure at that timestep


    # self.cppmodel.output_failure_mindepths
    ## 1d array (dim = time): the minimum depth at which there is a factor of safety < 1 (WARNING, 9999 if no failure)
    # self.cppmodel.output_failure_maxdepths
    ## 1d array (dim = time): the maximum depth at which there is a factor of safety < 1 (WARNING, 0 if no failure)

    # self.cppmodel.output_Psi_timedepth
    ## 2D array (dims = time,depths) of Psi values
    # self.cppmodel.output_FS_timedepth
    ## 2D array (dims = time,depths) of factor of safety values



class MonteCarlo_Iverson(object):

  def __init__(self, alpha_min = 0.1, D_0_min = 5e-6,K_sat_min = 5e-8, d_min = 2,Iz_over_K_steady_min = 0.2,
      friction_angle_min = 0.38, cohesion_min = 12000, weight_of_water_min = 9800,
      weight_of_soil_min = 19000, alpha_max = 0.1, D_0_max = 5e-6,K_sat_max = 5e-8, d_max = 2,Iz_over_K_steady_max = 0.2,
      friction_angle_max = 0.38, cohesion_max = 12000, weight_of_water_max = 9800,
      weight_of_soil_max = 19000, depths = "default"):



    self.alpha_min = alpha_min
    self.D_0_min = D_0_min
    self.K_sat_min = K_sat_min
    self.d_min = d_min
    self.Iz_over_K_steady_min = Iz_over_K_steady_min
    self.friction_angle_min = friction_angle_min
    self.cohesion_min = cohesion_min
    self.weight_of_water_min = weight_of_water_min
    self.weight_of_soil_min = weight_of_soil_min
    self.alpha_max = alpha_max
    self.D_0_max = D_0_max
    self.K_sat_max = K_sat_max
    self.d_max = d_max
    self.Iz_over_K_steady_max = Iz_over_K_steady_max
    self.friction_angle_max = friction_angle_max
    self.cohesion_max = cohesion_max
    self.weight_of_water_max = weight_of_water_max
    self.weight_of_soil_max = weight_of_soil_max
    self.depths = depths

  def run_MC_failure_test(self,durations_of_prec, intensities_of_prec, n_process = 4, output_name = "test_MC.csv", n_iterations = 100, replace = False):

    # first checking if the file exists
    if(os.path.exists(output_name) and replace == False):
      raise "Your file already exists, I am aborting. if you do not care about replacing the existing file and loose the data, you can add replace=True in the function call"

    df = pd.DataFrame({"alpha": [],"D_0": [],"K_sat": [],"d": [],"Iz_over_K_steady": [],"friction_angle": [],"cohesion": [],"weight_of_water": [],"weight_of_soil": [], "time_of_failure": [], "factor_of_safety": [], "min_depth": []})
    df.name_to_save = output_name

    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()

    pool = mp.Pool(n_process)
    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    q.put([df,output_name])
    #fire off workers
    jobs = []
    for i in range(n_iterations):
      this_arg = {}

      this_arg["alpha"] = random.uniform(self.alpha_min,self.alpha_max)
      this_arg["D_0"] = 10**(random.uniform(np.log10(self.D_0_min),np.log10(self.D_0_max)))
      this_arg["K_sat"] = 10**(random.uniform(np.log10(self.K_sat_min),np.log10(self.K_sat_max)))
      this_arg["d"] = random.uniform(self.d_min,self.d_max)
      this_arg["Iz_over_K_steady"] = random.uniform(self.Iz_over_K_steady_min,self.Iz_over_K_steady_max)
      this_arg["friction_angle"] = random.uniform(self.friction_angle_min,self.friction_angle_max)
      this_arg["cohesion"] = random.uniform(self.cohesion_min,self.cohesion_max)
      this_arg["weight_of_water"] = random.uniform(self.weight_of_water_min,self.weight_of_water_max)
      this_arg["weight_of_soil"] = random.uniform(self.weight_of_soil_min,self.weight_of_soil_max)
      this_arg["depths"]  = self.depths
      this_arg["durations_of_prec"] = durations_of_prec
      this_arg["intensities_of_prec"] = intensities_of_prec

      job = pool.apply_async(worker, (this_arg, q))
      jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


def worker(arg, q):
    """
      THe Worker runs one simulation with random params
    """
    this_model = iverson_model(**arg)
    this_model.run( arg["durations_of_prec"], arg["intensities_of_prec"])

    time = this_model.cppmodel.output_times
    minimum_FS = this_model.cppmodel.output_minFS
    minimum_depth = this_model.cppmodel.output_depthsFS
    #output_FS_timedepth = this_model.cppmodel.output_FS_timedepth


    failure = np.cumsum(this_model.cppmodel.output_failure_bool.astype(np.int32))
    ToF = time[np.argwhere(failure == 1)]
    FoS = minimum_FS[np.argwhere(failure == 1)]
    min_depth = minimum_depth[np.argwhere(failure == 1)]

    if(ToF.shape[0] == 0):
      ToF = -9999
    else:
      ToF = ToF[0][0]

    if(FoS.shape[0] == 0):
      FoS = -9999
    else:
      FoS = FoS[0][0]

    if(min_depth.shape[0] == 0):
      min_depth = -9999
    else:
      min_depth = min_depth[0][0]


    df = pd.DataFrame({"alpha": [arg["alpha"]],"D_0": [arg["D_0"]],"K_sat": [arg["K_sat"]],"d": [arg["d"]],"Iz_over_K_steady": [arg["Iz_over_K_steady"]],
      "friction_angle": [arg["friction_angle"]],"cohesion": [arg["cohesion"]],"weight_of_water": [arg["weight_of_water"]],
      "weight_of_soil": [arg["weight_of_soil"]], "time_of_failure": [ToF], "factor_of_safety": [FoS], "min_depth": [min_depth]})

    q.put(df)

    return df

def listener(q):
  '''listens for messages on the q, writes to file. '''
  create = True
  cpt = 0
  while 1:
    m = q.get()
    cpt += 1
    if(isinstance(m,str)):
      if m == 'kill':
        df.to_csv(output_name, index = False)
        break

    if(create):
      df = m[0]

      output_name = m[1]
      create = False
    else:
      df = df.append(m,ignore_index=True)

    if(cpt % 10 ==0):
      df.to_csv(output_name, index = False)
      gc.collect()
