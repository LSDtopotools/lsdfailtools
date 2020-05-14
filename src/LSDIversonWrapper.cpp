//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
//
// LSDPorewaterParams
// Land Surface Dynamics PorewterParams object
//
// An object within the University
//  of Edinburgh Land Surface Dynamics group topographic tools
//  This object interfaces with teh porewater column object
//  In a landsacpe each pixel will have its own pore ressures but 
//  the parameters will be constant (or have similar statistical properties)
//  across a landscape. This structure tries to minimize memory requirements
//
// Developed by:
//  Simon M. Mudd
//  Stuart W.D. Grieve
//
// Copyright (C) 2016 Simon M. Mudd 2013 6
//
// Developer can be contacted by simon.m.mudd _at_ ed.ac.uk
//
//    Simon Mudd
//    University of Edinburgh
//    School of GeoSciences
//    Drummond Street
//    Edinburgh, EH8 9XP
//    Scotland
//    United Kingdom
//
// This program is free software;
// you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation;
// either version 2 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY;
// without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the
// GNU General Public License along with this program;
// if not, write to:
// Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor,
// Boston, MA 02110-1301
// USA
//
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#ifndef LSDIversonWrapper_CPP
#define LSDIversonWrapper_CPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "LSDIversonWrapper.hpp"

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same

// #include "TNT/tnt.h"
using namespace std;
// using namespace TNT;



void lsdiversonwrapper::get_duration_intensity_from_preprocessed_input(vector<float> duration_s, vector<float> this_intensity)
{
  // Converting to the right units
  // first taking care of dimensionnalizing the intensity 
  for(size_t i=0; i<duration_s.size(); i++)
  {
    this_intensity[i] = (this_intensity[i]*0.001)/(K_sat);
    if(this_intensity[i]>1) // Cannot be >1
      this_intensity[i]=1;
    else if(this_intensity[i]>1) // Cannot be >1
      this_intensity[i]=0;	
  }

  float sec_of_pressure = 0;
  times = vector<float>();
  for(size_t i=0; i<duration_s.size();i++)
  {
    times.push_back(sec_of_pressure);
    sec_of_pressure = sec_of_pressure + duration_s[i];
  }

  durations = duration_s;
  intensities = this_intensity;
}



//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This scans a timeseries for a failure
// The code takes two vectors with durations and intensities. It can then calculate 
// the pore pressure at any time given these inputs. 
// So you supply a time vector and loop through it to see when failure occurs. 
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void lsdiversonwrapper::ScanTimeseriesForFailure()
{
  // loop through times
  int n_times = int(times.size());
  float depth_of_minFS;
  float min_FS;
  vector<float> vec_of_depth, vec_of_minFS;

  // Map storing the FS for times
  map<float, vector<float> > FSmap, F_c_map, F_w_map;
  map<float,float> F_f_map;

  vector<float> temp_float_vec;
  vector<bool> temp_bool_vec;

  potential_failure_times = temp_float_vec;
  potential_failure_min_depths = temp_float_vec;
  potential_failure_max_depths = temp_float_vec;
  potential_failure_bool = temp_bool_vec;


  for(int i = 0; i< n_times; i++)
  {
    current_tested_time_by_scanner = times[i];
    // get the pore pressure
    CalculatePsiFromTimeSeries(times[i]);

    // get the min
    GetMinFS(minimum_depth, depth_of_minFS, min_FS);

    vec_of_depth.push_back(depth_of_minFS);
    vec_of_minFS.push_back(min_FS);

    FSmap[times[i]] = current_FS;
    F_f_map[times[i]] = current_F_f_float;
    F_c_map[times[i]] = current_F_c_vec;
    F_w_map[times[i]] = current_F_w_vec;

  }

  // 1D output
  output_times = xt::adapt(times);
  output_depthsFS = xt::adapt(vec_of_depth);
  output_minFS = xt::adapt(vec_of_minFS);
  output_PsiFS = xt::adapt(Psi);
  output_durationFS = xt::adapt(durations);
  output_intensityFS = xt::adapt(intensities);

  output_failure_times = xt::adapt(potential_failure_times);
  output_failure_mindepths = xt::adapt(potential_failure_min_depths);
  output_failure_maxdepths = xt::adapt(potential_failure_max_depths);

  // 2D output col = time row = depths

  output_Psi_timedepth = xt::zeros<float>({Depths.size(),times.size()});
  for(size_t i=0; i<Depths.size(); i++)
  {
    
    for(size_t j=0; j< times.size();j++)
    {
      float this_time = times[j];
      output_Psi_timedepth(i,j) = vec_of_Psi[this_time][i];
    }
  }


  output_FS_timedepth = xt::zeros<float>({Depths.size(),times.size()});
  for(size_t i=0; i<Depths.size(); i++)
  {
    
    for(size_t j=0; j< times.size();j++)
    {
      float this_time = times[j];
      output_FS_timedepth(i,j) = FSmap[this_time][i];
    }
  }
}


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This calculates the Psi value based on iverson's equation 27
// It parses a time series
// The durations and time are in seconds. 
// **IMPORTANT** The intensities are in Iz_over_Kz
// This wraps the transient components
// The end result of this calculation is that the pore pressure Psi (this combines
//  steady and transient components) is stored in the data member Psi
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void lsdiversonwrapper::CalculatePsiFromTimeSeries(float t)
{
  // Get the steady state time
  vector<float> steady_psi = this->calculate_steady_psi();
  vector<float> cumulative_psi = steady_psi;

  // Now we try to construct the transient pressure. 
  // loop through the record getting cumulative times
  vector<float> starting_times;
  starting_times.push_back(0);
  float cumulative_time = 0;
  int count = 0; 
  bool end_count_found = false;
  int end_count = 0;
  
  for (int i = 0; i< int(durations.size()); i++)
  {
    cumulative_time += durations[i];
    
 
    // the cumulative time is the time at the end of this timestep. 
    // if the cumulative time  is less than the time of simulation, 
    // then we need to acount for this pulse of rainfall        
    if (t < cumulative_time)
    {
      if (end_count_found == false)
      {
        end_count_found = true;
        end_count = count;
      }
    }
    count++;
    starting_times.push_back(cumulative_time);
  }
  
  // we don't need the last element
  starting_times.pop_back();

  // If we didn't find the end count it means the rainfall records have ended and we need
  // all of the data        
  if (end_count_found == false)
  {
    // The minus one is needed since we have counted past the end of the index
    end_count = count-1;
  }

  // okay, now get the transients from superposition 
  // First we need to figure out how many of these we will need
  float eff_t, this_intensity, this_duration;
  vector<float> this_transient_Psi;
  for(int i = 0; i< int(starting_times.size()); i++)
  {
    if(i<= end_count)
    {
      eff_t = t-starting_times[i];
      this_intensity = intensities[i];
      this_duration = durations[i];
           
      // get this steps Psi value
      this_transient_Psi = CalculatePsiDimensionalTimeTransient(eff_t, this_duration, this_intensity);


      // add this step's transient Psi values.
      for(int i = 0; i<int(cumulative_psi.size()); i++)
      {
        cumulative_psi[i]+=this_transient_Psi[i];
      }
    }
  }
  
  // I commented that - BG  
  // // see what the result is:
  // for(int i = 0; i<int(cumulative_psi.size()); i++)
  // {
  //   cout << "Psi["<<i<<"]: " << cumulative_psi[i] << endl;
  // }
  
  Psi = cumulative_psi;
  // Saving the transient psi at dimensional time
  // cout << "WARUM FUNKTIONIERT DU NICHT - WILKOMMEN" << endl;
  vec_of_Psi[t] = cumulative_psi;
  // cout << "WARUM FUNKTIONIERT DU NICHT - WOLKSWAGEN" << endl;

  
}

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This uses the parameters to get a steady state pressure profile
// See the first column in Iverson 2000 page 1902 for explanation
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<float> lsdiversonwrapper::calculate_steady_psi()
{
  vector<float> Psi;
  for(int i = 0 ; i < int(Depths.size()); i++ )
  {
    Psi.push_back(beta*(Depths[i]-d));
  }
  return Psi;

}

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Compute psi from equation 27a and b, but using dimensional time
// A bit slow since I haven't vectorised the calculations.
// Only calculates the transient component of psi for use with 
// time series of rainfall
// times need to be in seconds
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<float> lsdiversonwrapper::CalculatePsiDimensionalTimeTransient(float t, float T, float Iz_over_Kz)
{

  vector<float> transient_Psi(Depths.size());
  float t_star;
  float T_star;
  float zsquare;

  float R;
  
  // loop through depths: each depth has a different t_star and T_star since
  // these depend on depth
  for(int i = 0; i< int(Depths.size()) ; i++)
  {
    // first get the nondimensional time. Note that according to
    // equations 27c,d the dimensionless time is a function of depth,
    // so each point below the surface has a different t_star and T_star
    zsquare = Depths[i]*Depths[i];
    t_star = t * D_hat / zsquare;
    T_star = T * D_hat / zsquare;
    
    if (t_star < T_star)
    {
      R = CalculateResponseFunction(t_star);
    }
    else
    {
      R = CalculateResponseFunction(t_star)-CalculateResponseFunction(t_star-T_star);
    }
    transient_Psi[i] =Depths[i]*Iz_over_Kz*R;
    
  }
  return transient_Psi;
}

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This caluclates the response function
// THis comes from iverson's equation 27e
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
float lsdiversonwrapper::CalculateResponseFunction(float t_star)
{
    float R;
    
    float sqrt_term = sqrt(t_star/M_PI);
    float exp_term = exp(-1/t_star);
    
    float multiple_bit = sqrt_term*exp_term;
    
    if (t_star != 0)
    {
      R = multiple_bit- erfcf(1/ (sqrt(t_star)));
    }
    else   // If t_star is 0, then 1/sqrt(t_star) is infinity, meaning erfc is 0)
    {
      R = 0;
    }

    return R;
}


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This checks to see the minimum factor of safety in the column. 
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void lsdiversonwrapper::GetMinFS(float minimum_depth, float& depth_of_minFS, float& minFS)
{
  depth_of_minFS = minimum_depth;
  float min_FS = 9999;
  
  // get the factor of safety vector
  vector<float> FoS = FS();

  float min_depth = 99999, tempolitarute = 0.; 
  bool failure_does_happen = false;
  
  int N_depths = int(Depths.size());
  for(int i = 0; i< N_depths; i++)
  {
    // only check FS if above minimum depth
    if(Depths[i]>= minimum_depth)
    {
      if(FoS[i] < min_FS)
      {
        depth_of_minFS = Depths[i];
        min_FS  = FoS[i];
      }

      // check where failure might happen
      if(FoS[i]<=1)
      {
        failure_does_happen = true;
        if(Depths[i] > tempolitarute)
          tempolitarute = Depths[i];
        if(Depths[i]< min_depth)
          min_depth = Depths[i];
      }
    }

  }
  
  minFS = min_FS;
  potential_failure_times.push_back(current_tested_time_by_scanner);
  potential_failure_min_depths.push_back(min_depth);
  potential_failure_max_depths.push_back(tempolitarute);
  potential_failure_bool.push_back(failure_does_happen);
}


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This is the total factor of safety (combining different components of the FS)
// calculation. See equations 28a-d in Iverson 2000
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<float> lsdiversonwrapper::FS()
{

  // get the components of the factor of safety
  float F_f_float = F_f();
  vector<float> F_c_vec = F_c();
  vector<float> F_w_vec = F_w();

  // get the Factor safety I guess
  vector<float> FS = F_c_vec;
  for(int i = 0; i< int(F_c_vec.size()); i++)
  {
    FS[i] = F_f_float+F_c_vec[i]+F_w_vec[i];
  }


  // saving the current state of factors
  current_FS = FS;
  current_F_f_float = F_f_float;
  current_F_c_vec = F_c_vec;
  current_F_w_vec = F_w_vec;
  
  return FS;

}


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Factor of safety calculations
// See iverson 2000 equation 28b
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This is the friction
float lsdiversonwrapper::F_f()
{
  float tan_alpha = tan(alpha);
  float tan_friction_angle = tan(friction_angle);
  
  return tan_friction_angle/tan_alpha;
}

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This is from the cohesion
// See iverson 2000 equation 28d
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<float> lsdiversonwrapper::F_c()
{
  
  float denom;
  float denom2 = sin(alpha)*cos(alpha);
  
  vector<float> F_c_vec;
  
  for(int i = 0; i< int(Depths.size()); i++)
  {
    denom = Depths[i]*weight_of_soil;
    
    F_c_vec.push_back( cohesion/(denom*denom2) );
  }
  
  return F_c_vec;

}


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// This is the factor of safety contribution from the water
// See iverson 2000 equation 28c
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<float> lsdiversonwrapper::F_w()
{
    
  float denom, num1, num2;
  float denom2 = sin(alpha)*cos(alpha);
  float denom_tot;
  
  vector<float> F_w_vec;
  
  for(int i = 0; i< int(Depths.size()); i++)
  {
    // cout << "JY DINK JY IS COOLER AS EKKE? " <<  Psi[i] << endl;
    num1 = Psi[i]*weight_of_water;
    num2 = -num1*tan(friction_angle);
    
    denom = Depths[i]*weight_of_soil;
    denom_tot = denom*denom2;
    
    F_w_vec.push_back( num2/denom_tot );
  }
  
  return F_w_vec;

}


#endif