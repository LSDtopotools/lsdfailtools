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

#ifndef LSDIversonWrapper_H
#define LSDIversonWrapper_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same


// #include "LSDRaster.hpp"
// #include "TNT/tnt.h"
using namespace std;
// using namespace TNT;


class lsdiversonwrapper
{
  public:
  	lsdiversonwrapper() {create();}
  	lsdiversonwrapper(float talpha,float tD_0,float tK_sat, float td, float tIz_over_K_steady,
      float tfriction_angle,float tcohesion,float tweight_of_water,float tweight_of_soil, float tminimum_depth) {create( talpha, tD_0, tK_sat, td, tIz_over_K_steady,
  		 tfriction_angle, tcohesion, tweight_of_water, tweight_of_soil, tminimum_depth); }


  void set_duration_intensity(vector<float> duration_s, vector<float> this_intensity);

  void ScanTimeseriesForFailure();

  void CalculatePsiFromTimeSeries(float t);

  vector<float> calculate_steady_psi();

  vector<float> CalculatePsiDimensionalTimeTransient(float t, float T, float Iz_over_Kz);

  float CalculateResponseFunction(float t_star);

  void GetMinFS(float minimum_depth, float& depth_of_minFS, float& minFS);

  vector<float> FS();

  float F_f();

  vector<float> F_c();

  vector<float> F_w();

  void calculate_beta();

  void calculate_D_hat();

  void set_depths_vector(xt::pytensor<float,1> tdepths){Depths = tdepths;};


  protected:
  	vector<float> Depths;
    
    /// This holds the slope in radians
    float alpha;
    
    /// The  hydraulic diffusivity in m^2/s
    float D_0;
    
    /// The saturated hydraulic conductivity in m/s
    float K_sat;
    
    /// The dimensionless hydraulic diffusivity
    float D_hat;
    
    /// The depth to saturated water table at steady infiltration in metres
    float d;
    
    /// A parameter that describes the steady state pressure profile
    /// It is equal to cos^2 alpha - (Iz_over_K_steady) and alpha is the slope angle
    /// see Iverson 2000 page 1902
    float beta;
    
    /// The infiltration rate at steady state. Dimensionless since I_z is in m/s and K is in m/s
    float Iz_over_K_steady;
    
    /// The friction andgle (in radians)
    float friction_angle;
    
    /// The cohesion in Pa
    float cohesion;
    
    /// The weight of water (density times gravity; use SI units (kg/(s*m^2))
    float weight_of_water;
    
    /// The weight of soil (density times gravity; use SI units (kg/(s*m^2))
    float weight_of_soil;

    float minimum_depth;

    float current_tested_time_by_scanner;

    /// This holds the pressure heads
    vector<float> Psi;
    /// This holds the transient pressure head for each time 
    map<float, vector<float> > vec_of_Psi;
    /// When scanning for failure, saves the current factor of safety. Each value correspond to am associated depth
    vector<float> current_FS;
    float current_F_f_float;
    vector<float> current_F_c_vec;
    vector<float> current_F_w_vec;


    vector<float> durations;
    vector<float> intensities;
    vector<float> times;

    vector<float> potential_failure_times;
    vector<float> potential_failure_min_depths;
    vector<float> potential_failure_max_depths;
    vector<bool> potential_failure_bool;

    xt::pytensor<float,1> output_times;
    xt::pytensor<float,1> output_depthsFS;
    xt::pytensor<float,1> output_minFS;
    xt::pytensor<float,1> output_PsiFS;
    xt::pytensor<float,1> output_durationFS;
    xt::pytensor<float,1> output_intensityFS;
    xt::pytensor<float,1> output_failure_times;
    xt::pytensor<float,1> output_failure_mindepths;
    xt::pytensor<float,1> output_failure_maxdepths;

    xt::pytensor<float,2> output_Psi_timedepth;
    xt::pytensor<float,2> output_FS_timedepth;


  private:
    void create(); 
    void create(float talpha,float tD_0,float tK_sat,,float td,float tIz_over_K_steady,
      float tfriction_angle,float tcohesion,float tweight_of_water,float tweight_of_soil, float tminimum_depth){alpha = talpha ;D_0= tD_0;K_sat= tK_sat;d= td;
      Iz_over_K_steady= tIz_over_K_steady;friction_angle= tfriction_angle;cohesion= tcohesion;
      weight_of_water= tweight_of_water; weight_of_soil= tweight_of_soil, minimum_depth = tminimum_depth;calculate_beta();calculate_D_hat();};

};

#endif