//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
// test_iverson
//
// This tests the porewater pressure routines
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
// Copyright (C) 2016 Simon M. Mudd 2016
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
// either version 3 of the License, or (at your option) any later version.
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
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "../LSDPorewaterParams.hpp"
#include "../LSDPorewaterColumn.hpp"
#include "../LSDRaster.hpp"
#include "../LSDStatsTools.hpp"
using namespace std;

int main (int nNumberofArgs,char *argv[])
{

  string path_name;
  
  // load porewater parameter object
  LSDPorewaterParams LSDPP;
  
  //Test for correct input arguments
  if (nNumberofArgs == 1 )
  {
    LSDPorewaterParams TempPP;
    LSDPP = TempPP;
    
    path_name = "./";
  }
  else if (nNumberofArgs == 3 )
  {
    path_name = argv[1];
    string f_name = argv[2];

    // Make sure the path has an extension
    path_name = FixPath(path_name);

    LSDPorewaterParams TempPP(path_name,f_name);
    LSDPP = TempPP;
  }
  else
  {
    cout << "=========================================================" << endl;
    cout << "|| Welcome to the test_iverson program                 ||" << endl;
    cout << "|| This program is for testing the porewater column    ||" << endl;
    cout << "|| in LSDTopoTools. These are used for hydrology and   ||" << endl;
    cout << "|| slope stability calculations.                       ||" << endl;
    cout << "|| This program was developed by                       ||" << endl;
    cout << "|| Simon M. Mudd and Stuart W.D. Grieve                ||" << endl;
    cout << "||  at the University of Edinburgh                     ||" << endl;
    cout << "=========================================================" << endl;
    cout << "This program requires two inputs: " << endl;
    cout << "* First the path to the parameter file." << endl;
    cout << "* Second the name of the param file (see below)." << endl;
    cout << "---------------------------------------------------------" << endl;
    cout << "Then the command line argument will be, for example: " << endl;
    cout << "In linux:" << endl;
    cout << "./test_iverson.out /LSDTopoTools/Topographic_projects/Test_data/ SoilColumn.param" << endl;
    cout << "=========================================================" << endl;
    exit(EXIT_SUCCESS);
  }

  // get a steady state column    
  //cout << "I just loaded my data!!!!"<< endl;
  //LSDPP.print_parameters_to_screen();

  LSDPorewaterColumn LSD_PC(LSDPP);
  LSDPP.print_parameters_to_screen();
  
  // cout << "THe K_sat is: " << LSDPP.get_K_sat() << endl;
  // cout << "Loading some rainfall data" << endl;
  // string rainfall_fname = "MidasDoverSmall.csv";
  // vector<float> intensities;
  // vector<int> days;
  // LSDPP.parse_MIDAS_rainfall_file(path_name, rainfall_fname,days,intensities);
  

  // lets make a duration intensity record to test the model
  vector<float> intensities_new;
  // vector<float> durations_weeks;
  vector<float> durations_seconds;


  LSDPP.get_duration_intensity_from_preprocessed_input(durations_seconds, intensities_new);
  
  // intensities_new.push_back(0.5);
  // intensities_new.push_back(1.0);
  // intensities_new.push_back(0.5);
  
  // durations_weeks.push_back(5);
  // durations_weeks.push_back(6);
  // durations_weeks.push_back(4);
  
  // vector<float> durations_seconds = LSDPP.weeks_to_seconds(durations_weeks);
  
  //cout << "Durations: " <<durations_seconds[0] << "," << durations_seconds[1] << "," << durations_seconds[2] << endl;
  
  // now run the model
  // float week_of_pressure = 11;
  // float sec_of_pressure = LSDPP.weeks_to_seconds(week_of_pressure);
  // cout << "Sec of pressure = " <<  sec_of_pressure << endl;
  float sec_of_pressure = 0;
  vector<float> times;
  for(size_t i=0; i<durations_seconds.size();i++)
  {
    times.push_back(sec_of_pressure);
    sec_of_pressure = sec_of_pressure + durations_seconds[i];
    // cout << sec_of_pressure << endl;

  }

    
    
  // LSD_PC.CalculatePsiFromTimeSeries(durations_seconds, intensities_new, LSDPP, sec_of_pressure);
  
  if(!LSDPP.get_full_1D_output() && !LSDPP.get_output_2D())
  {
    cout << "You have not asked for any outputs. " << endl;
    cout << "If you want basic outputs, use the following flag: " << endl;
    cout << "get_full_1D_output: true" << endl;  
    
  }
    
  if(LSDPP.get_full_1D_output())
  {
    // vector<float> FS = LSD_PC.FS(LSDPP);
    float minimum_depth = 0.2;
    LSD_PC.ScanTimeseriesForFailure(durations_seconds, intensities_new,LSDPP, 
                                      minimum_depth, times);
  }



  if(LSDPP.get_output_2D())

  {

    // first getting the slope raster 
    LSDRaster this_alpha = LSDPP.get_alpha_raster();
    float minimum_depth = 0.2;

    // Now I have the slope, I want to feed 2 rasters with the same dimensions: min depth and FS rasters
    float ndv = -9999; int nrows=this_alpha.get_NRows(), ncols=this_alpha.get_NCols();
    Array2D<float> FSRaster(nrows,ncols,ndv);
    Array2D<float> min_depth_FSRaster(nrows,ncols,ndv);

    // Finding the time of discrete analysis
    float this_time; float wanted_time = LSDPP.get_time_of_spatial_analysis();
    bool has_been_modified = false;
    for(size_t relevant_variable_name_0 = 0; relevant_variable_name_0 < times.size() && has_been_modified == false ; relevant_variable_name_0++)
    {
      // Iterating through the time section and stopping when the wanted time is reached
      if(times[relevant_variable_name_0]>= wanted_time){this_time = times[relevant_variable_name_0]; has_been_modified = true;} // one line conditions for simple are making the code actually clearer. #Change_my_mind
      // cout << times[relevant_variable_name_0] << " wanted: " << wanted_time << endl;
    }
    // I am just checking heere that I found a time to look for
    if(has_been_modified ==  false){cout << "FATAL_ERROR::I did not find the wanted time. check your input format" << endl; exit(EXIT_FAILURE);}


    // Alright, let's do it
    int n_threads =  LSDPP.get_n_threads();
    #pragma omp parallel num_threads(n_threads)
    {
      // Entering the parallel zone 
      // Here is a small joke Declan told me:
      // 
      //
      // The barman ask the first one what he wants to drink.
      // Two std::thread enter in a bar
      //
      //
      // anyway back to business
      int i,j;
      #pragma omp for 
      for(i=0;i<nrows;i++)
      for(j=0;j<ncols;j++)
      {
        if(this_alpha.get_data_element(i,j) != this_alpha.get_NoDataValue() )
        {
          // For each of my pixels I need to create a new column
          LSDPorewaterColumn this_LSD_PC(LSDPP,i,j);

          // Get the pore pressure head to a specific time
          this_LSD_PC.CalculatePsiFromTimeSeries(durations_seconds, intensities_new, LSDPP, this_time);
          // Calculate the minimum FS at that time
          float this_min_FS, this_depth_of_min_FS;
          // min depth and FS get calculated during that part
          this_LSD_PC.GetMinFS(LSDPP, minimum_depth, this_depth_of_min_FS, this_min_FS);
          // Feeding the FFS raster
          FSRaster[i][j] = this_min_FS;
          // Feeding the min_depth raster
          min_depth_FSRaster[i][j] = this_depth_of_min_FS;

        }
      }
      // end of parallel section
    }

    cout << "I am done with the calculation, Let me save the rasters quickly" << endl;

    // I should have everything I need, let's save the ouput
    string outname = LSDPP.get_path_csv()+LSDPP.get_saving_prefix()+ "_minFS_at_" + to_string(this_time);
    LSDRaster temp_rast(this_alpha.get_NRows(), this_alpha.get_NCols(), this_alpha.get_XMinimum(), this_alpha.get_YMinimum(), this_alpha.get_DataResolution(), this_alpha.get_NoDataValue(), FSRaster, this_alpha.get_GeoReferencingStrings());
    temp_rast.write_raster(outname, "bil");

    outname = LSDPP.get_path_csv()+LSDPP.get_saving_prefix()+ "_depths_minFS_at_" + to_string(this_time);
    LSDRaster temp_rast2(this_alpha.get_NRows(), this_alpha.get_NCols(), this_alpha.get_XMinimum(), this_alpha.get_YMinimum(), this_alpha.get_DataResolution(), this_alpha.get_NoDataValue(), min_depth_FSRaster, this_alpha.get_GeoReferencingStrings());
    temp_rast2.write_raster(outname, "bil");

  }

  cout << "Finished with simulation." << endl;

}
