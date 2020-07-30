

# FORESEE development

FORESEE is a Python library for dealing with word pluralization.

## Installation
1 - Follow documentation indications from `plot_iverson_lsdtt`.This will indicate how to run the MonteCarlo model and how to get the output files.
you only need to follow up to the step 2.3.1.
2 - Go to the lsdfailtools-master and folow the documentation from the README file there. This is in charge of creating and installing the cpp-python interface. This will create a wheel which can be installed via pip.
note: need to install gcc version 5+. conda install -c omgarcia gcc-6. This install version gcc (GCC) 6.1.0.
note 2: before building the wheel, you need to install conda install -c conda-forge xtensor xtensor-python
TRYING THE FOLLOWING WHILE ON THE INCLINOMETERS FOLDER

note 3: Need to also install conda install -c conda-forge numpy scipy pandas pytables matplotlib
note 4: Need to also import shapefile, pyshp, scikit-learn
note 5: At this point it tells me that osgeo is not installed. I have to install gdal via conda install -c conda-forge gdal
6 - seems to run fine. need to still install shapely and skimage. conda install -c conda-forge shapely scikit-image geopandas
7 - When installing geopandas, there was a downgrade of gdal but everything seemed to be working.

3 - Once this is all installed, we can start running the code.

Scripts to run first:
    1.- Alldata_processing: process all the input data: Inclinometers, InSAR, Piezometers, Precipitation, Sentinel, Cosmo-SKYMed.
        COMBINED SENTINEL COSMO
          Combo_functions.py: Obtains time series of ground motion data in line of sight directions. Calculates velocity and acceleration. Detects when the failure happens.
          Plots the rain along with the precipitation, cumulative displacement, velocity and acceleration for each of the pixels.
          Plots linear fit on the displacement axis for each pixel.
          Plots the rain along with the slope for each pixel.
          Plots cumulative displacement, absolute velocity and displacement acceleration.
          Combine_sentinel_cosmo.py: Seems to do similar things to process_combo.py.?
          Process_combo.py: Finds which pixels in the DEM have one or more failures and when. Uses Sentinel and Cosmo-SkyMed data.
          Loads sentinel, SkyMed, slope, and rainfall data. Makes a ground movement time series and saves the times of failure.
          Saves the first three failures for East, West, Ascending and Descending data.
        INCLINOMETERS
            Data needs:
                - Terrestrial data: inclinometer data and coordinates in .csv format.
                - Inclinometer data recorded: see README file in Inclinometers folder for more info.


            Make_shapefiles.py: Load inclinometer data. Transforms inclinometer data .csv to velocity .shp, makes sure coordinate system is consistent. Uses functions.py.
        PIEZOMETERS - measures ground water pressure
          Data needs:
            - Terrestrial data: piezometer data and coordinates in .csv format.
            - see README file for further info about piezometer data.

          Make_shapefiles.py: same as for inclinometer. Transforms the piezometer csv into shp with location and data of the instrument. Uses functions.py.

        InSAR -
          Data needs: Interferometry data. InSAR shows the deformation of the soil column at ground level. Vertical and EW directions.
           - Topography data - DEM.  


          Process_insar_EWV: This is a file to process the East-West and Vertical InSAR data, which are in the same format and seem to have the same dates.
          Here, processing means finding which pixels on our DEM have one or more failures and when. Uses Insar_functions.py. Calculates the 2D displacement velocity and magnitude in each direction for all dates. if the velocity is greater than the threshold, failure + time is recorded. This is the same as for the Sentinel-Cosmo SkyMed data.
        PRECIPITATION
          Data Needs: Data has already been downloaded. If additional data needs to be downloaded, follow the README document on the Precipitation folder. This has the instructions on how to download the data.

        SENTINEL
          Data needs: sentinel timeseries shapefile and DEM file. These include velocity and acceleration. Precipitation data as downloaded from the GPM website. InSAR data for combining sentinel with InSAR.

          Process_sentinel.py: this follows the same logic as previous processing files. Find out what pixels have failures and when. Creates sentinel failtime files.

          Combine_sentinel.py: Takes the InSAR data from A, D, EW and retains a failure if all 3 datasets show failures. Finds the dates of the failures. Creates the All-failtime files.

      2.-CALIBRATION (This is given the new code)
          Data Needs: calibration_parameters.csv: Nruns, itermax, Num_cal, StartDate, EndDate, failinterval
                      iverson_MC_parameters.csv: D_0, K_sat, Iz_over_K_steady, friction_angle, cohesion, weight_of_water, weight_of_soil, depth
                      Ground Motion Failure data: for given time interval (.bil format)
                      DEM: .bil file
                      slope file: .bil file
                      cut file: .bil file (need to check what this is)
                      Road line file: .shp file
                      Precipitation command run file: .py file for precipitation
                      Piezometer: .csv file
          Run_calibration.py: Select the pixels based on the closest points to the road and the number of pixels that we want. Run calibration with these points.
      3.-VALIDATION
          Data Needs (same as for calibration): calibration_parameters.csv
                      Iverson_MC_parameters.csv
                      Ground Motion Failure
                      DEM file
                      slope file
                      cut file
                      Road File
                      Precipitation command run file: .py file for precipitation
                      Piezometer: .csv file
                      Calibrated points: .csv file
          Run_validation.py: Loads rasters into arrays and reads calibration points and parameters. Reads also Iverson parameters. Performs and maps the validation.
        4.-VISUALISATION:
            Data Needs: InSAR failure and prefailure files in all directions.
                        DEM file (.bil)
                        Slope file (.bil)
                        Read file (.shp)
                        Calibration points file (.csv)
                        Rainfall Intensity (.csv)
           Final_outputs_visualisation.py: Maps calibrated points, failtime distribution, parameter distribution, validation map, rain data plot with associated failures, with and w/o validation data.
           Tries PCA on calibration points to see the correlation between the rain and the calibrated parameters.





TO DO:

It would potentially be useful to have a file with the file paths so that people don't need to go into the code.
Implement this: https://stackoverflow.com/questions/45507805/methods-to-avoid-hard-coding-file-paths-in-python
