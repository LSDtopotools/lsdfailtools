# FORESEE development #

FORESEE is a Python software for predicting landslide failures based on precipitation, ground motion data and groundwater pressure.

## Installation ##
1. Follow documentation indications from `plot_iverson_lsdtt`.This will indicate how to run the MonteCarlo model and how to get the output files. Only follow up to step 2.3.1.
2. Go to the lsdfailtools-master and follow the documentation from the README file there. This is in charge of creating and installing the cpp-python interface. This will create a wheel which can be installed via pip.
Note: you will need to install gcc version 5+ if you have an older version:
`conda install -c omgarcia gcc-6`
This install version gcc (GCC) 6.1.0.
Note #2: before building the wheel, you need to may need to install xtensor:
`conda install -c conda-forge xtensor xtensor-python`

3. Install the following packages using `conda install -c conda-forge <package_name>`:
* `numpy`
* `scipy `
* `pandas`
* `pytables`
* `matplotlib`
* `pyshp`
* `scikit-learn`
* `gdal`
* `shapely`
* `scikit-image`
* `geopandas`

(When installing geopandas, there is a downgrade of gdal but everything should still be working).

3. Once this is all installed, we can start running the code.

Scripts to run first:

1. Alldata_processing: process all the input data: Inclinometers, InSAR, Piezometers, Precipitation, Sentinel, Cosmo-SKYMed.

    1. COMBINED SENTINEL COSMO

        * `Combo_functions.py`: Obtains time series of ground motion data in line of sight directions. Calculates velocity and acceleration. Detects when the failure happens.
        Plots the rain along with the precipitation, cumulative displacement, velocity and acceleration for each of the pixels.
        Plots linear fit on the displacement axis for each pixel.
        Plots the rain along with the slope for each pixel.
        Plots cumulative displacement, absolute velocity and displacement acceleration.

        * `Combine_sentinel_cosmo.py`: Similar to process_combo.py

        * `Process_combo.py`: Finds which pixels in the DEM have one or more failures and when. Uses Sentinel and Cosmo-SkyMed data.
        Loads sentinel, Cosmo-SkyMed, slope, and rainfall data. Makes a ground movement time series and saves the times of failure.
        Saves the first three failures for East, West, Ascending and Descending data.

    2. INCLINOMETERS

        Data needs:

            * Terrestrial data: inclinometer data and coordinates in .csv format.
            * Inclinometer data recorded: see README file in Inclinometers folder for more info.


        * `Make_shapefiles.py`: Loads inclinometer data. Transforms inclinometer data .csv to velocity .shp, makes sure coordinate system is consistent. Uses functions.py.

    3. PIEZOMETERS - measure ground water pressure

      Data needs:

        * Terrestrial data: piezometer data and coordinates in .csv format.
        * See piezometer README file for further info about piezometer data.

      * `Make_shapefiles.py`: same as for inclinometer. Transforms the piezometer .csv into .shp with location and data of the instrument. Uses functions.py.

    4. InSAR - shows the deformation of the soil column at ground level. Vertical and EW directions.

      Data needs:

        * Interferometry data.
        * Topography data - DEM.  


      * `Process_insar_EWV`: This is a file to process the East-West and Vertical InSAR data, which are in the same format and have the same dates.
      Here, processing means finding which pixels on our DEM have one or more failures and when. Uses Insar_functions.py. Calculates the 2D displacement velocity and magnitude in each direction for all dates. If the velocity is greater than the threshold, failure + failure time is recorded. This is the same as for the Sentinel-Cosmo SkyMed data.
      * `Process_insar_AD` : This is a file to process the Ascending and Descending InSAR data. Follows the same procedure as `Process_insar_EWV`.

    5. PRECIPITATION

      Data Needs:

        * Precipitation Data (has already been downloaded). If additional data needs to be downloaded, follow the README document on the Precipitation folder. This has the instructions on how to download the data.

    6. SENTINEL (this can possibly be deleted and use COMBINED SENTINEL COSMO)

      Data needs:

        * Sentinel timeseries shapefile
        * DEM file. These include velocity and acceleration.
        * Precipitation data as downloaded from the GPM website.
        * InSAR data for combining sentinel with InSAR.

      * `Process_sentinel.py`: this follows the same logic as previous processing files. Find out what pixels have failures and when. Creates sentinel failtime files.

      * `Combine_sentinel.py`: Takes the InSAR data from A, D, EW and retains a failure if all 3 datasets show failures. Finds the dates of the failures. Creates the All-failtime files.

  2. CALIBRATION

      Data Needs:

        * calibration_parameters.csv: Nruns, itermax, Num_cal, StartDate, EndDate, failinterval
        * iverson_MC_parameters.csv: D_0, K_sat, Iz_over_K_steady, friction_angle, cohesion, weight_of_water, weight_of_soil, depth
        * Ground Motion Failure data: for given time interval (.bil format)
        * DEM: .bil file
        * slope file: .bil file
        * cut file: .bil file (need to check what this is)
        * Road line file: .shp file
        * Precipitation command run file: .py file for precipitation
        * Piezometer: .csv file

      * `Run_calibration.py`: Select the pixels based on the closest points to the road and the number of pixels that we want. Run calibration with these points.

  3. VALIDATION

      Data Needs (same as for calibration):

        * calibration_parameters.csv
        * Iverson_MC_parameters.csv
        * Ground Motion Failure
        * DEM file
        * Slope file
        * Cut file
        * Road File
        * Precipitation command run file: .py file for precipitation
        * Piezometer: .csv file
        * Calibrated points: .csv file

      * `Run_validation.py`: Loads rasters into arrays and reads calibration points and parameters. Reads also Iverson parameters. Performs and maps the validation.

  4. VISUALISATION

      Data Needs:

        * InSAR failure and prefailure files in all directions.
        * DEM file (.bil)
        * Slope file (.bil)
        * Read file (.shp)
        * Calibration points file (.csv)
        * Rainfall Intensity (.csv)

     * `Final_outputs_visualisation.py`: Maps calibrated points, failtime distribution, parameter distribution, validation map, rain data plot with associated failures, with and w/o validation data.
     Tries PCA on calibration points to see the correlation between the rain and the calibrated parameters.
