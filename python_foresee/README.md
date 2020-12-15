# FORESEE development #

Python software for predicting landslide failures based on precipitation, ground motion data and groundwater pressure. The main outputs of this model are the identified failure locations and the timing of the failure. Additional outputs such as depth of failure and factor of safety can also be obtained.

## Installation ##
1. DOCKER INSTRUCTIONS



## Command line instructions ##
If you want to know how to run the code from the command line, you will find all the instructions in the file INSTRUCTIONS.md.

## Directory Structure, Usability, I/O data ##
**ALLDATA_PROCESSING**

 Process all the input data: Inclinometers, Piezometers, Precipitation, Sentinel and Cosmo-SKYMed interferometry data.

* The inclinometers and the piezometer data must be obtained from on-site locations or purchased.
* The precipitation data is obtained from the Global Precipitation Measurement Mission by NASA, which is freely available online but requires the creation of a free account in their website. If alternative data sources are to be used instead, they must be in a .csv file, with the vector columns: Duration of precipitation (s) and Precipitation intensity (mm/s)
* The sentinel interferometry data has been provided from the University of Cantabria, processes using the ISBAS method to obtain a timeseries. Alternatively, it can be downloaded from the Sentinel-1 website, which is freely available to access.
* The Cosmo-SKYMed interferometry data must be purchased from TELESPAZIO VEGA UK.
* The DEM data can be supplied by the user in any resolution. In our case we use the EU-DEM 25 m, which is obtained freely from Copernicus Land Monitoring Service website <https://www.eea.europa.eu/data-and-maps/data/copernicus-land-monitoring-service-eu-dem>.
* The topographic slope file can be obtained from ...


1. INCLINOMETERS

    Data needs:
    * Terrestrial data: inclinometer data and coordinates in .csv format.

    * Inclinometer data recorded: see README file in Inclinometers folder for more info.

* `Make_shapefiles.py`: Loads inclinometer data. Transforms inclinometer data .csv to velocity .shp and makes sure coordinate system is consistent. Uses `functions.py`.

2. PIEZOMETERS

     Data needs:

    * Terrestrial data: piezometer data and coordinates in .csv format.
    * See piezometer README file for further info about piezometer data.

* `Make_shapefiles.py`: same as for inclinometer. Transforms the piezometer .csv into .shp with location and data of the instrument. Uses `functions.py`.

3. COMBINED SENTINEL COSMO - Sentinel and CosmoSkyMed data are processed together.

      Data needs:

    * CosmoSkyMed InSAR data: Ascending, Descending, Vertical and EW.
    * Sentinel-1 InSAR data timeseries
    * Topographic slope file with of the area of interest.

* `Combine_sentinel_cosmo.py`: Similar to process_combo.py
* `Process_combo.py`: Finds which pixels in the DEM have one or more failures and when. Uses Sentinel and Cosmo-SkyMed data. Makes a ground movement time series and saves the times of failure.
OUTPUT: .bil file with the failing pixels and the time of failure from the combination of the AD and the EWV components of Cosmo SkyMed and Sentinel-1 data.



4. PRECIPITATION

      Set of scripts that generate the precipitation data. For documentation on how to download and process the data please refer to the Precipitation folder within ALLDATA_PROCESSING.
      OUTPUT: .csv file with the time passes between consecutive precipitation events and the precipitation intensity.

NOTE: If the user does not require the Sentinel and the Cosmo-SkyMed data to be combined (as per step 2.) and instead only one of the two data sources are to be included for the calibration and the validation process, follow steps 4. or 5. accordingly. This output data will substitute consequent data inputs where InSAR data is required in Calibration, Validation or Visualisation processes.

5. InSAR_SENTINEL.

    Data needs:

  * Sentinel-1 InSAR data timeseries
  * DEM file of the area of interest.

* `Process_sentinel.py`: Find out what pixels have failures and when. Takes area of interest from the DEM provided and checks whether the pixels are inside the area. It calculates the acceleration at each point, if it is above a threshold, the point is considered as a failure. OUTPUT: .bil file with the failing pixels and the time of failure.


6. InSAR_CSK

    Data needs:

  * CosmoSkyMed InSAR data: Ascending, Descending, Vertical and EW.
  * DEM file of the area of interest.  


* `Process_insar_EWV`: This is a file to process the East-West and Vertical InSAR data, which are in the same format and have the same dates. Here, processing means finding which pixels on our DEM have one or more failures and when. Calculates the 2D displacement velocity and magnitude in each direction for all dates. If the velocity is greater than the threshold, the failure and the failure time are recorded.
OUTPUT: .bil file with the failing pixels and the time of failure from the EWV component.

* `Process_insar_AD` : This is a file to process the Ascending and Descending InSAR data. Follows the same procedure as `Process_insar_EWV`.
OUTPUT: .bil file with the failing pixels and the time of failure from the AD component.

* `Combine_insar`: Combines the .bil outputs from `Process_insar_AD` and `Process_insar_EWV`. Takes the earliest possible time when combining the failure outputs for each pixel.
OUTPUT: .bil file with the failing pixels and the time of failure from the combination of the AD and the EWV components.



**CALIBRATION**

  Data Needs:

* Calibration parameters: .csv containing Nruns, itermax, Num_cal, StartDate, EndDate, failinterval
* Parameters Iverson Monte  Carlo: .csv containing D_0, K_sat, Iz_over_K_steady, friction angle, cohesion, weight of water, weight of soil, depth
* Ground Motion InSAR Failure data (.bil format). This is the output from `Process_combo.py`.
* DEM: .bil file of the area of interest with EPSG:32633
* DEM slope: .bil file of the slope values in the area of interest with EPSG:32633
* Road file: .shp file with the outline of the road.
* Piezometer: .csv file
* Rainfall Intensity: .csv file


`Run_calibration.py`: Select the pixels based on the closest points to the road and the number of pixels that we want. Run calibration of the Iverson Model with these points to choose the optimal parameters for the simulation.
OUTPUT: .csv file with the observed and the modelled time of failure, the pixel positions, the factor of safety and the depth of failure. It also includes the chosen parameter values for each point.

**VALIDATION**

  Data Needs :

* Calibration parameters: .csv containing Nruns, itermax, Num_cal, StartDate, EndDate, failinterval
* Parameters Iverson Monte  Carlo: .csv containing D_0, K_sat, Iz_over_K_steady, friction angle, cohesion, weight of water, weight of soil, depth
* Ground Motion InSAR Failure data (.bil format). This is the output from `Process_combo.py`.
* DEM: .bil file of the area of interest with EPSG:32633
* DEM slope: .bil file of the slope values in the area of interest with EPSG:32633
* Road file: .shp file with the outline of the road.
* Piezometer: .csv file
* Calibrated points: .csv file
* Rainfall Intensity: .csv file



`Run_validation.py`: Performs the validation for the area of interest using the calibrated points, along with the InSAR data from both CosmoSkyMed and Sentinel.
OUTPUT: .csv file with the observed and the modelled time of failure, the pixel positions, the factor of safety and the depth of failure. It also includes the chosen parameter values for each point.

**VISUALISATION**

Data Needs:

* Ground Motion InSAR Failure data (.bil format). This is the output from `Process_combo.py`.
* DEM: .bil file of the area of interest with EPSG:32633
* DEM slope: .bil file of the slope values in the area of interest with EPSG:32633
* Road file: .shp file with the outline of the road.
* Calibrated points: .csv file
* Rainfall Intensity: .csv file

`Final_outputs_visualisation.py`: Generates a collection of maps and graphs representing the data:

* Map of calibrated points
* Distribution of failure times, showing calibration and validation points as well as the precipitation record.
* Plots showing the distribution of parameters with respect to height and elevation.
* Map of the validated and calibrated points. The points indicate where failure happens and whether it was predicted before, after or within a 25-day window of the observed failure.
* Zoomed-in version of the map detailed above.
* Map of validated points with a colourbar indicating the exact number of days between the observed failure events and the modelled failure event.
* Plot of the rainfall data as a function of time.
* Plot of the rainfall data along with the calibrated failure points as a function of time.
* Probability density function and histogram of the temporal distribution of failures both for the modelled and the observed failure events.
* Violin plot with the temporal distribution of modelled failures split into time intervals (violins) with respect to observed failure time distribution.

`convert_csv_to_shapefile.py`: Converts a .csv file with x,y locations in an array to a raster image of the points and a point shapefile with the desired attributes.

`voronoi_from_point_shp.py`: Uses the shapefile from `convert_csv_to_shapefile.py` as an input. The points from the shapefile are converted into a voronoi cells and saved as a multipolygon shapefile.
