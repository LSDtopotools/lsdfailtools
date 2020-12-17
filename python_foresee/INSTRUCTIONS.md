# FORESEE #
## Installation ##
DOCKER INSTRUCTIONS:

**INPUTS**:

* **Piezometer data**: .csv file with piezometer readings and the information about where they are located. They must be obtained from on-site locations or purchased.

Piezometer reading example (.csv file):

ID=unique identifier

DATE = reading date

READING NUMBER = progressive number of reading with time

FF = Depth of hole bottom

LIV = Depth of water from ground level - When the reading is 'dry' the number 999 is used

|ID	|DATE|	READ_NUM	|FF1	|LIV1|	FF2	|LIV2|	FF3	|LIV3|	FF4|	LIV4|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|7|	23/10/2014|	0|	3.7	|3.5|	6.2	|999|	6.1	|999|	9.6	|5.5|
|7|	24/06/2016|	1	|3.5|	3.5	|11.9|	7.4|	18.5	|7.8	|9.8	|3.7|
|7|	15/05/2017|	2|	3.5	|3.5|	11.8	|7.4	|18.1	|7.4	|9.8	|3.8|

Piezometer location information example (.csv file):

ID=unique identifier

NAME = Piezometer name

PROGR_KM = KP of A16 motorway from "Autostrade"

CARR = Carriageway (East or West) from "Autostrade"

LONGITUDE, LATITUDE = coordinates in decimal degrees in WGS84

ELEVATION = elevation in meters from "Autostrade"

|ID	|NAME|	PROGR_KM|	Carr|	LONGITUDE|	LATITUDE	|ELEVATION	|LENGTH|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|1|	PzA	|89+500|	E|	15.145169|	41.090146|	440|	25.6|
|2|	Pz4	|89+500|	E	|15.144353|	41.090037|	447	|23.8|
|3|	Pz23	|89+500	|E	|15.14341	|41.088143	|476|	29.9|





* **Precipitation data**: obtained from the Global Precipitation Measurement Mission by NASA, which is freely available online but requires the creation of a free account in their website. If alternative data sources are to be used instead, they must be in a .csv file, with columns indicating the duration of precipitation (s) and the precipitation intensity (mm/s).

Example:

|duration_s	|intensity_mm_sec|
| ----------- | ----------- |
|86400|	0|
|86400|	2.26E-07|
|86400|	1.99E-06|
|86400|	8.75E-07|



* **Sentinel interferometry** :  Shapefile containing a time series of displacement derived from an ISBAS analysis of Sentinel-1 images. For further information on the data format access the file: FORESEE_D2.3_TimeSeries_sentinel1_CaseStudy2.pdf.

* **Cosmo-SKYMed interferometry**: Shapefiles containing a time series of displacement derived from PSP-IfSAR analysis of COSMO-SkyMed images. Ascending, descending, East-West and Vertical components required. For further information on the data format for each of the components, access the files: FORESEE_D2.7_TimeSeries_A_CSK_CaseStudy2.pdf, FORESEE_D2.7_TimeSeries_D_CSK_CaseStudy2.pdf, FORESEE_D2.7_TimeSeries_EW_CSK_CaseStudy2.pdf, FORESEE_D2.7_TimeSeries_VERT_CSK_CaseStudy2.pdf.

* **Area of Interest**: Polygon shapefile outlining the area of interest where the calibration and validation points will be sampled from. Required projection: EPSG 4326

* **Digital Elevation Model (DEM)**: .bil raster file containing the elevation of the area of interest with coordinate system in EPSG:32633 (UTM zone 33N). In our case we use the EU-DEM 25 m which is obtained freely from the Copernicus Land Monitoring Service <https://www.eea.europa.eu/data-and-maps/data/copernicus-land-monitoring-service-eu-dem>. Alternatively, a 10m DEM can be found on the Tinitaly website <http://tinitaly.pi.ingv.it/>. However, other resolutions are also accepted by the model, although it will affect the calibration and validation process.

* **Topographic slope**: .bil raster file containing the slope of the area of interest with coordinate system in EPSG:32633 (UTM zone 33N). This file can be derived from the DEM using ArcMap or open source software such as LSDTopoTools.

* **Road**: line shapefile with the outline of the road of interest for the study. Required projection: EPSG:32633 (UTM zone 33N).

* **Monte Carlo parameters**: .csv file with the ranges of the parameters used in the Monte Carlo simulation. The first row of values corresponds to the minimum values and the second row to the maximum values.

Example file:

|D_0|	K_sat|	Iz_over_K_steady	|friction_angle	|cohesion|	weight_of_water|	weight_of_soil|	depth|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|0.000001	|0.00000001|	0.1|	0.2	|5000|	9800	|15000|	0.1|
|0.0001	|0.000001|	0.8|	0.5|	20000|	9800	|25000	|3|

Variable names and units:

|PARAMETER NAME| UNITS|
| ----------- | ----------- |
|HYDRAULIC DIFFUSIVITY (D_0)| $m^{2} s^{-1}$|
|HYDRAULIC CONDUCTIVITY (K_sat)|$m s^{-1}$ |
|STEADY STATE (LONG TERM) WATER BALANCE (IZ/KZ) (Iz_over_K_steady)|dimensionless |
|TANGENT OF FRICTION ANGLE| dimensionless|
|COHESION |$Pa$|
|COLUMN WEIGHT OF THE WATER (DENSITY TIME GRAVITATIONAL ACCELERATION)| $m^{-2} kg s^{-2}$ |
|VOLUME WEIGHT OF THE SOIL (DENSITY TIME GRAVITATIONAL ACCELERATION)| $m^{-2} kg s^{-2}$ |
|SOIL COLUMN DEPTH |$m$|





* **Calibration parameters**: .csv file including the number of Monte Carlo runs (Nruns), the maximum number of iterations of the Monte Carlo process (itermax), the number of points to calibrate (Num_cal), the start (StartDate) and end date (EndDate) of the timeseries which correspond to the length of the precipitation record, and the failure interval (failinterval) which is the  accepted time window (in days) to simulate acceptable failure times.

Example file:

|Nruns|	itermax	|Num_cal|	StartDate|	EndDate	|failinterval|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|25	|50	|200	|01/01/2014|	31/12/2019|	25|





**OUTPUTS**:

* **Calibration .csv file** (see table below for example).
Contains the calibrated parameter values for the calibrated pixels (location given by row,col) as well as the modelled time of failure, the factor of safety, the depth of failure and the observed failure time.


| | alpha  |   D_0 |    K_sat  |d|Iz_over_K_steady|friction_angle|cohesion|weight_of_water|weight_of_soil|time_of_failure|factor_of_safety|min_depth|S|Z|row|col|observed_failtime|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|0|	0.076776527|	1.25E-05|	3.16E-07|	3.236842105	|0.635793647	|0.354652987	|12032.7136	|9800|	19356.08113	|97977600|	-0.74747467|	0.100000001	|0.07677653	|547.6984|	369|	562|	96422400
|1|	0.272359937	|7.71E-06|	1.68E-07|	3.236842105	|0.629424281	|0.263523691|	9599.602129	|9800.473225|	11034.71452	|70243200|	0.567260742|	0.100000001	|0.27235994	|441.05658	|431|	648	|71539200|
|2|	0.170809358	|2.23E-06|	6.16E-08|	3.236842105	|0.232207709|	0.426850153	|11188.8693	|9800|	17405.85364	|114998400|	0.303287506	|0.100000001|	0.17080936|	528.0549|	437	|825|	114393600|

* **Validation .csv file** (see table below for example).

Contains the validated parameter values for the validated pixels (location given by row,col) as well as the modelled time of failure, the factor of safety, the depth of failure and the observed failure time.


|alpha|	D_0	|K_sat|	d|	Iz_over_K_steady|	friction_angle|	cohesion|	weight_of_water|	weight_of_soil|	time_of_failure|	factor_of_safety|	min_depth|	S|	Z|	row	|col|	observed_failtime|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|0.050537445|	4.64E-06|	2.21E-08|	3.236842105	|0.241673545|	0.200190446	|12116.30719|	9800.851942	|16740.39976|	100224000|	-0.557540894	|0.100000001|	0.050537445	|652.6312256	|4	|690|	16588800|
|0.058242787|	4.64E-06|	2.21E-08|	3.236842105	|0.241673545	|0.200190446	|12116.30719|	9800.851942	|16740.39976|	100224000|	-0.467391968|	0.100000001	|0.058242787|	655.8518066	|5	|687|	87091200|
|0.034425307|	1.45E-05	|8.70E-08|	3.236842105	|0.136849459	|0.291813387	|17356.14574|	9800	|18178.94782|	75254400|	-2.77532959	|0.100000001|	0.034425307	|684.3273315|	14	|770|	91756800|

* **Calibration shapefile**

Multipolygon shapefile with the calibration points from the .csv file transformed into Voronoi polygons (using Voronoi tessellation). The attributes of each polygon are the time of failure, the factor of safety and the depth of failure. Coordinate system: EPSG:4326.

* **Validation shapefile**

Multipolygon shapefile with the validation points from the .csv file transformed into Voronoi polygons (using Voronoi tessellation). The attributes of each polygon are the time of failure, the factor of safety and the depth of failure. Coordinate system: EPSG:4326.



**ALLDATA_PROCESSING**: Process all the input data: Piezometers, Precipitation, Sentinel and Cosmo-SKYMed interferometry data.


1. PIEZOMETERS

Data input:

* Piezometer data: piezometer data and coordinates in .csv format.

Data output:
* .shp with location and data of the piezometer


Modify `file_paths_piezometer.json` to include paths to input and output directories.
Then run the command:

```bash
python Make_shapefiles.py
```

2. PRECIPITATION

Data input:
* Area of interest

Data output:
* .csv file with the duration of precipitation events and the precipitation intensity.

Please refer to the README.md file within the Precipitation folder for details of how to obtain the precipitation data.
Below is an example of the command to be used:

```bash
python PPT_CMD_RUN.py --ProdTP GPM_30min --StartDate 2018-01-01 --EndDate 2018-12-31 --ProcessDir ~./mydirectory --SptSlc ~./boundary.shp --OP
```

3. COMBINED SENTINEL COSMO

Data input:

* CosmoSkyMed InSAR data: Ascending, Descending, Vertical and EW.
* Sentinel-1 InSAR data.
* Topographic slope file.

Data output:
* .bil file with the failing pixels and the time of failure from the combination of the AD and the EWV components of Cosmo SkyMed and Sentinel-1 data.


Modify `file_paths_combined_sentinel_cosmo.json` to include paths to input and output directories.
Then run the command:

```bash
python Process_combo.py
```



**NOTE**: If the user does not require the Sentinel and the Cosmo-SkyMed data to be combined (as per step 2.) and instead only one of the two data sources are to be included for the calibration and the validation process, follow steps 4. or 5. accordingly. This output data will substitute consequent data inputs where InSAR data is required in Calibration, Validation or Visualisation processes.

4. InSAR_SENTINEL

Data input:

* Sentinel-1 InSAR data
* DEM file

Data output:
* .bil file with the failing pixels and the time of failure.

Modify `file_paths_insar_sentinel.json` to include paths to input and output directories.
Then run the command:

```bash
python Process_sentinel.py
```


5. InSAR_CSK

Data input:

* CosmoSkyMed InSAR data: Ascending(A), Descending(D), Vertical(V) and East-West(W).
* DEM file of the area of interest.

Data output:

* .bil file with the failing pixels and the time of failure from the combination of the AD and the EWV components.

Modify `file_paths_insar_csk.json` to include paths to input and output directories.
Then run the command:

```bash
python Process_insar_AD.py
python Process_insar_EWV.py
python Combine_insar.py
```



**CALIBRATION**

Data input:
* Calibration parameters:
* Monte Carlo parameters:
* Ground Motion InSAR Failure data (.bil format). This is the output from `Process_combo.py`.
* DEM
* Topographic slope
* Road file
* Piezometer data
* Rainfall Intensity data

Data output:

* .csv file with the observed and the modelled time of failure, the pixel positions, the factor of safety and the depth of failure. It also includes the chosen parameter values for each point.

Modify `file_paths_calibration.json` to include paths to input and output directories.
Then run the command:

```bash
python Run_calibration.py
```

**VALIDATION**

Data input:

* Calibration parameters
* Monte  Carlo parameters
* Ground Motion InSAR Failure data (.bil format). This is the output from `Process_combo.py`.
* DEM
* Topographic slope: .bil file of the slope values in the area of interest with EPSG:32633
* Road file
* Piezometer
* Calibrated points (.csv file). This is the output from `Run_calibration.py`.
* Rainfall Intensity

Data output:

* .csv file with the observed and the modelled time of failure, the pixel positions, the factor of safety and the depth of failure. It also includes the chosen parameter values for each point.

Modify `file_paths_validation.json` to include paths to input and output directories.
Then run the command:

```bash
python Run_validation.py
```


**VISUALISATION**

Data input:

* Ground Motion InSAR Failure data (.bil format). This is the output from `Process_combo.py`.
* DEM
* Topographic slope
* Road file
* Calibrated points: .csv file. This is the output from `Run_calibration.py`
* Validated points: .csv file. This is the output from `Run_validation.py`
* Rainfall Intensity
* Calibration parameters

Data output:

* Shapefile with Voronoi polygons of the points obtained from the calibration and/or the validation phases. 
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


Modify `file_paths_visualisation.json` to include paths to input and output directories.

Then run the commands:

```bash
python Final_outputs_visualisation.py
python map_validation.py
python map_validation_zoom.py
python map_validation_colourbar.py
```



To convert the .csv output from the validation (or calibration) file into a point shapefile each attribute that needs to be included must be processed separately. In our case we have the `time_of_failure`, `factor_of_safety` and `depth` attributes to include, which correspond to certain columns in the .csv file. The following command must be run in this case 3 times. Each time the `file_paths_visualisation.json` file must be updated with the attribute of interest and its column number in the corresponding .csv file (note this is 0-indexed).

```bash
python convert_csv_to_shapefile.py
```

To convert from point shapefiles to a multipolygon shapefile using Voronoi tessellation. This assumes that the attributes are `time_of_failure`, `factor_of_safety` and `depth`.

```bash
python voronoi_with_attributes.py
```