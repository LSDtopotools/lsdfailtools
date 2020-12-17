# FORESEE development #

Python software for predicting landslide failures based on precipitation, ground motion data and groundwater pressure. The main outputs of this model are the identified failure locations and the timing of the failure. Additional outputs such as depth of failure and factor of safety can also be obtained.

## Installation ##
1. DOCKER INSTRUCTIONS:



**ALLDATA_PROCESSING**: Process all the input data: Inclinometers, Piezometers, Precipitation, Sentinel and Cosmo-SKYMed interferometry data.

* The inclinometers and the piezometer data must be obtained from on-site locations or purchased.
* The precipitation data is obtained from the Global Precipitation Measurement Mission by NASA, which is freely available online but requires the creation of a free account in their website.
* If alternative data sources are to be used instead, they must be in a .csv file, with columns indicating the duration of precipitation (s) and the precipitation intensity (mm/s).
* The sentinel interferometry data has been obtained from the University of Cantabria through TELESPAZIO VEGA.  
* The Cosmo-SKYMed interferometry data must be obtained from TELESPAZIO VEGA.



1. INCLINOMETERS

Modify `file_paths_inclinometer.json` to include paths to input and output directories.
Then run the command:

```bash
python Make_shapefiles.py
```

2. PIEZOMETERS

Modify `file_paths_piezometer.json` to include paths to input and output directories.
Then run the command:

```bash
python Make_shapefiles.py
```

3. COMBINED SENTINEL COSMO

Modify `file_paths_combined_sentinel_cosmo.json` to include paths to input and output directories.
Then run the command:

```bash
python Process_combo.py
```

3. PRECIPITATION

Please refer to the README.md file within the Precipitation folder for details of how to obtain the precipitation data.
Below is an example of the command to be used:

```bash
python PPT_CMD_RUN.py --ProdTP GPM_30min --StartDate 2018-01-01 --EndDate 2018-12-31 --ProcessDir ~./mydirectory --SptSlc ~./boundary.shp --OP
```

NOTE: If the user does not require the Sentinel and the Cosmo-SkyMed data to be combined (as per step 2.) and instead only one of the two data sources are to be included for the calibration and the validation process, follow steps 4. or 5. accordingly. This output data will substitute consequent data inputs where InSAR data is required in Calibration, Validation or Visualisation processes.

4. InSAR_SENTINEL

Modify `file_paths_insar_sentinel.json` to include paths to input and output directories.
Then run the command:

```bash
python Process_sentinel.py
```


5. InSAR_CSK

Modify `file_paths_insar_csk.json` to include paths to input and output directories.
Then run the command:

```bash
python Process_insar_AD.py
python Process_insar_EWV.py
```



**CALIBRATION**

Modify `file_paths_calibration.json` to include paths to input and output directories.
Then run the command:

```bash
python Run_calibration.py
```

**VALIDATION**

Modify `file_paths_validation.json` to include paths to input and output directories.
Then run the command:

```bash
python Run_validation.py
```


**VISUALISATION**

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
