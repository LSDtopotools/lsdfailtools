# PPTs
Precipitation Processing Tools (PPTs) it's an open source code developed by Vinícius Mesquita to download and process satellite precipitation data from NASA Tropical Rainfall Measuring Mission (TRMM) and Global Precipitation Measurement Mission (GPM)

# PPTs_FORESEE
This tool is an adapted version of the PPTs tool described above (https://github.com/lapig-ufg/PPTs). It was modified from the original code by Guillaume Goodwin (University of Edinburgh - School of GeoSciences) to fit the purposes of landslide failure modelisation within the FORESEE project. It offers less flexibility than the original PPTs tool and focuses on downloading rainfall data from GPM instead of offering various data sources. It also contains an additional module to generate time-series of rainfall intensity in over a specified area of interest, again to fit the purposes of landslide modelling.

# ACCESSING DATA
Before you try to download any data, ensure that you have followed the steps in the Docker installation entering your NASA username and password

# HOW TO RUN

Run using:

```bash
python PPT_CMD_RUN.py --ProdTP GPM_30min --StartDate 2018-01-01 --EndDate 2018-12-31 --ProcessDir /path/to/directory --SptSlc /path/to/shapefile.shp --OP
```
Where, 

***--ProdTP*** = 'GPM_30min' (default)

GPM_30min: GPM half-hourly (IMERGM v6)
GPM_D: GPM daily (IMERGM v6)
GPM_M: GPM monthly (IMERGM v6)

***--StartDate*** = Insert the start date

***--EndDate*** = Insert the end date

***--ProcessDir*** = Insert the processing directory path

***--SptSlc*** = Insert the cutline feature path (if not used, it assumes a global product)

***--OP*** = Call this argument if you already have the data and want to process it. Make sure you have a directory with a raw files subfolder!!!!

This will generate a csv file titled StartDate + "to" + EndDate + "_Intensity.csv" e.g. 2018-01-01_to_2018-01-02_Intensity.csv


# HOW TO ADD ON MORE DATA

If you have more data you would like to add directly following time period generated by the above script you can do so using the `Add_forecast.py` script. 

Firstly make sure the data is in the correct format, it should be a csv file with 2 columns: duration_s and intensity_mm_sec. If in any doubt refer to the format of the csv file generated by `PPT_CMD_RUN.py`. The format has to be the same!

The output csv should be named as below depending on the date range of the final combined file:
StartDate + "to" + EndDate + "_Intensity.csv" e.g. 2018-01-01_to_2018-01-03_Intensity.csv

Run using

```bash
python Add_forecast.py --base_rainfall /path/to/base_rainfall.csv --supp_rainfall /path/to/rainfall_to_be_added.csv --output_file /path/to/output_file.csv
```
Where:

***--base_rainfall*** = the pathway to and including the base rainfall csv file

***--supp_rainfall*** = the pathway to and including the additional rainfall csv file

***--supp_rainfall*** = the pathway to and including the additional rainfall csv file

You can then change the `file_paths_combined_sentinel_cosmo.json` and `file_paths_visualisation.json` to update the rainfall file used in other analyses to this new combined file.
