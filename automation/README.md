# FORESEE Project EU Horizon 2020


Marina Ruiz Sánchez-Oro
09/12/2021


## Data input:
* `input_dir`: directory where the input data is located and where the output data will be saved.
* `coords_input_file`: file with the coordinates to test for failures. The coordinates must be in EPSG:4326. It must be a `.csv` document with the following format (keeping column names as below):

lat | lon
--- | ---
Y1.Y1 | X1.X1
Y2.Y2 | X2.X2
... | ...


* `rainfall_input_file`: file with the daily rainfall data. It must be a `.csv` document with the following format (keeping column names as below):

duration_s | intensity_mm_sec
--- | ---
86400 | xx.xx
86400 | yy.yy
... | ...

The column `duration_s` indicates that the rainfall information is daily (86400s in 24 hrs). The values of rainfall intensity must be in mm/s.

## Usage:
* `python run_full_landslide_simulation.py input_dir coords_input_file rainfall_input_file`
