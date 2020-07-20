# PPTs
Precipitation Processing Tools (PPTs) is an open source code developed by Vinícius Mesquita to download and process satellite precipitation data from NASA Tropical Rainfall Measuring Mission (TRMM) and Global Precipitation Measurement Mission (GPM)

# PPTs_FORESEE
This tool is an adapted version of the PPTs tool described above (https://github.com/lapig-ufg/PPTs). It was modified from the original code by Guillaume Goodwin (University of Edinburgh - School of GeoSciences) to fit the purposes of landslide failure modelisation within the FORESEE project. It offers less flexibility than the original PPTs tool and focuses on downloading rainfall data from GPM instead of offering various data sources. It also contains an additional module to generate time-series of rainfall intensity in over a specified area of interest, again to fit the purposes of landslide modelling.

# ACCESSING DATA
Before you try to download the data, create an account in NASA EartData website (https://urs.earthdata.nasa.gov), make login, click in Applications>Authorized Apps> Approve More Applications and select NASA GESDISC DATA ARCHIVE.


Requisites:

  * Python 3.6 or above

  * PyQt5 python package

  * Gdal python package and Gdal Binaries (if Unix system)

  * **WGET package** (If windows!, download Cygwin setup here http://cygwin.com/install.html, install with wget package and add C:\cygwin64\bin to variables of the system) - Tutorial: http://www.bloggingtips.info/install-wget-windows/. __Don't forget to add Cygwin (C:\cygwin64\bin) to the systen vauable PATH__


Recommendations:
   * Install Anaconda Python 3.6 or above (https://www.anaconda.com/download/) and the Gdal package (https://anaconda.org/conda-forge/gdal) and, for Windows users, add some system variables like:
     * PATH =  C:\ProgramData\Miniconda3; C:\ProgramData\Miniconda3\Library\bin; C:\ProgramData\Miniconda3\Scripts;
     * GDAL_DATA = C:\ProgramData\Miniconda3\Library\share\gdal
   * Before you try to download the data, create an account in NASA EartData website (https://urs.earthdata.nasa.gov), make login, click in Applications>Authorized Apps> Approve More Applications and select ***NASA GESDISC DATA ARCHIVE***.

# HOW TO RUN

Parse some arguments to ***Integration.py***:


***--ProdTP*** = 'GPM_30min' (default)

GPM_30min: GPM half-hourly (IMERGM v6)
GPM_D: GPM daily (IMERGM v6)
GPM_M: GPM monthly (IMERGM v6)

***--StartDate*** = Insert the start date

***--EndDate*** = Insert the end date

***--ProcessDir*** = Insert the processing directory path

***--SptSlc*** = Insert the cutline feature path (if not used, it assumes a global product)

***--OP*** = Call this argument if you already have the data and want to process it. Make sure you have a directory with a raw files subfolder!!!!


 **E.G.***: python PPT_CMD_RUN.py --ProdTP GPM_30min --StartDate 2018-01-01 --EndDate 2018-12-31 --ProcessDir ~./mydirectory --SptSlc ~./boundary.shp --OP


 ***UNDER CONSTRUCTION!***
