import numpy as np
#import system
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import product
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal, ogr, osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import pandas as pd
import numpy as np
#import shapefile
import itertools
import json
import os
import rasterio
from rasterio.features import shapes

import sys
import fiona

# Script to select the closest point to a given one given we are in the area of interest.
## Will need a list of points
