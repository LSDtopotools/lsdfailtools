
################################################################################
################################################################################
# Import external and internal modules

import os
import numpy as np
import pandas as bb
import datetime as dt
import matplotlib.pyplot as plt
import pickle
import shapefile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, ogr, osr
import pandas as pd
import scipy.spatial
import scipy.cluster
import shapely.geometry
import shapefile
import skimage.morphology
import sklearn.cluster
import sklearn.preprocessing

import scipy.ndimage as ndimage
from scipy.ndimage.measurements import label


################################################################################
################################################################################
def create_point(coords):
    """
    This function creates a point with ogr.

    Args:
        coords [list]: the coordinates. [lat, lon] works fine for WGS84 for example.

    Returns:
        an ogr point object

    Author: GCHG
    """
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(coords[0], coords[1])
    return pts.ExportToWkt()

################################################################################
################################################################################
def write_shapefile(fields, values, coords, EPSG, out_shp):

    # set up the shapefile driver
    driver = ogr.GetDriverByName('Esri Shapefile')
    # create the data source
    ds = driver.CreateDataSource(out_shp)
    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    # create the layer
    layer = ds.CreateLayer('', None, ogr.wkbPoint)

    # Add the fields we're interested in
    for i in range(len(fields)):

        type = type = ogr.OFTReal
        try:
            float(values[i][0])
        except ValueError:
            type = ogr.OFTString

        #if str(values[i][0]).isdigit():   type = ogr.OFTReal
        #else:   type = ogr.OFTString
        print (fields[i], str(values[i][0]), type)

        layer.CreateField(ogr.FieldDefn(fields[i], type))

    # Process the text file and add the attributes and features to the shapefile
    for i in range(len(values[0])):
      # create the feature
      feature = ogr.Feature(layer.GetLayerDefn())
      # Set the attributes using the values from the delimited text file
      for j in range(len(fields)):
          feature.SetField(fields[j], values[j][i])

      # create the WKT for the feature using Python string formatting
      wkt = "POINT(%f %f)" %  (coords[1][i] , coords[0][i])

      # Create the point from the Well Known Txt
      point = ogr.CreateGeometryFromWkt(wkt)

      # Set the feature geometry using the point
      feature.SetGeometry(point)
      # Create the feature in the layer (shapefile)
      layer.CreateFeature(feature)
      # Dereference the feature
      feature = None

    # Save and close the data source
    data_source = None


################################################################################
################################################################################
def inclino_to_one_df(df_loc, df_data):

    l = len(df_loc)
    for i in range(1,l+1):
        L = len(df_data[df_data['ID']== i])

        df_start = df_loc[df_loc['ID'] <= i]
        df_copy = df_loc[df_loc['ID'] == i]
        df_end = df_loc[df_loc['ID'] > i]

        if L > 1:
            df_start = df_start.append([df_copy]*(L-1), ignore_index = True)
        df_loc = df_start.append([df_end], ignore_index = True)

    df_data = df_data.drop('ID', axis = 1)
    df = bb.concat([df_loc,df_data], axis=1)

    return df




################################################################################
################################################################################
def inclino_to_many_shp(df, path, filename):
    """
    This  function saves the terrestrial .csv data into multiple shapefiles for mapping. It stores statistics of the data at each timestep

    Args:
        Data2D (2D numpy array): the 2D array you want a distribution for
        Nodata_value (float): The value for ignored elements

    Returns:
        bins [1D numpy array]: the value bins
        hist [1D numpy array]: the probability associated to the bins

    Author: GCHG
    """


    # Now define the dates at which you create a shapefile
    years = []; dates = list(df["DATE"])
    for y in range(len(dates)):
        ddt = dt.datetime.strptime(dates[y], "%d/%m/%Y")
        years.append(ddt.year)
    survey_years = sorted(set(years))

    # Now for each year, make some statistics and save a shapefile

    # loop over all survey years
    for y in survey_years:
        print (); print (y)
        List_to_transpose = []
        coords = []

        # Loop over all instrument IDs
        for id in range(1, max(df['ID']) + 1):
            print (id)
            A = df[df['ID']== id]

            # Select the measures taken at the year of interest
            arr = np.asarray(A['DATE'])
            indices = []
            for i in range(len(arr)):
                if arr[i].endswith(str(y)):
                    indices.append(i)
            df_selected = A.iloc[indices]

            if len(df_selected) > 0 :
                X = df_selected.drop (["NAME", "READ_TYP"], axis = 1)
                Xmed = X.median()
                List_to_transpose.append(list(Xmed) + [X['DATE'].iloc[0]]  )
                coords.append([Xmed['LATITUDE'], Xmed['LONGITUDE']])

                X = X.drop (["DATE"], axis = 1)
                fields = list(X.columns.values) + ['DATE']

        final_list = [list(i) for i in zip(*List_to_transpose)] # list transposition
        final_coords = [list(i) for i in zip(*coords)] # list transposition


        write_shapefile (fields, final_list, final_coords, 32633, path + str(y)+'.shp')

        print ('made the shapefile here:',  path + 'Inclino_' + str(y)+'.shp')



################################################################################
################################################################################
def inclino_to_velocity_shp(df, path, filename, EPSG):
    """
    This  function saves the terrestrial .csv data into multiple shapefiles for mapping. It stores statistics of the data at each timestep

    Args:
        Data2D (2D numpy array): the 2D array you want a distribution for
        Nodata_value (float): The value for ignored elements

    Returns:
        bins [1D numpy array]: the value bins
        hist [1D numpy array]: the probability associated to the bins

    Author: GCHG
    """

    All_df = bb.DataFrame()
    # Loop over all instrument IDs
    for id in range(1, max(df['ID']) + 1):
        print (id)
        A = df[df['ID'] == id]

        # Find the first and last measures
        A_ref = A[A['READ_NUM'] == 0]
        start_date = dt.datetime.strptime( A_ref['DATE'].iloc[0], '%d/%m/%Y')
        A_last= A[A['READ_NUM'] == max(A['READ_NUM'])]
        end_date = dt.datetime.strptime( A_last['DATE'].iloc[0], '%d/%m/%Y')

        # Deduce the values of Cum_Disp and their respective Az values to obtain a total movement vector.
        # Divide by the time between survey dates to get a velocity vector
        survey_duration = end_date - start_date
        survey_duration = survey_duration.total_seconds() / (3600 * 24 * 365.)

        velocity = ( np.asarray(A_last['CUM_DISP']) -  np.asarray(A_ref['CUM_DISP']) ) / survey_duration
        vel_rotation = np.asarray(A_last['AZ_CUM']) -  np.asarray(A_ref['AZ_CUM'])

        # Make a df
        Df = A_last
        Df = Df.drop (["NAME", "WORKING", "READ_TYP"], axis = 1)
        Df['REF_DATE'] = start_date
        Df['VELOCITY'] = velocity
        Df['ROTATION'] = vel_rotation


        All_df = All_df.append(Df, ignore_index = True)

    # Fiw the date formats
    All_df['DATE'] = All_df['DATE'].astype(str)
    All_df['REF_DATE'] = All_df['REF_DATE'].astype(str)


    #prepare for shapefilisation
    fields = All_df.columns.values
    data = list(All_df.values)
    data = [list(i) for i in zip(*data)]
    lat = All_df["LATITUDE"]
    lon = All_df["LONGITUDE"]
    coords = [list(lat),list(lon)]
    write_shapefile (fields, data, coords, 4326, path + 'velocity.shp')
    print ('made the shapefile here:',  path + 'velocity_'+str(start_date) + "_"+ str(end_date) +'.shp')



################################################################################
################################################################################
def piezo_to_shp(df, path, filename):
    """
    This  function saves the terrestrial .csv data into multiple shapefiles for mapping. It stores statistics of the data at each timestep

    Args:
        Data2D (2D numpy array): the 2D array you want a distribution for
        Nodata_value (float): The value for ignored elements

    Returns:
        bins [1D numpy array]: the value bins
        hist [1D numpy array]: the probability associated to the bins

    Author: GCHG
    """
    # Now define the dates at which you create a shapefile
    years = []; dates = list(df["DATE"])
    for y in range(len(dates)):
        ddt = dt.datetime.strptime(dates[y], "%d/%m/%Y")
        years.append(ddt.year)
    survey_years = sorted(set(years))

    # Now for each year, make some statistics and save a shapefile
    # loop over all survey years
    for y in survey_years:
        print (); print (y)
        List_to_transpose = []
        coords = []

        # Loop over all instrument IDs
        print (id)
        A = df

        # Select the measures taken at the year of interest
        arr = np.asarray(A['DATE'])
        indices = []
        for i in range(len(arr)):
            if arr[i].endswith(str(y)):
                indices.append(i)
        df_selected = A.iloc[indices]

        X = df_selected.drop (["Unnamed: 12", "Unnamed: 13"], axis = 1)

        fields = list(X.columns.values)
        data = list(X.values)
        data = [list(i) for i in zip(*data)]
        lat = X["LATITUDE"]
        lon = X["LONGITUDE"]
        coords = [list(lat),list(lon)]

        write_shapefile (fields, data, coords, 4326, path + 'Piezo_' + str(y)+'.shp')
        print ('made the shapefile here:',  path + 'Piezo_' + str(y)+'.shp')





################################################################################
################################################################################
def buildShapefile(input_dir, shapefile_out, samples_per_pc = 100):
    '''
    '''

    #ds_ChangeIDs = [gdal.Open(infile, 0) for infile in sorted(glob.glob(input_dir+'/ChangeID_'+str(y1)+'_'+str(y2)+'_*.tif'))]
    ds_ChangeIDs = [gdal.Open(infile, 0) for infile in sorted(glob.glob(input_dir+'/ChangeID.tif'))]

    shapefile_out = os.path.abspath(os.path.expanduser(shapefile_out))

    if type(ds_ChangeIDs) != list: ds_ChangeIDs = [ds_ChangeIDs]

    driver = ogr.GetDriverByName("ESRI Shapefile")

    if os.path.exists(shapefile_out):
        driver.DeleteDataSource(shapefile_out)

    outDatasource = driver.CreateDataSource(shapefile_out)
    srs = osr.SpatialReference()
    srs.ImportFromWkt( ds_ChangeIDs[0].GetProjectionRef() )

    # Create output layer
    outLayer = outDatasource.CreateLayer(shapefile_out, srs)

    # ChangeID output field
    newField = ogr.FieldDefn('ChangeID', ogr.OFTInteger)
    outLayer.CreateField(newField)
    outField = outLayer.GetLayerDefn().GetFieldIndex("ChangeID")


    sample_vals = []
    for ds_ChangeID in ds_ChangeIDs:
        if samples_per_pc is None:
            N_samples = 0 # Number of polygons
        else:
            N_samples = samples_per_pc

        # Build ChangeID mask. Select random sample of change events
        ds_mask = getMask(ds_ChangeID)

        gdal.Polygonize(ds_ChangeID.GetRasterBand(1), ds_mask.GetRasterBand(1), outLayer, outField, [], callback=None )

        sample_vals.extend(getSample(ds_ChangeID, samples_per_pc = N_samples))

    #sample = np.isin(np.array([feature.GetField('ChangeID') for feature in outLayer]), np.array(sample_vals))

    # Sub-sample output field
    sampleField = ogr.FieldDefn('sample', ogr.OFTInteger)
    outLayer.CreateField(sampleField)

    count = 0
    for feature in outLayer:
        count +=1
        outLayer.SetFeature(feature)
        if feature.GetField('ChangeID') in sample_vals:
            feature.SetField('sample', 1)
        else:
            feature.SetField('sample', 0)

        feature.SetField('ChangeID', count)

        outLayer.SetFeature(feature)

    outDatasource = None

    return shapefile_out



################################################################################
################################################################################
def outputGeoTiff(data, filename, geo_t, proj, output_dir = os.getcwd(), dtype = 6, nodata = None):
    """
    Writes a GeoTiff file to disk.

    Args:
        data: A numpy array.
        geo_t: A GDAL geoMatrix (ds.GetGeoTransform()).
        proj: A GDAL projection (ds.GetProjection()).
        filename: Specify an output file name.
        output_dir: Optioanlly specify an output directory. Defaults to working directory.
        dtype: gdal data type (gdal.GDT_*). Defaults to gdal.GDT_Float32.
        nodata: The nodata value for the array
    """


    # Get full output path
    output_path = '%s/%s.tif'%(os.path.abspath(os.path.expanduser(output_dir)), filename.rstrip('.tif'))

    # Save image with georeference info
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_path, data.shape[1], data.shape[0], 1, dtype, options = ['COMPRESS=LZW'])
    ds.SetGeoTransform(geo_t)
    ds.SetProjection(proj)

    # Set nodata
    if nodata != None:
        ds.GetRasterBand(1).SetNoDataValue(nodata)

    # Write data for masked and unmasked arrays
    if np.ma.isMaskedArray(data):
        ds.GetRasterBand(1).WriteArray(data.filled(nodata))
    else:
        ds.GetRasterBand(1).WriteArray(data)
    ds = None
