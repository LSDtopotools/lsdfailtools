# taken from exaple code in https://towardsdatascience.com/how-to-create-voronoi-regions-with-geospatial-data-in-python-adbb6c5f2134

import json
import fiona
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from shapely.geometry import mapping, Polygon

from geovoronoi import voronoi_regions_from_coords, points_to_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

with open("file_paths_visualisation.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)

print("The base output directory is {}".format(FILE_PATHS["figures_dir"]))


# point shapefile with the validation locations
shpdir = FILE_PATHS["figures_dir"]

gdf = gpd.read_file(shpdir + "validation_csv_to_shapefile.shp")

AoI = FILE_PATHS["AoI_file"]

boundary = gpd.read_file(AoI)

# need to change the projection from 32633 to 4326
gdf = gdf.set_crs(epsg=32633, inplace = True)
gdf_proj =gdf.to_crs(epsg=4326)

# set the projection of the AoI shapefile
boundary = boundary.set_crs(epsg=4326, inplace = True)

boundary_shape = cascaded_union(boundary.geometry)
coords = points_to_coords(gdf_proj.geometry)

# Calculate Voronoi Regions
poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape)#, farpoints_max_extend_factor=200)

# plot the voronoi regions in a map
fig, ax = subplot_for_map()
plot_voronoi_polys_with_points_in_area(ax, boundary_shape, poly_shapes, pts, poly_to_pt_assignments)
ax.set_title('Voronoi regions')
plt.tight_layout()
plt.savefig(shpdir + 'Voronoi_polygons.png')


# save the polygon objects in a shapefile

time_of_failure = gdf['Z']
poly = poly_shapes


# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'time_of_failure': 'int'},
}

# Write a new Shapefile
with fiona.open(shpdir+'voronoi_polygons.shp', 'w', 'ESRI Shapefile', schema) as c:
    for i in range(len(poly_shapes)):
        c.write({
            'geometry': mapping(poly[i]),
            'properties': {'time_of_failure': time_of_failure[i]},
        })
