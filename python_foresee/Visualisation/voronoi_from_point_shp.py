# taken from exaple code in https://towardsdatascience.com/how-to-create-voronoi-regions-with-geospatial-data-in-python-adbb6c5f2134

import numpy as np
import json
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from geovoronoi import voronoi_regions_from_coords, points_to_coords

with open("../file_with_paths.json") as file_with_paths :
    FILE_PATHS = json.load(file_with_paths)
# point shapefile with the validation locations
faildir = FILE_PATHS["interferometry_out_dir"]
shpdir = FILE_PATHS["rain_intensity_caliv_valid"]

gdf = gpd.read_file(shpdir + "points_val_shapefile.shp")
print(gdf.head())
print(gdf.dtypes)
topodir = FILE_PATHS["topo_dir"]
AoI = topodir + "AoI.shp"

# Import area of interest in Italy for region clipping
#AoI = gpd.read_file(topodir + "AoI.shp")

boundary = gpd.read_file(AoI)
'''
fig, ax = plt.subplots(figsize=(12, 10))
boundary.plot(ax=ax, color='gray')
gdf.plot(ax=ax, markersize=3.5, color = 'black')
ax.axis('off')
plt.axis('equal')
plt.show()
'''

#4326
boundary = boundary.set_crs(epsg=4326, inplace = True)
gdf_proj = gdf.set_crs(epsg=4326, inplace = True)

boundary_shape = cascaded_union(boundary.geometry)
coords = points_to_coords(gdf_proj.geometry)
print(boundary_shape)
print(coords)

# Calculate Voronoi Regions
poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape)#,shapes_from_diff_with_min_area=5,
                                #accept_n_coord_duplicates=None,
                                #return_unassigned_points=False,
                                #farpoints_max_extend_factor=20)

print(poly_shapes[0])


'''
fig, ax = subplot_for_map()
plot_voronoi_polys_with_points_in_area(ax, boundary_shape, poly_shapes, pts, poly_to_pt_assignments)
ax.set_title('Voronoi regions')
plt.tight_layout()

plt.savefig('Voronoi_polygons.png')
'''


# save the polygon objects in a shapefile
from shapely.geometry import mapping, Polygon
import fiona
print(gdf.head())
time_of_failure = gdf['Z']
poly = poly_shapes
print(time_of_failure)
print(len(poly))

# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'time_of_failure': 'int'},
}

# Write a new Shapefile
with fiona.open('ToF_voronoi_polygons.shp', 'w', 'ESRI Shapefile', schema) as c:
    for i in range(len(poly_shapes)):
        c.write({
            'geometry': mapping(poly[i]),
            'properties': {'time_of_failure': time_of_failure[i]},
        })
