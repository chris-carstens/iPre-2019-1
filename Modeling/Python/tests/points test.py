# points test.py:
# - Python version: 3.7.1
# - Author: Mauro S. Mendoza Elguera
# - Institution: Pontifical Catholic University of Chile
# - Date: 2019-08-28

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

pd.set_option('display.max_columns', None)

dallas_shp = gpd.read_file('../Data/Councils/Councils.shp')

points = np.ndarray(
        shape=(3, 3),
        buffer=np.array([[2.52462227e+06, 2.52442353e+06, 2.49574658e+06],
                         [6.99700923e+06, 6.99943305e+06, 6.93259018e+06],
                         [2.72768639e+02, 3.25375089e+02, 3.09736772e+02]])
)

df_points = pd.DataFrame(
        {'x': points[0, :], 'y': points[1, :], 't': points[2, :]}
)

inc_points = df_points[['x', 'y']].apply(lambda row:
                                         Point(row['x'], row['y']),
                                         axis=1)
geo_inc = gpd.GeoDataFrame(
        {'geometry': inc_points, 'day': df_points['t']}
)

geo_inc.crs = {'init': 'epsg:2276'}  # Para borrar el warning asociado a != epsg

valid_inc = gpd.tools.sjoin(geo_inc,
                            dallas_shp,
                            how='inner',
                            op='intersects')

valid_inc_2 = valid_inc[['geometry', 'day']]

x = valid_inc_2['geometry'].apply(lambda row: row.x)
y = valid_inc_2['geometry'].apply(lambda row: row.y)
t = valid_inc_2['day']

v_points = np.array([x, y, t])

# concatenation = np.concatenate((points, points.T), axis=1)
