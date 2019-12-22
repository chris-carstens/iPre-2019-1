"""
aux_functions.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 22-12-19
Pontifical Catholic University of Chile

"""

from time import time

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point


def timer(fn):
    def inner_1(*args, **kwargs):
        st = time()

        fn(*args, **kwargs)

        print(f"\nFinished! ({time() - st:3.1f} sec)")

    return inner_1


def checked_points(points):
    dallas_shp = gpd.read_file('../../Data/Councils/Councils.shp')

    df_points = pd.DataFrame(
        {'x': points[0, :], 'y': points[1, :], 't': points[2, :]}
    )

    inc_points = df_points[['x', 'y']].apply(lambda row:
                                             Point(row['x'], row['y']),
                                             axis=1)
    geo_inc = gpd.GeoDataFrame({'geometry': inc_points, 'day': df_points['t']})

    # Para borrar el warning asociado a != epsg distintos
    geo_inc.crs = {'init': 'epsg:2276'}

    valid_inc = gpd.tools.sjoin(geo_inc,
                                dallas_shp,
                                how='inner',
                                op='intersects').reset_index()

    valid_inc_2 = valid_inc[['geometry', 'day']]

    x = valid_inc_2['geometry'].apply(lambda row: row.x)
    y = valid_inc_2['geometry'].apply(lambda row: row.y)
    t = valid_inc_2['day']

    v_points = np.array([x, y, t])

    return v_points


if __name__ == '__main__':
    pass
