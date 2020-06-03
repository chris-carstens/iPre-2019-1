"""
test.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 02-06-20
Pontifical Catholic University of Chile

"""

import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point
from pyproj import Proj, transform

if __name__ == '__main__':
    # source  WGS84 EPSG: 4326 (WGS84) (*) 900913
    # destination   EPSG: 3857

    ans = gpd.GeoDataFrame(
        geometry=[Point((-96.798645, 32.742206))],
        crs=4326,
    )
    ans.to_crs(epsg=3857, inplace=True)

    # P3857 = Proj(init='epsg:3857')
    # P4326 = Proj(init='epsg:4326')
    #
    # x, y = transform(P4326, P3857, 32.742206, -96.798645)
