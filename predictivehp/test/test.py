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
    # destination m EPSG: 3857

    # ans = gpd.GeoDataFrame(
    #     geometry=[Point((-96.798645, 32.742206))],
    #     crs=4326,
    # )
    # ans.to_crs(epsg=3857, inplace=True)

    dll = gpd.read_file('./../data/streets.shp')
    dll.crs = 2276  # Source en ft
    # dll.to_crs(epsg=3857, inplace=True)  # Destination en m
