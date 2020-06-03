"""
test.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 02-06-20
Pontifical Catholic University of Chile

"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

if __name__ == '__main__':
    # source        EPSG: 900913 (WGS84) (*)
    # destination   EPSG: 3857

    ans = gpd.GeoDataFrame(
        geometry=[Point((32.643838, -96.999343))],
        crs='epsg:900913'
    )
    ans.to_crs(epsg=3857, inplace=True)
    # s_shp = gpd.read_file('./../data/streets.shp')
