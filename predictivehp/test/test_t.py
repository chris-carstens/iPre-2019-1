import geopandas as gpd


dallas_shp = gpd.read_file('../data/citylimit.shp')
dallas_shp.plot()
