# %%
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data, PreProcessing, shps_proccesing
from predictivehp.visualization.plotter import Plotter

# %% Data

b_path = 'predictivehp/data/'
s_shp_p = b_path + 'streets.shp'
c_shp_p = b_path + 'councils.shp'
cl_shp_p = b_path + 'citylimit.shp'


shps = shps_proccesing(s_shp_p, c_shp_p, cl_shp_p)


pm = ProMap(bw=bw, shps=shps, read_density=True)
pp = PreProcessing(pm)
pm.predict(*pp.preparing_data())

#################


# Plotter
pltr = Plotter(models=[
    pm
])

pltr.hr()
pltr.pai()
pltr.heatmap()

if __name__ == '__main__':
    pass
