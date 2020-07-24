# %%
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data, PreProcessing
from predictivehp.visualization.plotter import Plotter

# %% Data

b_path = 'predictivehp/data/'
s_shp_p = b_path + 'streets.shp'
c_shp_p = b_path + 'councils.shp'
cl_shp_p = b_path + 'citylimit.shp'

shps = {}
df, shps['streets'], shps['councils'], shps['c_limits'] = \
    get_data(year=2017, n=150000,
             s_shp=s_shp_p, c_shp=c_shp_p, cl_shp=cl_shp_p)


# %% STKDE
#stkde = STKDE(df=df, bw=bw)

# %% Random Forest Regressor
# rfr = RForestRegressor(i_df=df, shps=shps,
#                        xc_size=100, yc_size=100, layers_n=7,
#                        read_data=True, read_df=True)

# %%
pm = ProMap(i_df=df, bw=bw, shps=shps)
pm.predict(*PreProcessing(pm).preparing_data())



# Plotter
pltr = Plotter(models=[
    #stkde,
    pm
    #rfr,
])

pltr.hr()
pltr.pai()
#pltr.heatmap()

if __name__ == '__main__':
    pass
