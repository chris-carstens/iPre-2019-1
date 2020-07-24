# %%
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data, PreProcessing
from predictivehp.visualization.plotter import Plotter

# %% Data

b_path = 'predictivehp/data/'
s_shp_p = b_path + 'streets.shp'
c_shp_p = 'predictivehp/data/councils.shp'
cl_shp_p = 'predictivehp/data/citylimit.shp'

shps = {}
df, shps['streets'], shps['councils'], shps['c_limits'] = \
    get_data(year=2017, n=150000,
             s_shp=s_shp_p, c_shp=c_shp_p, cl_shp=cl_shp_p)

# TODO
#   extraer x_min, y_min, x_max, y_max de la db cuando el user no
#   entrega shapefiles

# %% STKDE
stkde = STKDE(bw=bw_stkde, sample_number=3600)

#stkde.heatmap()

# %% Random Forest Regressor
#rfr = RForestRegressor(i_df=df, shps=shps,
 #                      xc_size=100, yc_size=100, layers_n=7,
  ##                     read_data=False, read_df=False)
#rfr.heatmap()

# rfr.to_pickle('data.pkl')
# rfr.to_pickle('df.pkl')

# %#%
#pm = ProMap(i_df=df, bw=bw, read_files=False)
#pm.heatmap(c=0)


pp = PreProcessing(model=stkde, df=df)
stkde.fit(*pp.prepare_stkde())

# %% Plotter
pltr = Plotter(models=[
     stkde,
 ##    rfr,
 #    pm
  ])

pltr.hr()
pltr.pai()

#pltr.heatmap()

if __name__ == '__main__':
    pass
