# %%
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import PreProcessing
from predictivehp.visualization.plotter import Plotter

# %% Data

b_path = 'predictivehp/data/'
s_shp_p = b_path + 'streets.shp'
c_shp_p = 'predictivehp/data/councils.shp'
cl_shp_p = 'predictivehp/data/citylimit.shp'

shps = {}


# TODO
#   extraer x_min, y_min, x_max, y_max de la db cuando el user no
#   entrega shapefiles

# %% STKDE
stkde = STKDE(bw=bw_stkde)

# %% Random Forest Regressor
#rfr = RForestRegressor(i_df=df, shps=shps,
 #                      xc_size=100, yc_size=100, layers_n=7,
  ##                     read_data=False, read_X=False)
#rfr.heatmap()

# rfr.to_pickle('data.pkl')
# rfr.to_pickle('df.pkl')

# %#%
#pm = ProMap(bw=bw, shps=shps)
#pm.heatmap(c=0)

stkde.fit(*PreProcessing([stkde]).prepare_stkde())
#stkde.heatmap()

# %% Plotter
#pm.predict(*PreProcessing(model=pm, df=df).prepare_promap())

pltr = Plotter(models=[
     stkde,
 #    rfr,
  #   pm
  ])

pltr.hr()
pltr.pai()
#stkde.heatmap()

#pltr.heatmap()

if __name__ == '__main__':
    pass
