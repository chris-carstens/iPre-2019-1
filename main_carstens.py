# %%
from predictivehp.models._models import STKDE
from predictivehp.models.parameters import *
import predictivehp.processing.data_processing as dp
from predictivehp.visualization._plotter import Plotter

# %% Data
b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'

pp = dp.PreProcessing()
shps = pp.shps_processing(s_shp_p, c_shp_p, cl_shp_p)

# %% STKDE
stkde = STKDE(shps=shps, bw=bw_stkde)
pp.models = [stkde]
pp.define_models()

stkde.fit(*pp.prepare_stkde())
#stkde.spatial_pattern()

#stkde.heatmap(incidences=True, c=0.1)

# %% Plotter
pltr = Plotter(models=[
     stkde,
])
#pltr.hr()
#pltr.pai()
#stkde.heatmap()
pltr.heatmap()

if __name__ == '__main__':
    pass
