# %%
from predictivehp.models._models import ProMap
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


# %% PROMAP
pm = ProMap(shps=shps)
pm.set_parameters(bw = bw)
pp.model = [pm]
pp.define_models()


pm.fit()
pm.predict(*pp.prepare_promap())
pm.heatmap(incidents=True)


#
pltr = Plotter(models=[
     pm,
])
pltr.hr()
pltr.pai()



if __name__ == '__main__':
    pass
