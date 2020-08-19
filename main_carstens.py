# %%
from predictivehp.models import STKDE, create_model
import predictivehp.utils as ut
from predictivehp.visualization import Plotter

# %% Data
b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'

shps = ut.shps_processing(s_shp_p, c_shp_p, cl_shp_p)
data = ut.get_data()

m = create_model(data, shps, use_stkde=True)

# %% STKDE

m.fit(*m.prepare_stkde())
#stkde.spatial_pattern()

#stkde.heatmap(incidences=True, c=0.1)

# %% Plotter
pltr = Plotter(m)
#pltr.hr()
#pltr.pai()
#stkde.heatmap()
pltr.heatmap()

if __name__ == '__main__':
    pass
