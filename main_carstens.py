import predictivehp.utils as ut
from predictivehp.models import create_model
from predictivehp.visualization import Plotter

b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'

shps = ut.shps_processing(s_shp_p, c_shp_p, cl_shp_p)
data = ut.get_data()

m = create_model(data, shps, use_stkde=True)

# %% STKDE
m.set_parameters(m_name='STKDE', bw=[780, 1090, 25])

pp = m.prepare_data()
m.fit(pp)
m.predict()

# stkde.heatmap(incidences=True, c=0.1)

# %% Plotter
pltr = Plotter(m)
#pltr.hr()
#pltr.pai()
# stkde.heatmap()
pltr.heatmap(ap=[0.1, 0.5], incidences=True)
#m.validate(ap=0.2)
