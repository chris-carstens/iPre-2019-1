# %%
from predictivehp.models._models import create_model
import predictivehp.utils as ut
from predictivehp.visualization import Plotter

# %% Data
b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'


shps = ut.shps_processing(s_shp_p, c_shp_p, cl_shp_p)
data = ut.get_data(2017, 150_000)


# %% PROMAP

modelos = create_model(data,shps, use_promap=True)
modelos.set_parameters('ProMap', read_density=True)
modelos.fit()
modelos.predict()

pm = modelos.models[0]


#
pltr = Plotter(modelos)
pltr.hr()
pltr.pai()


pltr.heatmap(c=[0.1, 0.4], incidences=True, show_score=True,
             savefig=False, fname='hm_example.png')



# pltr.heatmap(c = 0.2, incidences=True)
# pltr.heatmap(c =[0.1, 0.2], incidences=True)


if __name__ == '__main__':
    pass