# %%
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data
from predictivehp.visualization.plotter import Plotter


print('-' * 100)

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
# stkde = STKDE(df=df)
# stkde.heatmap(100,100)

# %% Random Forest Regressor
#rfr = RForestRegressor(i_df=df, shps=shps,
#                       xc_size=100, yc_size=100, layers_n=7,
#                       read_data=False, read_df=False)
#rfr.heatmap()

# rfr.to_pickle('data.pkl')
# rfr.to_pickle('df.pkl')

# %%

promaps = [ProMap(name= f'Ventana: {i} días',i_df=df, bw=bw, read_files=True,
                  ventana_dias=i) for i in range(1,8)]

pltr = Plotter(models=promaps)

pltr.hr()
pltr.pai()

pm = promaps[-1]
pm.plot_incident(pm.training_matrix, f'días: {8}')
pm.heatmap()
pm.plot_incident(pm.testing_matrix, f'días: {8}')




# %% Plotter


if __name__ == '__main__':
    pass
