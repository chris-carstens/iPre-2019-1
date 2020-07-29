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


# %% STKDE
# stkde = STKDE(bw=bw_stkde, shps=shps)
# stkde.fit(*PreProcessing(stkde).preparing_data())
#stkde.predict()


# %% Random Forest Regressor
# rfr = RForestRegressor(i_df=df, shps=shps,
#                        xc_size=100, yc_size=100, layers_n=7,
#                        read_data=True, read_df=True)

# %%

pm = ProMap(bw=bw, read_density=False)



pm.predict(*PreProcessing(pm, df=df).preparing_data())

#################

pm2 = ProMap(bw=bw, shps=shps, read_density=False)

pm.predict(*PreProcessing(pm, df=df).preparing_data())



# Plotter
pltr = Plotter(models=[

    pm

])

# pltr.hr()
pltr.pai()
#pltr.heatmap()

if __name__ == '__main__':
    pass
