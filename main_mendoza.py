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

# %% Random Forest Regressor
rfr = RForestRegressor(i_df=df, shps=shps,
                       xc_size=100, yc_size=100, n_layers=7,
                       read_data=False, read_df=False)
pp = PreProcessing(rfr, df)
X_train, y_train = pp.prepare_rfr('train')
X_test, y_test = pp.prepare_rfr('test')

rfr.fit(X_train, y_train)
rfr.predict(X_test)

rfr.to_pickle("data.pkl")
rfr.to_pickle('X.pkl')

# %%
# Plotter
pltr = Plotter(models=[
    rfr,
])

pltr.hr()
pltr.pai()
pltr.heatmap()

if __name__ == '__main__':
    pass
