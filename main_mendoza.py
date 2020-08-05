# %%


import predictivehp.processing.data_processing as dp
from predictivehp.models.models import RForestRegressor
from predictivehp.visualization.plotter import Plotter

# %% Data

b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'

pp = dp.PreProcessing()
shps = pp.shps_processing(s_shp_p, c_shp_p, cl_shp_p)
df = pp.get_data(year=2017, n=150000)

# %% Random Forest Regressor
rfr = RForestRegressor(i_df=df, shps=shps,
                       xc_size=100, yc_size=100, n_layers=7,
                       read_data=False, read_X=False)
pp = dp.PreProcessing([rfr])
X_train, y_train = pp.prepare_rfr('train', 'default')
X_test, y_test = pp.prepare_rfr('test', 'default')

rfr.fit(X_train, y_train)
rfr.predict(X_test)

# %%
# Plotter
pltr = Plotter(models=[
    rfr,
])

pltr.hr()

if __name__ == '__main__':
    pass
