# %%


from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.visualization.plotter import Plotter
import predictivehp.processing.data_processing as dp

# %% Data

b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'

shps = dp.shps_proccesing(s_shp_p, c_shp_p, cl_shp_p)
df = dp.get_data(year=2017, n=150000)

# %% Random Forest Regressor
rfr = RForestRegressor(i_df=df, shps=shps,
                       xc_size=100, yc_size=100, n_layers=7,
                       read_data=True, read_df=True)
pp = dp.PreProcessing(rfr, df)
X_train, y_train = pp.prepare_rfr('train', 'weighted')
X_test, y_test = pp.prepare_rfr('test', 'weighted')

rfr.fit(X_train, y_train)
rfr.predict(X_test)

# rfr.to_pickle("data.pkl")
# rfr.to_pickle('X.pkl')

# %%
# Plotter
pltr = Plotter(models=[
    rfr,
])

pltr.hr()
pltr.pai()
pltr.heatmap()

# TODO
#    - ¿Que sucede con la normalización de la label 'y'?

if __name__ == '__main__':
    pass
