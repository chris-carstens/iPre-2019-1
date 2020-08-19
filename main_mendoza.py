# %%
from datetime import date
from predictivehp.models import RForestRegressor as RFR, create_model
import predictivehp.utils as ut

b_path = 'predictivehp/data'
s_shp_p = f'{b_path}/streets.shp'
c_shp_p = f'{b_path}/councils.shp'
cl_shp_p = f'{b_path}/citylimit.shp'

shps = ut.shps_processing(s_shp_p, c_shp_p, cl_shp_p)
data = ut.get_data(year=2017, n=150000)

m = create_model(
    data=data, shps=shps,
    start_prediction=date(2017, 11, 1), length_prediction=7,
    use_stkde=True, use_promap=False, use_rfr=False,
)
m.fit(data)
m.predict()
