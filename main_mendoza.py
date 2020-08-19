from datetime import date
from predictivehp.models import create_model
from predictivehp.visualization import Plotter
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
    use_stkde=False, use_promap=False, use_rfr=True,
)
m.set_parameters(m_name='RForestRegressor', t_history=4,
                 xc_size=100, yc_size=100, n_layers=7,
                 label_weights=None,
                 read_data=False, read_X=False,
                 w_data=True, w_X=True)
data_p = m.prepare_data()
m.fit(data_p)
m.predict()

pltr = Plotter(m)
pltr.hr()
pltr.pai()
pltr.heatmap(c=None, incidences=True, show_score=True,
             savefig=False, fname='hm_example.png')
