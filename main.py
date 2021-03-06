# %%
from predictivehp.models._models import STKDE, RForestRegressor, ProMap
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data
from predictivehp.visualization._plotter import Plotter

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
stkde = STKDE(df=df, bw=bw)
#
# # %% Random Forest Regressor
#rfr = RForestRegressor(data_0=data, shps=shps,
                        #xc_size=100, yc_size=100, layers_n=7,
                        #read_data=False, read_X=False)
# rfr.heatmap()
# #
# rfr.to_pickle('data.pkl')
# rfr.to_pickle('data.pkl')

# %%
pm = ProMap(i_df=df, bw=bw, shps=shps)


#pm.heatmap()

# %% Plotter
pltr = Plotter(models=[
     #stkde,
     #rfr,
     #pm
  ])

pltr.hr()
pltr.pai()

if __name__ == '__main__':
    # TODO Reu. 20/05
    #   - Acondicionar modelos para recibir la initial database
    #   - self.set_params()
    #   - Implementar ambas opciones: ancho/largo celda y nro. celdas en x e y.
    #   - STKDE:
    #       None
    #   - rfr:
    #       √ Automatizar con parámetros las capas (evitar el hardcodeo)
    #       √ Implementar ambas opciones: ancho/largo celda y nro. celdas
    #       en x e y.
    #       Nro de capas: calcular en base al tamaño/nro de celdas
    #           y el radio de influencia (obs. redondear floats,
    #           uso de excepts) e.g. floor(radio / c_size)
    #       Elección de parámetro que defina la ventana de predicción
    #           'from' y el 'to' (pensar en la implementación de p_groups)
    #   - ProMap:
    #       Implementar ambas opciones: ancho/largo celda y nro. celdas
    #           en x e y.

    # TODO Reu.27/05
    #   - d_limits, automatizar desde:
    #       * √ shapefiles
    #       * ? Incidentes, cuando el user no entrega un shapefile
    #   - STKDE:
    #       Estudiar umbral (3600) para nro. máximo de registros a usar
    #       en el algoritmo

    pass
