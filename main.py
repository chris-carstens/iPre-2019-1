# %%
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.visualization.plotter import Plotter

# %% Data

s_shp_path = 'predictivehp/data/streets.shp'
c_shp_path = 'predictivehp/data/councils.shp'
cl_shp_path = 'predictivehp/data/citylimit.shp'

df, streets, councils, c_limits = get_data(year=2017, n=150000,
                                           s_shp=s_shp_path,
                                           c_shp=c_shp_path,
                                           cl_shp=cl_shp_path)
print("\nReading Shapefiles...")
shps = {'streets': streets, 'councils': councils, 'c_limits': c_limits}

# TODO
#   extraer x_min, y_min, x_max, y_max de la db cuando el user no
#   entrega shapefiles

# %% STKDE
stkde = STKDE(df=df)

# %% Random Forest Regressor
#rfr = RForestRegressor(i_df=df, shps=shps,
#                       xc_size=100, yc_size=100, layers_n=7,
 #                      read_data=False, read_df=False)

# %%

#pm = ProMap(i_df=df, bw=bw, read_files=False)

# %% Plotter

pltr = Plotter(models=[stkde])
pltr.pai()
stkde.predict(c=np.linspace(0, 1, 10))


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
