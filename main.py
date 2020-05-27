# %%
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data
from predictivehp.models.models import STKDE, RForestRegressor, ProMap
from predictivehp.visualization.plotter import Plotter

# %% Initial Database
df = get_data(year=2017, n=150000)

# %% STKDE
stkde = STKDE(n=1000, year='2017')

# %% Random Forest Regressor
rfr = RForestRegressor(i_df=df, xc_size=100, yc_size=100, n_capas=7,
                       read_data=False, read_df=False)

# %%
# pm = ProMap(n=150_000, year="2017", bw=bw, i_df=df, read_files=False)
pm = ProMap(bw=bw, i_df=df, read_files=False)

# %% Plotter

pltr = Plotter(models=[stkde, rfr, pm])

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
    #   -

    pass
