"""
main.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 11-05-20
Pontifical Catholic University of Chile

"""

# %%
from predictivehp.models.parameters import *
from predictivehp.processing.data_processing import get_data
from predictivehp.models.models import STKDE, RForestRegressor, ProMap

# %% Initial Database
df = get_data(year=2017, n=150000)

# %% STKDE

#stkde = STKDE(n=1000, year='2017')

# %%
#rfr = RForestRegressor(i_df=df, xc_size=None, yc_size=None,
#                       read_df=False, read_data=False)
#rfr.plot_statistics(n=500)

# %%
#pm = ProMap(n=150_000, year="2017", bw=bw, i_df=df, read_files=False)

# %%


if __name__ == '__main__':
    # TODO Reu.20/05
    #   - Acondicionar modelos para recibir la initial database
    #   - self.set_params()
    #   - Implementar ambas opciones: ancho/largo celda y nro. celdas en x e y.
    #   - STKDE:
    #       None
    #   - rfr:
    #       Automatizar con parámetros las capas (evitar el hardcodeo)
    #       Implementar ambas opciones: ancho/largo celda y nro. celdas
    #       en x e y.
    #       Nro de capas: calcular en base al tamaño/nro de celdas
    #           y el radio de influencia (obs. redondear floats,
    #           uso de excepts) e.g. floor(radio / c_size)
    #       Elección de parámetro que defina la ventana de predicción
    #           'from' y el 'to' (pensar en la implementación de p_groups)
    #   - ProMap:
    #       Implementar ambas opciones: ancho/largo celda y nro. celdas
    #           en x e y.

    pass
