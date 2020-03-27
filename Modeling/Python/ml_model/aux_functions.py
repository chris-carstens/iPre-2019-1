"""
aux_functions.py:
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 16-09-19
Pontifical Catholic University of Chile

Notes

-
"""

from time import time

import numpy as np
import geopandas as gpd

from math import floor
from scipy.signal import convolve2d


def timer(fn):
    def inner_1(*args, **kwargs):
        st = time()

        fn(*args, **kwargs)

        print(f"\n\tFinished! ({time() - st:3.1f} sec)")

    return inner_1


def n_i(xi, x_min, hx):
    """
    Retorna la coordenada Nx_i, el cálculo es análogo para Ny_i.

    :param xi:
    :param x_min:
    :param hx:
    :return:
    """

    return floor((xi - x_min) / hx)


def cell_index(nx_i, ny_i, n_x):
    """
    Retorna el cell_index, asociado a la i-ésima celda en la malla.

    :param nx_i:
    :param ny_i:
    :param n_x:
    :return:
    """

    return nx_i + n_x * ny_i


def diamond(d=3):
    """
    Entrega la matriz diamante con 1s en el límite y 0s en su interior.

    :param d: Dimensión dxd del diamante, número impar mayor a 3
    :return: ndarray con la matriz diamante
    """

    t = np.zeros(shape=(d, d), dtype=int)

    l = np.eye(*t.shape, k=t.shape[0] // 2, dtype=int) + \
        np.eye(*t.shape, k=(t.shape[0] // 2) * -1, dtype=int)
    l = l + np.fliplr(l)

    # Corrección de los 2s

    l[0, t.shape[0] // 2] = 1
    l[t.shape[0] // 2, 0] = 1
    l[t.shape[0] // 2, t.shape[0] - 1] = 1
    l[t.shape[0] - 1, t.shape[0] // 2] = 1

    return l


def il_neighbors(matrix, i=1):
    """
    Calcula la cantidad de incidentes en la i-ésima capa (tipo ProMap)
    para cada una de las celdas en el arreglo matrix.

    :type matrix: np.ndarray
    :param matrix: ndarray con la cantidad de incidentes ocurridos en
        cada celda de la malla
    :param i: int que indica la i-ésima considerada
    :return: ndarray con la suma de incidentes de la capa i-ésima para
        cada celda
    """

    kernel = diamond(d=2 * i + 1)

    return convolve2d(in1=matrix, in2=kernel, mode='same')


def to_df_col(D):
    """
    Transforma el array para su inclusión directa como una columna de un
    Pandas Dataframe

    :param np.ndarray D:
    :return:
    """

    # return np.flipud(np.flipud(D.T)).flatten()
    return D.flatten()


def filter_cells(df):
    """
    Completa la columna "in_dallas" del dataframe, explicitando cuales de las
    celdas se encuentran dentro de Dallas.

    :param pandas.DataFrame df: Dataframe que contiene información de celdas
        que no necesariamente están en Dallas
    :return: Dataframe con celdas filtradas, i.e., que están
        dentro de Dallas 
    :rtype: pandas.DataFrame
    """

    aux_df = df

    print('\tFiltering cells...')

    print('\t\tLoading shapefile...')
    dallas_shp = gpd.read_file("../../Data/Councils/Councils.shp")
    dallas_shp.to_crs(epsg=3857, inplace=True)

    print('\t\tCreating GeoDataframe...')
    geo_pd = gpd.GeoDataFrame(aux_df[[('geometry', ''), ('in_dallas', '')]])
    geo_pd.crs = dallas_shp.crs  # Mismo crs que el shp para evitar warnings

    # Borramos el segundo nivel ''
    geo_pd = geo_pd.T.reset_index(level=1, drop=True).T

    print('\t\tFiltering...')
    geo_pd = gpd.tools.sjoin(geo_pd, dallas_shp,
                             how='left',  # left para conservar indices
                             op='intersects')[['in_dallas', 'index_right']]

    print('\t\tUpdating dataframe... ', end='')
    geo_pd.fillna(value={'index_right': 14}, inplace=True)  # para filtrar
    geo_pd.loc[geo_pd['index_right'] < 14, 'in_dallas'] = 1

    # Añadimos la columna del gpd filtrado al df inicial
    aux_df[[('in_dallas', '')]] = geo_pd[['in_dallas']]

    # Filtramos el df inicial con la columna añadida
    aux_df = aux_df[aux_df[('in_dallas', '')] == 1]

    print(f'finished!', end=" ")

    return aux_df


if __name__ == '__main__':
    # Test para diamond() y il_neighbors()
    # dim = 6
    # layer = 2
    #
    # D = np.random.randint(low=0, high=10, size=(dim, dim), dtype=int)
    #
    # print(D, il_neighbors(matrix=D, i=layer), sep='\n' * 2)

    # TODO
    #       - Revisar problema de la relación entre las funciones diamond(),
    #           il_neighbors() y to_df_col(). Recordemos que los patrones de
    #           hotspots en mpl_plot_4.py deben cer 'circulares' y no líneas
    #           verticales, como ocurre actualmente.

    import pandas as pd

    x, y = np.mgrid[0:3, 0:5]
    D = np.random.randint(10, size=(3, 5), dtype=int)
    df = pd.DataFrame(to_df_col(D))
    df['extra'] = 'test'
