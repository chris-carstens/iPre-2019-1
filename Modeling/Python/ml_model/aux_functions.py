# aux_functions.py:
# Python Version: 3.8.1

# iPre - Big Data para Criminología
# Created by Mauro S. Mendoza Elguera at 16-09-19
# Pontifical Catholic University of Chile

# Notes

# -

import numpy as np
import pandas as pd
import geopandas as gpd

from math import floor

from scipy.signal import convolve2d


def n_i(xi, x_min, hx):
    """
    Retorna la coordenada Nx_i, el cálculo es análogo para Ny_i.

    :param xi:
    :param x_min:
    :param hx:
    :return:
    """

    return floor((xi - x_min) / hx)


def id_i(nx_i, ny_i, n_x):
    """
    Retorna el id_i, asociado a la i-ésima celda en la malla.

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


def nc_incidents(D):
    """
    [DEPRECATED]

    Calcula la cantidad de incidentes en las celdas vecinas de cada una de
    las celdas en la matriz D.

    OBS. Notemos que con aplicar
    np.flipud(nc_incidents(np.flipud(m.T))).flatten()
    se obtiene rápidamente un vector columna para ingresar al dataframe
    correspondiente, con la indexación para df.

    :param D: np.array que bajo la perspectiva matricial, contiene la
    cantidad de incidentes que ocurren en determinada celda
    :return: np.ndarray que contiene en cada posición la cantidad de incidentes
    en celdas vecinas de la celda correspondiente en la matriz D.
    """

    Dl = np.pad(array=D[:-1, :],
                pad_width=((1, 0), (0, 0)),
                mode='constant',
                constant_values=0)

    Dr = np.pad(array=D[1:, :],
                pad_width=((0, 1), (0, 0)),
                mode='constant',
                constant_values=0)

    Dd = np.pad(array=D[:, :-1],
                pad_width=((0, 0), (1, 0)),
                mode='constant',
                constant_values=0)

    Du = np.pad(array=D[:, 1:],
                pad_width=((0, 0), (0, 1)),
                mode='constant',
                constant_values=0)

    Dld = np.pad(array=D[:-1, :-1],
                 pad_width=((1, 0), (1, 0)),
                 mode='constant',
                 constant_values=0)

    Dlu = np.pad(array=D[:-1, 1:],
                 pad_width=((1, 0), (0, 1)),
                 mode='constant',
                 constant_values=0)

    Drd = np.pad(array=D[1:, :-1],
                 pad_width=((0, 1), (1, 0)),
                 mode='constant',
                 constant_values=0)

    Dru = np.pad(array=D[1:, 1:],
                 pad_width=((0, 1), (0, 1)),
                 mode='constant',
                 constant_values=0)

    # np.sum(data[:, 1:], axis=1) # Column of the df

    return Dl + Dlu + Du + Dru + Dr + Drd + Dd + Dld


def to_df_col(D):
    """
    Transforma el array para su inclusión directa como una columna de un
    Pandas Dataframe

    :param D:
    :return:
    """

    return np.flipud(np.flipud(D.T)).flatten().tolist()


def filter_cells(df):
    """
    Completa la columna "in_dallas" del dataframe, explicitando cuales de las
    celdas se encuentran dentro de Dallas.

    :param df: Pandas dataframe
    :return: Pandas Dataframe
    """

    print('\tFiltering cells...')

    print('\t\tLoading shapefile...')
    dallas_shp = gpd.read_file("../../Data/Councils/Councils.shp")
    dallas_shp.to_crs(epsg=3857, inplace=True)

    print('\t\tCreating GeoDataframe...')
    # Trans. a gpd para usar el sjoin()

    # print(df[[('geometry', ''), ('in_dallas', '')]].head())
    geo_pd = gpd.GeoDataFrame(df[[('geometry', ''), ('in_dallas', '')]])

    geo_pd.crs = dallas_shp.crs  # Mismo crs que el shp para evitar warnings

    # Borramos el segundo nivel ''.
    geo_pd = geo_pd.T.reset_index(level=1, drop=True).T

    print('\t\tFiltering...')
    geo_pd = gpd.tools.sjoin(geo_pd, dallas_shp,
                             how='left',  # left para conservar indices
                             op='intersects')[['in_dallas',
                                               'index_right']]

    print('\t\tUpdating dataframe... ', end='')
    geo_pd.fillna(value={'index_right': 0}, inplace=True)  # para filtrar
    geo_pd.loc[geo_pd['index_right'] != 0, 'in_dallas'] = 1

    # Añadimos la columna del gpd al df inicial
    df[[('in_dallas', '')]] = geo_pd[['in_dallas']]

    # Filtramos el df inicial con la columna añadida
    df = df[df[('in_dallas', '')] == 1]

    print(f'finished!', end=" ")

    return df


if __name__ == '__main__':
    dim = 6
    layer = 2

    D = np.random.randint(low=0, high=10, size=(dim, dim), dtype=int)

    print(D, il_neighbors(matrix=D, i=layer), sep='\n' * 2)
