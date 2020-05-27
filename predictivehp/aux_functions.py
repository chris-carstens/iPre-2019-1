"""
aux_functions.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 01-05-20
Pontifical Catholic University of Chile

"""

from time import time

import numpy as np
import pandas as pd

from math import floor, sqrt, ceil
from scipy.signal import convolve2d

import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import seaborn as sns


# General

def timer(fn):
    def inner_1(*args, **kwargs):
        st = time()

        fn(*args, **kwargs)

        print(f"\nFinished! ({time() - st:3.1f} sec)")

    return inner_1


# Plots

def lineplot(x, y, x_label=None, y_label=None, title=None):
    rc = {
        'figure.facecolor': 'black',
        # 'figure.figsize': (5.51, 3.54),

        'xtick.color': 'white',
        'ytick.color': 'white',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.bottom': True,
        'xtick.top': False,
        'ytick.left': True,
        'ytick.right': False,

        'axes.facecolor': sns.dark_palette("black", 100)[60],
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'white',
        'text.color': 'white',

        'grid.color': sns.dark_palette("red", 100)[0],

    }

    # Para testear colores de la paleta (mantener comentado)
    # sns.palplot(sns.dark_palette("red", 100))

    sns.set(
        palette=sns.color_palette("Reds", 5)[::-1],
        rc=rc
    )

    fig = sns.lineplot(x=x, y=y)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)


# STKDE

def checked_points(points):

    dallas_shp = gpd.read_file('predictivehp/data/councils.shp')

    df_points = pd.DataFrame(
        {'x': points[0, :], 'y': points[1, :], 't': points[2, :]}
    )

    inc_points = df_points[['x', 'y']].apply(lambda row:
                                             Point(row['x'], row['y']),
                                             axis=1)
    geo_inc = gpd.GeoDataFrame({'geometry': inc_points, 'day': df_points['t']})

    # Para borrar el warning asociado a != epsg distintos
    geo_inc.crs = {'init': 'epsg:2276'}

    valid_inc = gpd.tools.sjoin(geo_inc,
                                dallas_shp,
                                how='inner',
                                op='intersects').reset_index()

    valid_inc_2 = valid_inc[['geometry', 'day']]

    x = valid_inc_2['geometry'].apply(lambda row: row.x)
    y = valid_inc_2['geometry'].apply(lambda row: row.y)
    t = valid_inc_2['day']

    v_points = np.array([x, y, t])

    return v_points


# ML

def n_i(xi, x_min, hx):
    """
    Retorna la coordenada Nx_i, el cálculo es análogo para Ny_i.

    :param xi:
    :param x_min:
    :param hx:
    :return:
    """

    return floor((xi - x_min) / hx)


def cell_index(nx_i, ny_i, n_y):
    """
    Retorna el cell_index, asociado a la i-ésima celda en la malla.

    :param nx_i:
    :param ny_i:
    :param n_x:
    :return:
    """

    return nx_i * n_y + ny_i


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
    dallas_shp = gpd.read_file("Data/Councils/councils.shp")
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


# ProMap

def n_semanas(total_dias, dia):
    total_semanas = total_dias // 7 + 1
    semanas_transcurridas = dia // 7 + 1
    delta = total_semanas - semanas_transcurridas
    if delta == 0:
        delta = 1
    return delta


def cells_distance(x1, y1, x2, y2, hx, hy):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    d = 1 + floor(dx / hx) + floor(dy / hy)
    # print('CENTROIDE DISTANCE: ', d)
    return d


def linear_distance(a1, a2):
    linear_distance = abs(a1 - a2)
    # print('DISTANCIA LINEAL: ', linear_distance)
    return float(linear_distance)


def find_position(mgridx, mgridy, x, y, hx, hy):
    x_desplazada = mgridx - hx / 2
    y_desplazada = mgridy - hy / 2
    pos_x = np.where(x_desplazada <= x)[0][-1]
    pos_y = np.where(y_desplazada <= y)[1][-1]
    return pos_x, pos_y


def n_celdas_pintar(xi, yi, x, y, hx, hy):
    x_sum = floor(abs(xi - x) / hx)
    y_sum = floor(abs(yi - y) / hy)
    return 1 + x_sum + y_sum


def radio_pintar(ancho_celda, bw):
    return ceil(bw / ancho_celda)



def square_matrix(lado):
    return np.ones((lado, lado), dtype=bool)


def limites_x(ancho_pintura, punto, malla):
    izq = punto - ancho_pintura
    if izq < 0:
        izq = 0

    der = punto + ancho_pintura
    range_malla = malla.shape[0]
    if der > range_malla:
        der = range_malla
    return izq, der


def limites_y(ancho_pintura, punto, malla):
    abajo = punto - ancho_pintura
    if abajo < 0:
        abajo = 0

    up = punto + ancho_pintura
    range_malla = malla.shape[1]
    if up > range_malla:
        up = range_malla

    return abajo, up


def grafico(x, y, name_x, name_y):
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title(name_x + ' VS ' + name_y)
    plt.plot(x, y)
    plt.show()


def calcular_celdas(hx, hy, superficie):
    """
    :param hx: en metros
    :param hy: en metros
    :param superficie: en kilemtros
    :return: numero de celdas asociadas
    """

    superficie = round(superficie)
    hx = hx / 1000
    hy = hy / 1000
    raiz = sqrt(superficie)

    return round((raiz / hx) * (raiz / hy))


if __name__ == '__main__':
    print(calcular_celdas(100, 100, 1_000))
    pass
