"""
aux_functions.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 01-05-20
Pontifical Catholic University of Chile

"""

from math import floor, sqrt, ceil
from time import time

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from shapely.geometry import Point


# General

def timer(fn):
    def inner(*args, **kwargs):
        st = time()
        fn(*args, **kwargs)
        print(f"\nFinished! ({time() - st:3.1f} sec)")

    return inner


# Plots

def lineplot(x, y, x_label=None, y_label=None, title=None, label=None):
    """Lineplot modificado"""
    mpl.rcdefaults()
    rc = {
        'figure.facecolor': 'black',
        'figure.figsize': (6.75, 4),  # Values for JLab - (6.0, 4.0) default

        'xtick.color': 'white',
        'xtick.major.size': 3,
        'xtick.top': False,
        'xtick.bottom': True,

        'ytick.color': 'white',
        'ytick.major.size': 3,
        'ytick.left': True,
        'ytick.right': False,

        'axes.facecolor': '#100000',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'white',
        'axes.grid': True,
        'axes.axisbelow': True,

        'text.color': 'white',

        'label.shadow': True,
        'label.framealpha': 1.0,

        'grid.color': '#250000',
    }
    mpl.rcParams.update(rc)

    plt.plot(x, y, label=label)

    plt.xlabel(x_label) if x_label else None
    plt.ylabel(y_label) if y_label else None
    plt.title(title) if title else None


# STKDE

def checked_points(points):
    """

    Parameters
    ----------
    points

    Returns
    -------
    np.ndarray
    """
    # 'predictivehp/data/councils.shp'
    dallas_shp = gpd.read_file('predictivehp/data/councils.shp')
    dallas_shp.crs = 2276

    df_points = pd.DataFrame(
        {'x': points[0, :], 'y': points[1, :], 't': points[2, :]}
    )

    inc_points = df_points[['x', 'y']].apply(lambda row:
                                             Point(row['x'], row['y']),
                                             axis=1)
    geo_inc = gpd.GeoDataFrame({'geometry': inc_points, 'day': df_points['t']})

    # Para borrar el warning asociado a != epsg distintos
    geo_inc.crs = 2276

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
    """Retorna la coordenada Nx_i, el cálculo es análogo para Ny_i.

    Parameters
    ----------
    xi
    x_min
    hx

    Returns
    -------

    """
    return floor((xi - x_min) / hx)


def cell_index(nx_i, ny_i, n_x):
    """Retorna el cell_index, asociado a la i-ésima celda en la malla.

    Parameters
    ----------
    nx_i
    ny_i
    n_x

    Returns
    -------

    """
    return nx_i + n_x * ny_i


def diamond(d=3):
    """

    Parameters
    ----------
    d : int
      Dimensión dxd del diamante, número impar mayor a 3
    Returns
    -------
    np.ndarray
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
    """Calcula la cantidad de incidentes en la i-ésima capa (tipo ProMap)
    para cada una de las celdas en el arreglo matrix.

    Parameters
    ----------
    matrix : np.ndarray
      Matriz con la cantidad de incidentes ocurridos en cada celda de la
      malla
    i : int
      i-ésima capa considerada
    Returns
    -------
    np.ndarray
    """
    kernel = diamond(d=2 * i + 1)

    return convolve2d(in1=matrix, in2=kernel, mode='same')


def to_df_col(D):
    """Transforma el array para su inclusión directa como una columna de un
    Pandas Dataframe.

    Parameters
    ----------
    D : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return D.flatten()


def filter_cells(df, shp):
    """Completa la columna "in_dallas" del dataframe, explicitando cuales
    de las celdas se encuentran dentro de Dallas.

    Parameters
    ----------
    df : pd.DataFrame
      Dataframe que contiene información de celdas que no necesariamente
      están en Dallas
    shp : gpd.GeoDataFrame

    Returns
    -------
    pd.DataFrame
    """
    aux_df = df

    print('\tFiltering cells...')
    print('\t\tLoading shapefile...')
    dallas_shp = shp

    print('\t\tCreating GeoDataframe...')
    geo_pd = gpd.GeoDataFrame(aux_df[[('geometry', ''), ('in_dallas', '')]],
                              crs=3857)

    # Borramos el segundo nivel ''
    geo_pd = geo_pd.T.reset_index(level=1, drop=True).T
    geo_pd.crs = dallas_shp.crs
    print('\t\tFiltering...')
    geo_pd = gpd.tools.sjoin(geo_pd, dallas_shp,
                             how='left',
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
    """

    Parameters
    ----------
    total_dias
    dia

    Returns
    -------

    """
    total_semanas = total_dias // 7 + 1
    semanas_transcurridas = dia // 7 + 1
    delta = total_semanas - semanas_transcurridas
    if delta == 0:
        delta = 1
    return delta


def cells_distance(x1, y1, x2, y2, hx, hy):
    """

    Parameters
    ----------
    x1
    y1
    x2
    y2
    hx
    hy

    Returns
    -------

    """
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    d = 1 + floor(dx / hx) + floor(dy / hy)
    return d


def linear_distance(a1, a2):
    """

    Parameters
    ----------
    a1
    a2

    Returns
    -------

    """
    ld = abs(a1 - a2)
    return float(ld)


def find_position(mgridx, mgridy, x, y, hx, hy):
    """

    Parameters
    ----------
    mgridx
    mgridy
    x
    y
    hx
    hy

    Returns
    -------

    """
    x_desplazada = mgridx - hx / 2
    y_desplazada = mgridy - hy / 2
    pos_x = np.where(x_desplazada <= x)[0][-1]
    pos_y = np.where(y_desplazada <= y)[1][-1]
    return pos_x, pos_y


def n_celdas_pintar(xi, yi, x, y, hx, hy):
    """

    Parameters
    ----------
    xi
    yi
    x
    y
    hx
    hy

    Returns
    -------

    """
    x_sum = floor(abs(xi - x) / hx)
    y_sum = floor(abs(yi - y) / hy)
    return 1 + x_sum + y_sum


def radio_pintar(ancho_celda, bw):
    """

    Parameters
    ----------
    ancho_celda
    bw

    Returns
    -------

    """
    return ceil(bw / ancho_celda)


def limites_x(ancho_pintura, punto, malla):
    """

    Parameters
    ----------
    ancho_pintura
    punto
    malla

    Returns
    -------

    """
    izq = punto - ancho_pintura
    if izq < 0:
        izq = 0

    der = punto + ancho_pintura
    range_malla = malla.shape[0]
    if der > range_malla:
        der = range_malla
    return izq, der


def limites_y(ancho_pintura, punto, malla):
    """

    Parameters
    ----------
    ancho_pintura
    punto
    malla

    Returns
    -------

    """
    abajo = punto - ancho_pintura
    if abajo < 0:
        abajo = 0

    up = punto + ancho_pintura
    range_malla = malla.shape[1]
    if up > range_malla:
        up = range_malla

    return abajo, up


def calcular_celdas(hx, hy, superficie):
    """

    Parameters
    ----------
    hx : {int, float}
      en metros
    hy : {int, float}
      en metros
    superficie : {int, float}
      en kilómetros

    Returns
    -------
    {int, float}
      Número de celdas asociadas
    """
    superficie = round(superficie)
    hx = hx / 1000
    hy = hy / 1000
    sqrt_ = sqrt(superficie)

    return round((sqrt_ / hx) * (sqrt_ / hy))


def print_mes(m_train, m_predict, dias):
    """
    m_train: int
    m_predict: int
    dias: int
    :return str
    """

    meses = {1: 'Enero',
             2: 'Febrero',
             3: 'Marzo',
             4: 'Abril',
             5: 'Mayo',
             6: 'Junio',
             7: 'Julio',
             8: 'Agosto',
             9: 'Septiembre',
             10: 'Octubre',
             11: 'Noviembre',
             12: 'Diciembre'}

    return f'Entrenando hasta: {meses[m_train]}\n' \
           f'Predicción de: {dias} días para {meses[m_predict]} '


def checked_points_pm(points):
    # 'predictivehp/data/councils.shp'
    dallas_shp = gpd.read_file('predictivehp/data/councils.shp')
    dallas_shp.crs = 2276
    dallas_shp.to_crs(epsg=3857, inplace=True)

    df_points = pd.DataFrame({'x': points[0, :], 'y': points[1, :]})

    inc_points = df_points[['x', 'y']].apply(lambda row:
                                             Point(row['x'], row['y']),
                                             axis=1)

    geo_inc = gpd.GeoDataFrame({'geometry': inc_points}, crs=3857)

    geo_inc.crs = dallas_shp.crs

    valid_inc = gpd.tools.sjoin(geo_inc,
                                dallas_shp,
                                how='inner',
                                op='intersects').reset_index()

    return len(valid_inc)


if __name__ == '__main__':
    pass
