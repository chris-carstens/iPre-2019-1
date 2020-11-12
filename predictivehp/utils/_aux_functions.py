from datetime import datetime
from math import floor, sqrt, ceil
from time import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from shapely.geometry import Point
from sodapy import Socrata

import predictivehp._credentials as cre


# General

def timer(fn):
    def inner(*args, **kwargs):
        st = time()
        fn(*args, **kwargs)
        print(f"\nFinished! ({time() - st:3.1f} sec)")

    return inner


def get_data(year=2017, n=150000):
    # print("\nRequesting data...")
    with Socrata(cre.socrata_domain,
                 cre.API_KEY_S,
                 username=cre.USERNAME_S,
                 password=cre.PASSWORD_S) as client:
        query = \
            f"""
                select
                    incidentnum,
                    year1,
                    date1,
                    month1,
                    x_coordinate,
                    y_cordinate,
                    offincident
                where
                    year1 = {year}
                    and date1 is not null
                    and x_coordinate is not null
                    and y_cordinate is not null
                    and offincident = 'BURGLARY OF HABITATION - FORCED ENTRY'
                order by date1
                limit
                    {n}
                """

        results = client.get(cre.socrata_dataset_identifier,
                             query=query,
                             content_type='json')
        df = pd.DataFrame.from_records(results)
        # print("\n"
        #       f"\tn = {n} incidents requested  Year = {year}"
        #       "\n"
        #       f"\t{data.shape[0]} incidents successfully retrieved!")

        # DB Cleaning & Formatting
        for col in ['x_coordinate', 'y_cordinate']:
            df.loc[:, col] = df[col].apply(
                lambda x: float(x))
        df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(
            lambda x: float(x))
        df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(
            lambda x: float(x))
        df.loc[:, 'date1'] = df['date1'].apply(  # OJO AL SEPARADOR ' '
            lambda x: datetime.strptime(
                x.split(' ')[0], '%Y-%m-%d')
        )
        df.loc[:, 'date1'] = df["date1"].apply(lambda x: x.date())

        df = df[['x_coordinate', 'y_cordinate', 'date1', 'month1']]
        df.loc[:, 'y_day'] = df["date1"].apply(
            lambda x: x.timetuple().tm_yday
        )
        df.rename(columns={'x_coordinate': 'x',
                           'y_cordinate': 'y',
                           'date1': 'date'},
                  inplace=True)

        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


def get_Socrata_data(domain=cre.socrata_domain, app_token=cre.API_KEY_S,
                     username=cre.USERNAME_S, password=cre.PASSWORD_S,
                     year=2017,
                     offincident="'BURGLARY OF HABITATION - FORCED ENTRY'",
                     n=150000,
                     ds_identifier=cre.socrata_dataset_identifier,
                     content_type='json',
                     save=False,
                     path='../predictivehp/data/SOCRATA_DATA_Dallas.xlsx'):
    """

    Parameters
    ----------
    domain : str
      domain you wish you to access
    app_token : str
      Socrata application token
    username : str
      Socrata username
    password : str
      Socrata password
    year : int
      Año a filtrar de la database
    offincident : str
      Tipo de incidentes
    n : int
      Nº máximo de registros a extraer
    ds_identifier : str
      Socrata Dataset identifier
    content_type : str
    save : bool
    path : str
      Path donde guarda el archivo excel generado

    Returns
    -------
    pd.DataFrame
    """
    with Socrata(domain, app_token,
                 username=username, password=password) as client:
        query = \
            f"""
                select
                    incidentnum,
                    year1,
                    date1,
                    x_coordinate,
                    y_cordinate,
                    offincident
                where
                    year1 = {year}
                    and date1 is not null
                    and x_coordinate is not null
                    and y_cordinate is not null
                    and offincident = {offincident}
                order by date1
                limit
                    {n}
                """

        results = client.get(ds_identifier,
                             query=query, content_type=content_type)
        df = pd.DataFrame.from_records(results)

        # DB Cleaning & Formatting
        for col in ['x_coordinate', 'y_cordinate']:
            df.loc[:, col] = df[col].apply(
                lambda x: float(x))
        df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(
            lambda x: float(x))
        df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(
            lambda x: float(x))
        df.loc[:, 'date1'] = df['date1'].apply(  # OJO AL SEPARADOR ' '
            lambda x: datetime.strptime(
                x.split(' ')[0], '%Y-%m-%d')
        )
        df['date1'] = df['date1'].dt.date

        df = df[['x_coordinate', 'y_cordinate', 'date1']]
        df.rename(columns={'x_coordinate': 'x',
                           'y_cordinate': 'y',
                           'date1': 'date'},
                  inplace=True)

        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if save:
            df.to_excel(path)
        return df


def get_stored_data(path='../predictivehp/data/SOCRATA_DATA_Dallas.xlsx'):
    return pd.read_excel(path, index_col=0)


def shps_processing(s_shp='', c_shp='', cl_shp=''):
    """

    Parameters
    ----------
    s_shp
    c_shp
    cl_shp

    Returns
    -------

    """
    streets, councils, c_limits = [None, ] * 3
    shps = {}
    if s_shp:
        streets = gpd.read_file(filename=s_shp)
        streets.crs = 2276
        streets.to_crs(epsg=3857, inplace=True)
    if c_shp:
        councils = gpd.read_file(filename=c_shp)
        councils.crs = 2276
        councils.to_crs(epsg=3857, inplace=True)
    if cl_shp:
        c_limits = gpd.read_file(filename=cl_shp)
        c_limits.crs = 2276
        c_limits.to_crs(epsg=3857, inplace=True)

    shps['streets'], shps['councils'], shps['c_limits'] = \
        streets, councils, c_limits

    return shps


# Plots

def lineplot(x, y, c='r', ls='-', lw=1,
             label='', x_label='', y_label='', title='',
             savefig=False, fname='line_plot', **kwargs):
    """
    Forma compacta de realizar un lineplot

    Parameters
    ----------
    x
    y
    c : str
    ls :str
    lw : atr
    label
    x_label : str
    y_label : str
    title : str
    savefig : bool
    fname : str

    Returns
    -------

    """
    if label:
        plt.plot(x, y, c=c, ls=ls, lw=lw, label=label, **kwargs)
        plt.legend(loc='best')
    else:
        plt.plot(x, y, c=c, ls=ls, lw=lw, **kwargs)

    if x_label:
        plt.xlabel(x_label, **kwargs)
    if y_label:
        plt.ylabel(y_label, **kwargs)
    if title:
        plt.title(title, pad=15, **kwargs)
    if savefig:
        plt.savefig(fname, dpi=200, bbox_inches='tight', **kwargs)


# STKDE

def checked_points(points, shp):
    """

    Parameters
    ----------
    points
    shp : gpd.GeoDataFrame
      Councils shp
    Returns
    -------
    np.ndarray
    """
    dallas_shp = shp
    df_points = pd.DataFrame(
        {'x': points[0, :], 'y': points[1, :], 't': points[2, :]}
    )
    inc_points = df_points[['x', 'y']].apply(lambda row:
                                             Point(row['x'], row['y']),
                                             axis=1)
    geo_inc = gpd.GeoDataFrame({'geometry': inc_points, 'day': df_points['t']})
    geo_inc.crs = dallas_shp.crs
    valid_inc = gpd.tools.sjoin(geo_inc, dallas_shp,
                                how='inner', op='intersects').reset_index()
    valid_inc_2 = valid_inc[['geometry', 'day']]

    x = valid_inc_2['geometry'].apply(lambda row: row.x)
    y = valid_inc_2['geometry'].apply(lambda row: row.y)
    t = valid_inc_2['day']

    return np.array([x, y, t])


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


def filter_cells(df, shp, verbose=False):
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

    print('\tFiltering cells...') if verbose else None
    print('\t\tLoading shapefile...') if verbose else None
    dallas_shp = shp
    print('\t\tCreating GeoDataframe...') if verbose else None
    geo_pd = gpd.GeoDataFrame(aux_df[[('geometry', ''), ('in_dallas', '')]],
                              crs=3857)

    # Borramos el segundo nivel ''
    geo_pd = geo_pd.T.reset_index(level=1, drop=True).T
    geo_pd.crs = dallas_shp.crs
    print('\t\tFiltering...') if verbose else None
    geo_pd = gpd.tools.sjoin(geo_pd, dallas_shp,
                             how='left',
                             op='intersects')[['in_dallas', 'index_right']]

    print('\t\tUpdating dataframe... ', end='') if verbose else None
    geo_pd.fillna(value={'index_right': 14}, inplace=True)  # para filtrar
    geo_pd.loc[geo_pd['index_right'] < 14, 'in_dallas'] = 1

    # Añadimos la columna del gpd filtrado al data inicial
    aux_df[[('in_dallas', '')]] = geo_pd[['in_dallas']]
    # Filtramos el data inicial con la columna añadida
    aux_df = aux_df[aux_df[('in_dallas', '')] == 1]

    print(f'finished!', end=" ") if verbose else None
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


def checked_points_pm(points, shp):
    """

    Parameters
    ----------
    points
    shp : gpd.GeoDataFrame
      Councils shp
    Returns
    -------
    np.ndarray
    """
    dallas_shp = shp
    df_points = pd.DataFrame(
        {'x': points[0, :], 'y': points[1, :]}
    )
    inc_points = df_points[['x', 'y']].apply(lambda row:
                                             Point(row['x'], row['y']),
                                             axis=1)
    geo_inc = gpd.GeoDataFrame({'geometry': inc_points})
    geo_inc.crs = dallas_shp.crs
    valid_inc = gpd.tools.sjoin(geo_inc, dallas_shp,
                                how='inner', op='intersects').reset_index()
    return len(valid_inc)


def find_c(area_array, c_list, ap):
    """

    Parameters
    ----------
    area_array:
      lista con ap's (self.ap)
    c_list
      c_vector
    ap
    """
    area_array = np.array(area_array)
    area_array = area_array - ap
    return c_list[np.argmin(np.abs(area_array))]


def find_hr_pai(values, area_array, ap):
    values = np.array(values)
    area_array = np.array(area_array)
    area_array = area_array - ap
    return values[np.argmin(np.abs(area_array))]


if __name__ == '__main__':
    area_array = [1, 0.7, 0.6, 0.5, 0.45, 0.33, 0]
    c = [0, 0.2, 0.3, 0.4, 0.6, 0.8, 1]

    hr = [1, 2, 3, 4, 5, 6, 7]
    pai = [8, 9, 10, 11, 12, 13, 14]

    print(find_c(area_array, np.array(c), ap=0.2))
    print(find_hr_pai(hr, area_array, ap=0.2))
    print(find_hr_pai(pai, area_array, ap=0.2))
