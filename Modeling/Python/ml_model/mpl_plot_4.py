"""
mpl_plot_4.py

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 21-12-19
Pontifical Catholic University of Chile
"""

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import parameters as prm


def prediction_plot_4(geodata_d, geodata_nd):
    """
    Plotea las celdas de Dallas reconocidas como peligrosas o no-peligrosas
    de acuerdo al algoritmo.

    :param gpd.GeoDataFrame geodata_d: gdf con los puntos de celdas peligrosas
    :param gpd.GeoDataFrame geodata_nd: gdf con los puntos de celdas
        no-peligrosas
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_facecolor('xkcd:black')

    for district, data in d_districts.groupby('DISTRICT'):
        data.plot(ax=ax,
                  color=prm.d_colors[district],
                  linewidth=2.5,
                  edgecolor="black")

    handles = [Line2D([], [], marker='o', color='red',
                      label='Dangerous Cell',
                      linestyle='None'),
               Line2D([], [], marker='o', color="blue",
                      label="Non-Dangerous Cell",
                      linestyle='None')]

    d_streets.plot(ax=ax, alpha=0.4, color="dimgrey", zorder=2,
                   label="Streets")
    geodata_nd.plot(ax=ax, markersize=10, color='blue', marker='o',
                    zorder=3, label="Incidents")
    geodata_d.plot(ax=ax, markersize=10, color='red', marker='o',
                   zorder=3, label="Incidents")

    plt.legend(loc="best", bbox_to_anchor=(0.1, 0.7),
               frameon=False, fontsize=13.5, handles=handles)

    legends = ax.get_legend()
    for text in legends.get_texts():
        text.set_color('white')

    ax.set_axis_off()
    fig.set_facecolor('black')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # TODO
    #       - Dividir geo_data en Dangerous/Non Dangerous
    #       - Realizar plot para datos reales y datos generados por predicción
    #       - Trabajar el FutureWarning de .to_crs()

    print('Reading pickle...')
    data = pd.read_pickle('df.pkl')[[('geometry', ''),
                                     ('Dangerous_Oct', ''),
                                     ('Dangerous_pred_Oct', '')]]

    # Quitamos el nivel ''
    data = data.T.reset_index(level=1, drop=True).T

    # Creamos el df para los datos reales (1) y predichos (2).
    data1 = data[['geometry', 'Dangerous_Oct']]
    data2 = data[['geometry', 'Dangerous_pred_Oct']]

    # Filtramos las celdas detectadas como Dangerous para reducir los tiempos
    # de cómputo.
    data1_d = data1[data1['Dangerous_Oct'] == 1]
    data1_nd = data1[data1['Dangerous_Oct'] == 0]
    geodata1_d = gpd.GeoDataFrame(data1_d)
    geodata1_nd = gpd.GeoDataFrame(data1_nd)

    data2_d = data2[data2['Dangerous_pred_Oct'] == 1]
    data2_nd = data2[data2['Dangerous_pred_Oct'] == 0]
    geodata2_d = gpd.GeoDataFrame(data2_d)
    geodata2_nd = gpd.GeoDataFrame(data2_nd)

    # Realizamos los joins de las tablas, la idea es obtener un:
    # left outer join, inner join, right outer join
    # print('Making joins...')
    # print('\tLeft join...')
    # left = data1.join(other=data2,
    #                   how='left',
    #                   rsuffix='_other')
    # print('\tRight join...')
    # right = data2.join(other=data1,
    #                    how='right',
    #                    rsuffix='_other')
    # print('\tInner join...')
    # inner = data1.join(other=data2,
    #                    how='inner',
    #                    rsuffix='_other')
    #
    # # Borramos las columnas de other
    # left.drop(columns='geometry_other', inplace=True)
    # right.drop(columns='geometry_other', inplace=True)
    # inner.drop(columns='geometry_other', inplace=True)

    print('Reading shapefiles...')
    d_streets = gpd.GeoDataFrame.from_file(
        filename='../../Data/Streets/STREETS.shp'
    )
    d_districts = gpd.GeoDataFrame.from_file(
        filename='../../Data/Councils/Councils.shp'
    )

    d_streets.to_crs(epsg=3857, inplace=True)
    d_districts.to_crs(epsg=3857, inplace=True)

    prediction_plot_4(geodata1_d, geodata1_nd)
    prediction_plot_4(geodata2_d, geodata2_nd)
