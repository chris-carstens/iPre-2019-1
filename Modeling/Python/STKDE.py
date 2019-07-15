"""STKDE"""

import numpy as np
import pandas as pd
from time import time
import datetime

import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

from statsmodels.nonparametric.kernel_density import KDEMultivariate

from sodapy import Socrata
import credentials as cre
import parameters as params


# Observaciones
#
# 1. 3575 Incidents
# Training data 2926 incidents (January 1st - October 31st)
# Testing data 649 incidents (November 1st - December 31st)
#
# 2. Se requiere que la muestra sea "estable" en el periodo analizado

def _time(fn):
    def inner_1(*args, **kwargs):
        start = time()

        fn(*args, **kwargs)

        print(f"\nFinished in {round(time() - start, 3)} sec")

    return inner_1


class STKDE:
    """
    Class for a spatio-temporal kernel density estimation
    """

    def __init__(self,
                 n: int = 1000,
                 year: str = "2017",
                 bw=None):
        """
        n: Número de registros que se piden a la database.

        year: Año de los registros pedidos

        t_model: Entrenamiento del modelo, True en caso de que se quieran
        usar los métodos contour_plot o heatmap.
        """

        self.data = []
        self.training_data = []  # 3000
        self.testing_data = []  # 600

        self.n = n
        self.year = year

        self.get_data()

        self.kde = None

        self.train_model(
                x=np.array(self.training_data[['x']]),
                y=np.array(self.training_data[['y']]),
                t=np.array(self.training_data[['y_day']]),
                bw=bw
        )

    @_time
    def get_data(self):
        """
        Requests data using the Socrata API and saves in the
        self.data variable
        """

        print("\nRequesting data...")

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
                    time1,
                    x_coordinate,
                    y_cordinate,
                    offincident
                where
                    year1 = {self.year}
                    and date1 is not null
                    and time1 is not null
                    and x_coordinate is not null
                    and y_cordinate is not null
                    and offincident = 'BURGLARY OF HABITATION - FORCED ENTRY'
                order by date1
                limit
                    {self.n}
                """  #  571000 max. 09/07/2019

            results = client.get(cre.socrata_dataset_identifier,
                                 query=query,
                                 content_type='json')

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            df = pd.DataFrame.from_records(results)

            # DB Cleaning & Formatting

            df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(
                    lambda x: float(x))
            df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(
                    lambda x: float(x))
            df.loc[:, 'date1'] = df['date1'].apply(
                    lambda x: datetime.datetime.strptime(
                            x.split('T')[0], '%Y-%m-%d')
            )

            df = df[['x_coordinate', 'y_cordinate', 'date1']]
            df.loc[:, 'y_day'] = df["date1"].apply(
                    lambda x: x.timetuple().tm_yday
            )

            df.rename(columns={'x_coordinate': 'x',
                               'y_cordinate': 'y',
                               'date1': 'date'},
                      inplace=True)

            # Reducción del tamaño de la DB

            df = df.sample(n=3600,
                           replace=False,
                           random_state=250499)

            df.sort_values(by=['date'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            self.data = df

            # División en training y testing data

            self.training_data = self.data[
                self.data["date"].apply(lambda x: x.month) <= 10
                ]

            self.testing_data = self.data[
                self.data["date"].apply(lambda x: x.month) > 10
                ]

            # print(self.training_data.head())
            # print(self.training_data.tail())
            # print(self.testing_data.head())
            # print(self.testing_data.tail())

            print("\n"
                  f"n = {self.n} incidents requested  Year = {self.year}"
                  "\n"
                  f"{self.data.shape[0]} incidents successfully retrieved")

    @_time
    def train_model(self, x, y, t, bw=None):
        """
        Entrena el modelo y genera un KDE

        bw: Si es un arreglo, este debe contener los bandwidths
        dados por el usuario
        """

        print("\nBuilding KDE...")

        if bw is not None:
            self.kde = KDEMultivariate(data=[x, y, t],
                                       var_type='ccc',
                                       bw=bw)
            print(f"\nGiven Bandwidths: \n\n"
                  f"hx = {round(bw[0], 3)} ft\n"
                  f"hy = {round(bw[1], 3)} ft\n"
                  f"ht = {round(bw[2], 3)} days")

        else:
            self.kde = KDEMultivariate(data=[x, y, t],
                                       var_type='ccc',
                                       bw='cv_ml')

            print(f"\nOptimal Bandwidths: \n\n"
                  f"hx = {round(self.kde.bw[0], 3)} ft\n"
                  f"hy = {round(self.kde.bw[1], 3)} ft\n"
                  f"ht = {round(self.kde.bw[2], 3)} days")

    @_time
    def data_barplot(self,
                     pdf: bool = False):
        """
        Bar Plot

        pdf: True si se desea guardar el plot en formato pdf
        """

        print("\nPlotting Bar Plot...")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.tick_params(axis='x', length=0)

        for i in range(1, 13):

            count = self.data[
                (self.data["date"].apply(lambda x: x.month) == i)
            ].shape[0]

            plt.bar(x=i, height=count, width=0.25, color=["black"])
            plt.text(x=i - 0.275, y=count + 5, s=str(count))

        plt.xticks(
                [i for i in range(1, 13)],
                [datetime.datetime.strptime(str(i), "%m").strftime('%b')
                 for i in range(1, 13)]
        )

        sb.despine()

        plt.xlabel("Month",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold'},
                   labelpad=10
                   )
        plt.ylabel("Count",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold'},
                   labelpad=7.5
                   )

        if pdf:
            plt.savefig(f"output/barplot.pdf", format='pdf')

        plt.show()

    @_time
    def spatial_pattern(self,
                        pdf: bool = False):
        """
        Spatial pattern of incidents

        pdf: True si se desea guardar el plot en formato pdf
        """

        print("\nPlotting Spatial Pattern of incidents...")

        dallas = gpd.read_file('../Data/shapefiles/STREETS.shp')

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        # US Survey Foot: 0.3048 m
        # print("\n", f"EPSG: {dallas.crs['init'].split(':')[1]}")  # 2276

        geometry = [Point(xy) for xy in zip(
                np.array(self.testing_data[['x']]),
                np.array(self.testing_data[['y']]))
                    ]
        geo_df = gpd.GeoDataFrame(self.testing_data,
                                  crs=dallas.crs,
                                  geometry=geometry)

        dallas.plot(ax=ax,
                    alpha=.4,
                    color="gray",
                    zorder=1)
        geo_df.plot(ax=ax,
                    markersize=10,
                    color='red',
                    marker='o',
                    label='Incident',
                    zorder=2)
        plt.legend(prop={'size': 15})

        plt.title(f"Dallas Incidents - Spatial Pattern\n"
                  f"Year = {self.year}",
                  fontdict={'fontsize': 15,
                            'fontweight': 'bold'},
                  pad=20)

        if pdf:
            plt.savefig("output/spatial_pattern.pdf", format='pdf')
        plt.show()

    @_time
    def contour_plot(self,
                     bins: int,
                     ti: int,
                     pdf: bool = False):
        """
        Draw the contour lines

        bins:

        ti:

        pdf: True si se desea guardar el plot en formato pdf
        """

        print("\nPlotting Contours...")

        dallas = gpd.read_file('../Data/shapefiles/STREETS.shp')

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax,
                    alpha=.4,
                    color="gray",
                    zorder=1)

        x, y = np.mgrid[
               np.array(self.testing_data[['x']]).min():
               np.array(self.testing_data[['x']]).max():bins * 1j,
               np.array(self.testing_data[['y']]).min():
               np.array(self.testing_data[['y']]).max():bins * 1j
               ]

        z = self.kde.pdf(np.vstack([x.flatten(),
                                    y.flatten(),
                                    ti * np.ones(x.size)]))
        # z_2 = z * 3000 * (10 ** 6) / (.304) # P. Elwin

        contourplot = plt.contour(x, y, z.reshape(x.shape),
                                  cmap='jet',
                                  zorder=2)

        plt.title(f"Dallas Incidents - Contourplot\n"
                  f"n = {self.data.shape[0]}    Year = {self.year}",
                  fontdict={'fontsize': 15,
                            'fontweight': 'bold'},
                  pad=20)
        plt.colorbar(contourplot,
                     ax=ax,
                     shrink=.4,
                     aspect=10)

        if pdf:
            plt.savefig("output/dallas_contourplot.pdf", format='pdf')
        plt.show()

    @_time
    def heatmap(self,
                bins: int,
                ti: int,
                pdf: bool = False):
        """
        Plots the heatmap associated to a given t_i

        bins:

        ti:

        pdf:
        """

        print("\nPlotting Heatmap...")

        dallas = gpd.read_file('../Data/shapefiles/STREETS.shp')

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax,
                    alpha=.4,  # Ancho de las calles
                    color="gray",
                    zorder=1)

        x, y = np.mgrid[
               np.array(self.testing_data[['x']]).min():
               np.array(self.testing_data[['x']]).max():bins * 1j,
               np.array(self.testing_data[['y']]).min():
               np.array(self.testing_data[['y']]).max():bins * 1j
               ]

        z = self.kde.pdf(np.vstack([x.flatten(),
                                    y.flatten(),
                                    ti * np.ones(x.size)]))
        # z = np.ma.masked_array(z, z < .1e-11)

        heatmap = plt.pcolormesh(x, y, z.reshape(x.shape),
                                 shading='gouraud',
                                 alpha=.2,
                                 cmap=mpl.cm.get_cmap("jet"),
                                 zorder=2)

        plt.title(f"Dallas Incidents - Heatmap\n"
                  f"n = {self.data.shape[0]}   Year = {self.year}",
                  fontdict={'fontsize': 15,
                            'fontweight': 'bold'},
                  pad=20)
        cbar = plt.colorbar(heatmap,
                            ax=ax,
                            shrink=.5,
                            aspect=10)
        cbar.solids.set(alpha=1)

        if pdf:
            plt.savefig("output/dallas_heatmap.pdf", format='pdf')
        plt.show()


# Data
#
# 2012 - 58     incidents
# 2013 - 186    incidents
# 2014 - 54985  incidents
# 2015 - 94923  incidents   √
# 2016 - 102132 incidents   √
# 2017 - 96411  incidents   √
# 2018 - 98477  incidents
# 2019 - 64380  incidents (y creciendo)

if __name__ == "__main__":
    st = time()

    dallas_stkde = STKDE(n=150000,
                         year="2016",
                         bw=params.bw)
    # dallas_stkde.data_barplot(pdf=False)
    # dallas_stkde.spatial_pattern(pdf=False)
    # dallas_stkde.contour_plot(bins=100,
    #                           ti=183,
    #                           pdf=False)
    # dallas_stkde.heatmap(bins=100,
    #                      ti=365,
    #                      pdf=False)

    # for i in range(365):
    #     dallas_stkde.heatmap(bins=100,
    #                          ti=i,
    #                          pdf=False)

    bins = 100


    @_time
    def d_estimation():
        print("\nCreating the 3D grid...")

        x, y, t = np.mgrid[
                  np.array(dallas_stkde.testing_data[['x']]).min():
                  np.array(dallas_stkde.testing_data[['x']]).max():bins * 1j,
                  np.array(dallas_stkde.testing_data[['y']]).min():
                  np.array(dallas_stkde.testing_data[['y']]).max():bins * 1j,
                  np.array(dallas_stkde.testing_data[['y_day']]).min():
                  np.array(dallas_stkde.testing_data[['y_day']]).max():bins * 1j
                  ]

        print("\nEstimating densities...")

        d = dallas_stkde.kde.pdf(
                np.vstack([
                    x.flatten(),
                    y.flatten(),
                    t.flatten()
                ]))

        # print("\n", len(d))

        # print("\nx\n", x)
        # print("\ny\n", y)
        # print("\nt\n", t)


    # d_estimation()

    print(f"\nTotal time: {round((time() - st) / 60, 3)} min")
