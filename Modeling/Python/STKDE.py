"""STKDE"""

import numpy as np
import pandas as pd
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from sodapy import Socrata
import credentials as cre


# Observaciones
#
# 1. 3575 Incidents
# Training data 2926 incidents (January 1st - October 31st)
# Testing data 649 incidents (November 1st - December 31st)
#
# 2. Se requiere que la muestra sea "estable" en el periodo analizado

class STKDE:
    """STKDE class for a spatio-temporal kernel density estimation"""

    def __init__(self, n: int = 1000, year: str = "2017"):
        self.data = []

        self.x = np.array(self.data[["x"]])
        self.y = np.array(self.data[["y"]])
        self.t = np.array(self.data[["date_ordinal"]])

        self.training_data = []  # 3000
        self.testing_data = []  # 600
        self.n = n
        self.year = year

        self.get_data()

    def get_data(self):
        """Requests data using the Socrata API and saves in the
        self.data variable"""

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
                    y_cordinate
                where
                    year1 = {self.year}
                    and date1 is not null
                    and time1 is not null
                    and x_coordinate is not null
                    and y_cordinate is not null
                order by date1
                limit
                    {self.n}
                """  #  530000 max. 11/04

            results = client.get(cre.socrata_dataset_identifier,
                                 query=query,
                                 content_type='json')

            pd.set_option('display.max_columns', None)
            # pd.set_option('display.max_rows', None)

            df = pd.DataFrame.from_records(results)

            # print(df.head())

            # DB Cleaning & Formatting

            df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(
                    lambda x: float(x))
            df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(
                    lambda x: float(x))
            df.loc[:, 'date1'] = df['date1'].apply(
                    lambda x: datetime.datetime.strptime(x.split('T')[0],
                                                         '%Y-%m-%d'))

            df = df[['x_coordinate', 'y_cordinate', 'date1']]
            df.loc[:, 'date_ordinal'] = df.apply(
                    lambda row: row.date1.toordinal(),
                    axis=1)

            df.rename(columns={'x_coordinate': 'x',
                               'y_cordinate': 'y',
                               'date1': 'date'},
                      inplace=True)

            self.data = df

            print("\n"
                  f"n = {self.n}   Year = {self.year}"
                  "\n"
                  f"{self.data.shape[0]} rows successfully retrieved")

    def data_histogram(self):
        """Plots a histogram of the data"""

        df = self.data
        months = [i for i in range(1, 13)]

        fig, ax = plt.subplots()
        bins = np.arange(1, 14)

        ax.hist(df["date"].apply(lambda x: x.month),
                bins=bins,
                edgecolor="k",
                align='left')
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels(
                [datetime.date(1900, i, 1).strftime('%b') for i in bins[:-1]])

        plt.title(f"Database Request\n"
                  f"n = {self.n}   Year = {self.year}",
                  fontdict={'fontsize': 15,
                            'fontweight': 'bold'},
                  pad=20)

        # plt.savefig(f"histogram_{n}.pdf", format='pdf')
        plt.show()

    def contour_plot(self):
        """Draw the contour lines"""

        df = self.data[['x', 'y']]

        dallas = gpd.read_file('../../Data/shapefiles/STREETS.shp')

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax, alpha=.4, color="gray")
        # print(dallas.crs)

        # 0.3048 m US Survey Foot

        geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
        geo_df = gpd.GeoDataFrame(df,
                                  crs=dallas.crs,
                                  geometry=geometry)

        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax, alpha=.4, color="gray", zorder=1)
        geo_df.plot(ax=ax, markersize=10, color='red', marker='o',
                    label='Incident',
                    zorder=2)
        plt.legend(prop={'size': 15})

        nbins = 100
        data = np.array(df[['x', 'y']])

        x, y = data.T

        k = gaussian_kde(data.T)
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j,
                 y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # zi_2 = zi * 3000 * (10 ** 6) / (.304) # P. Elwin

        contourplot = plt.contour(xi, yi, zi.reshape(xi.shape), cmap='jet',
                                  zorder=3)

        plt.title(f"Dallas Incidents - Contourplot\n"
                  f"n = {self.n}    Year = {self.year}",
                  fontdict={'fontsize': 15,
                            'fontweight': 'bold'},
                  pad=20)
        plt.colorbar(contourplot, ax=ax, shrink=.4, aspect=10)

        # plt.savefig("Dallas.pdf", format='pdf')
        plt.show()

    def heatmap(self):
        df = self.data[['x', 'y']]

        dallas = gpd.read_file('../Data/shapefiles/STREETS.shp')

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax, alpha=.4, color="gray")

        nbins = 100
        data = np.array(df[['x', 'y']])

        x, y = data.T

        k = gaussian_kde(data.T)
        xi, yi = np.mgrid[self.x.min():self.x.max():nbins * 1j,
                 self.y.min():self.y.max():nbins * 1j]

        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        cmap2 = mpl.cm.get_cmap("jet")
        cmap2.set_under("k")

        heatmap = plt.pcolormesh(xi, yi, zi.reshape(xi.shape),
                                 shading='gouraud',
                                 cmap=cmap2,
                                 vmin=.6e-10)

        plt.title(f"Dallas Incidents - Heatmap\n"
                  f"n = {self.n}   Year = {self.year}",
                  fontdict={'fontsize': 15,
                            'fontweight': 'bold'},
                  pad=20)
        plt.colorbar(heatmap, ax=ax, shrink=.4, aspect=10)

        # plt.savefig("Dallas.pdf", format='pdf')
        plt.show()

    def calculate_bandwidths(self):
        """Calculate the hx, hy and ht bandwidths"""

        dens_u = KDEMultivariate(data=[self.x, self.y, self.t],
                                 var_type='ccc',
                                 bw='cv_ml')

        hx, hy, ht = dens_u.bw

        print(f"\nOptimal Bandwidths: \n\n"
              f"hx = {round(hx, 3)} \n"
              f"hy = {round(hy, 3)} \n"
              f"ht = {round(ht, 3)}")


dallas_stkde = STKDE(n=3600, year="2016")

# dallas_stkde.contour_plot()
# dallas_stkde.heatmap()
dallas_stkde.data_histogram()
# dallas_stkde.calculate_bandwidths()
