# Chris:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-07-07

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import geopandas as gpd

from database_request import df
from statsmodels.nonparametric.kernel_density import KDEMultivariate


class Prediction:
    def __init__(self, df, nbins, kde):
        self.df = df
        self.nbins = nbins
        self.kde = kde

        self.dens_u = None
        self.date = None
        self.x = None
        self.y = None
        self.heatmap = None

    def density_function(self):
        self.x = np.array(self.df[['x']])
        self.y = np.array(self.df[['y']])
        self.date = np.array(self.df[['date_ordinal']])

        self.dens_u = self.kde(data=[self.x, self.y, self.date],
                               var_type='ccc',
                               bw='cv_ml')

    def predict_map(self):
        self.density_function()

        dallas = gpd.read_file('../../Data/shapefiles/STREETS.shp')

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax, alpha=.4, color="gray")

        xi, yi = np.mgrid[
                 self.x.min():self.x.max():self.nbins * 1j,
                 self.y.min():self.y.max():self.nbins * 1j
                 ]
        zi = self.dens_u.pdf(
                np.vstack([xi.flatten(),
                           yi.flatten(),
                           735234 * np.ones(xi.size)]))

        cmap2 = mpl.cm.get_cmap("jet")
        cmap2.set_under("k")

        self.heatmap = plt.pcolormesh(xi, yi, zi.reshape(xi.shape),
                                      shading='gouraud')

        plt.title("Dallas Incidents - Heatmap",
                  fontdict={'fontsize': 20,
                            'fontweight': 'bold'},
                  pad=20)
        plt.colorbar(self.heatmap, ax=ax, shrink=.4, aspect=10)


a = Prediction(df, 100, KDEMultivariate)
a.predict_map()
