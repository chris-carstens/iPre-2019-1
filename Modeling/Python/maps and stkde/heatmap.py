import numpy as np

import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from database_request import df

df = df[['x', 'y']]

dallas = gpd.read_file('../../Data/shapefiles/STREETS.shp')

fig, ax = plt.subplots(figsize=(15, 15))
ax.set_facecolor('xkcd:black')

dallas.plot(ax=ax, alpha=.4, color="gray")

nbins = 500
data = np.array(df[['x', 'y']])

x, y = data.T

k = gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

cmap2 = mpl.cm.get_cmap("jet")
cmap2.set_under("k")

heatmap = plt.pcolormesh(xi, yi, zi.reshape(xi.shape),
                         shading='gouraud',
                         cmap=cmap2,
                         vmin=.6e-10)

plt.title("Dallas Incidents - Heatmap",
          fontdict={'fontsize': 20,
                    'fontweight': 'bold'},
          pad=20)

plt.colorbar(heatmap, ax=ax, shrink=.4, aspect=10)

plt.show()

# plt.savefig("Dallas.pdf", format='pdf')
