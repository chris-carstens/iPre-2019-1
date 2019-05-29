import numpy as np

import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from database_request import df

df = df[['x', 'y']]

dallas = gpd.read_file('../../Data/shapefiles/STREETS.shp')

fig, ax = plt.subplots(figsize=(15, 15))
ax.set_facecolor('xkcd:black')

dallas.plot(ax=ax, alpha=.4, color="gray")
# print(dallas.crs)

geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
geo_df = gpd.GeoDataFrame(df,
                          crs=dallas.crs,
                          geometry=geometry)

ax.set_facecolor('xkcd:black')

dallas.plot(ax=ax, alpha=.4, color="gray", zorder=1)
geo_df.plot(ax=ax, markersize=10, color='red', marker='o', label='Incident',
            zorder=2)
plt.legend(prop={'size': 15})

nbins = 100
data = np.array(df[['x', 'y']])

x, y = data.T

k = gaussian_kde(data.T)
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
# zi_2 = zi * 3000 * (10 ** 6) / (.304) # P. Elwin

contourplot = plt.contour(xi, yi, zi.reshape(xi.shape), cmap='jet', zorder=3)

plt.title("Dallas Incidents - Contourplot",
          fontdict={'fontsize': 20,
                    'fontweight': 'bold'},
          pad=20)
plt.colorbar(contourplot, ax=ax, shrink=.4, aspect=10)
plt.show()

# plt.savefig("Dallas.pdf", format='pdf')
