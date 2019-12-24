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

data = pd.read_pickle('df.pkl')[[('geometry', ''),
                                 ('Dangerous_Oct', ''),
                                 ('Dangerous_pred_Oct', '')]]
d_streets = gpd.GeoDataFrame.from_file(
    filename='../../Data/Streets/STREETS.shp'
)
d_districts = gpd.GeoDataFrame.from_file(
    filename='../../Data/Councils/Councils.shp'
)
geo_data = gpd.GeoDataFrame(data, crs=d_streets)

# TODO
#       - Dividir geo_data en Dangerous/Non Dangerous
#       - Realizar plot para datos reales y datos generados por predicción

geo_data_oct = gpd.GeoDataFrame(data, crs=d_streets) \
    .drop(columns=[('Dangerous_pred_Oct', '')])
geo_data_p_oct = gpd.GeoDataFrame(data, crs=d_streets) \
    .drop(columns=[('Dangerous_Oct', '')])

d_streets.to_crs(epsg=3857, inplace=True)
d_districts.to_crs(epsg=3857, inplace=True)

fig, ax = plt.subplots(figsize=(20, 15))
ax.set_facecolor('xkcd:black')

handles = []

for district, data in d_districts.groupby('DISTRICT'):
    data.plot(ax=ax,
              color=prm.d_colors[district],
              linewidth=2.5,
              edgecolor="black")
    handles.append(mpatches.Patch(color=prm.d_colors[district],
                                  label=f"Dallas District {district}"))

handles.sort(key=lambda x: int(x._label.split(' ')[2]))
handles = [Line2D([], [], marker='o', color='red', label='Incident',
                  linestyle="None"),
           Line2D([0], [0], color="steelblue", label="Streets")] \
          + handles

d_streets.plot(ax=ax, alpha=0.4, color="dimgrey", zorder=2, label="Streets")

# geo_data.plot(ax=ax, markersize=17.5, color='red', marker='o',
#               zorder=3, label="Incidents")

plt.legend(loc="best", bbox_to_anchor=(0.1, 0.7),
           frameon=False, fontsize=13.5, handles=handles)

legends = ax.get_legend()
for text in legends.get_texts():
    text.set_color('white')

ax.set_axis_off()
fig.set_facecolor('black')
plt.show()
