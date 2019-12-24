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
geo_data = gpd.GeoDataFrame()

# d_districts = gpd.GeoDataFrame.from_file(
#     filename='../../Data/Councils/Councils.shp'
# )
#
# d_streets = gpd.GeoDataFrame.from_file(
#     filename='../../Data/Streets/STREETS.shp'
# )
#
# fig, ax = plt.subplots(figsize=(15, 15))
# ax.set_facecolor('xkcd:black')
#
# handles = []
#
# for district, data in d_districts.groupby('DISTRICT'):
#     data.plot(ax=ax,
#               color=prm.d_colors[district],
#               linewidth=2.5,
#               edgecolor="black")
#     handles.append(mpatches.Patch(color=prm.d_colors[district],
#                                   label=f"Dallas District {district}"))
#
# handles.sort(key=lambda x: int(x._label.split(' ')[2]))
# handles = [Line2D([], [], marker='o', color='red',
#                   label='Incident', linestyle="None"),
#            Line2D([0], [0], color="steelblue", label="Streets")] \
#           + handles
#
# d_streets.plot(ax=ax,
#                alpha=0.4,
#                color="steelblue",
#                zorder=2,
#                label="Streets")
#
# plt.legend(loc="lower right",
#            frameon=False,
#            fontsize=13.5,
#            handles=handles)
#
# ax.set_axis_off()
# plt.show()
