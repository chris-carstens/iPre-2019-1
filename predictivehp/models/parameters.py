# STKDE

import numpy as np
import matplotlib as mpl
from calendar import monthrange
from datetime import date

d_colors = {
    "1": "darksalmon",
    "2": "royalblue",
    "3": "mediumpurple",
    "4": "green",
    "5": "tan",
    "6": "brown",
    "7": "pink",
    "8": "olive",
    "9": "peru",
    "10": "orange",
    "11": "darkkhaki",
    "12": "cadetblue",
    "13": "crimson",
    "14": "thistle"
}

# MATPLOTLIB RC

mpl.rcdefaults()
rc = {
    'figure.facecolor': 'black',
    'figure.figsize': (6.75, 4),  # Values for Jup. Lab // (6.0, 4.0) default

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

    'legend.shadow': True,
    'legend.framealpha': 1.0,

    'grid.color': '#250000',
}
mpl.rcParams.update(rc)
f_size = mpl.rcParams.get('figure.figsize')

# Optimal Bandwidths

bw = np.array([1577.681, 1167.16, 30.549])
bw_stkde = np.array([7859.14550575/10, 10910.84908688/10, 24.38113667])

# Obs. Limites de dallas en epsg: 3857

# ML array([-10801798.95312305,   3842929.67045132, -10738624.98979845,
#          3897486.42924465])

d_limits = {
    'x_min': -10801798.95312305, 'x_max': -10738624.98979845,
    'y_min': 3842929.67045132, 'y_max': 3897486.42924465
}

# ProMap

# bw = {'x': 1577.681, 'y': 1167.16, 't': 35.549}
# hx = 100
# hy = 100

if __name__ == '__main__':
    pass
