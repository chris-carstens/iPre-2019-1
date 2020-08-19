"""
_config.py

Seteamos parámetros de ploteo, etc.
"""

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

rc = {
    'figure.facecolor': 'black',
    'figure.figsize': (6.75, 4),  # (6.0, 4.0) defaults

    'xtick.color': 'white',
    'xtick.major.size': 3,
    'xtick.top': False,
    'xtick.bottom': True,

    'ytick.color': 'white',
    'ytick.major.size': 3,
    'ytick.left': True,
    'ytick.right': False,

    'axes.facecolor': '#030303',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'white',
    'axes.grid': True,
    'axes.axisbelow': True,

    'text.color': 'white',

    'legend.shadow': True,
    'legend.framealpha': 1.0,

    'grid.color': '#350000',
    'grid.alpha': 1.0,
    'grid.linestyle': '-',
    'grid.linewidth': 0.4,
}

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

# Optimal Bandwidths
bw = np.array([1577.681, 1167.16, 30.549])
bw_stkde = np.array([
    7859.14550575 / 10,
    10910.84908688 / 10,
    24.38113667
])

if __name__ == '__main__':
    import matplotlib as mpl

    mpl.rcParams.update(rc)
    print(mpl.rc_params())
