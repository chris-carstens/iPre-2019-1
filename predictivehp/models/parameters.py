# STKDE

import numpy as np
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

# Optimal Bandwidths

bw = np.array([1577.681, 1167.16, 35.549])

# Oct - Nov - Dic

w_day_oct, days_oct = monthrange(2017, 10)
w_day_nov, days_nov = monthrange(2017, 11)
w_day_dic, days_dic = monthrange(2017, 12)

days_oct_nov_dic = [date(2017, 10, i) for i in range(1, days_oct + 1)] + \
                   [date(2017, 11, i) for i in range(1, days_nov + 1)] + \
                   [date(2017, 12, i) for i in range(1, days_dic + 1)]

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
