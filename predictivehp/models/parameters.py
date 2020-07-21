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
bw2 = np.array([7859.14550575/4,10910.84908688/4,30.38113667])


# Oct - Nov - Dic
w_day_jan, n_days_jan = monthrange(2017, 1)
w_day_feb, n_days_feb = monthrange(2017, 2)
w_day_mar, n_days_mar = monthrange(2017, 3)
w_day_apr, n_days_apr = monthrange(2017, 4)
w_day_may, n_days_may = monthrange(2017, 5)
w_day_jun, n_days_jun = monthrange(2017, 6)
w_day_jul, n_days_jul = monthrange(2017, 7)
w_day_aug, n_days_aug = monthrange(2017, 8)
w_day_sep, n_days_sep = monthrange(2017, 9)
w_day_oct, n_days_oct = monthrange(2017, 10)
w_day_nov, n_days_nov = monthrange(2017, 11)
w_day_dec, n_days_dec = monthrange(2017, 12)

days_jan = [date(2017, 1, i) for i in range(1, n_days_jan + 1)]
days_feb = [date(2017, 2, i) for i in range(1, n_days_feb + 1)]
days_mar = [date(2017, 3, i) for i in range(1, n_days_mar + 1)]
days_apr = [date(2017, 4, i) for i in range(1, n_days_apr + 1)]
days_may = [date(2017, 5, i) for i in range(1, n_days_may + 1)]
days_jun = [date(2017, 6, i) for i in range(1, n_days_jun + 1)]
days_jul = [date(2017, 7, i) for i in range(1, n_days_jul + 1)]
days_aug = [date(2017, 8, i) for i in range(1, n_days_aug + 1)]
days_sep = [date(2017, 9, i) for i in range(1, n_days_sep + 1)]
days_oct = [date(2017, 10, i) for i in range(1, n_days_oct + 1)]
days_nov = [date(2017, 11, i) for i in range(1, n_days_nov + 1)]
days_dec = [date(2017, 12, i) for i in range(1, n_days_dec + 1)]

days_by_month = {1: n_days_jan, 2: n_days_feb, 3: n_days_mar, 4: n_days_apr, 5: n_days_may, 6: n_days_jun,
                 7: n_days_jul, 8: n_days_aug, 9: n_days_sep, 10: n_days_oct, 11: n_days_nov, 12: n_days_dec}
days_year = [days_jan, days_feb, days_mar, days_apr, days_may, days_jun, days_jul, days_aug, days_sep,
             days_oct, days_nov, days_dec]

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
