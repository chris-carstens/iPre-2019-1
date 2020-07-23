from datetime import date

import pandas as pd

from predictivehp.processing.data_processing import get_data
# from sklearn.model_selection import train_test_split

# 4 weeks train: 3 features, 1 label
# Final Date: 31-10-2017, de ahí hacia atrás formamos 4 semanas. La más
# cercana a Final Date es considerada label, el resto son features

# Obs. Para predecir: hay desplazamiento en una semana.

n_weeks = 4
f_date = date(2017, 10, 31)

weeks = []
yy, mm, _ = f_date.
for n in range(n_weeks):
    try:
        date(2017, 10, 18 - n)
    except ValueError as err:
        pass

# df, _, _, _ = get_data()
#
# weeks = [f"W {date(2017, 10, 18 - i)}" for i in [0, 7, 14][::-1]]
# layers = [f"Layer_{i}" for i in range(8)]
#
# Xt_columns = pd.MultiIndex.from_product([weeks, layers])
# X_train = pd.DataFrame(columns=Xt_columns)
# yt_columns = pd.MultiIndex.from_product([[f"W {date(2017, 10, 25)}"],
#                                       ['Dangerous']])
# y_train = pd.DataFrame(columns=yt_columns)
#
#
# weeks = [f"W {date(2017, 10, 25 - i)}" for i in [0, 7, 14][::-1]]
# columns = pd.MultiIndex.from_product([weeks, layers])
# X_test = pd.DataFrame(columns=columns)
#
# columns = pd.MultiIndex.from_product([[f"W {date(2017, 11, 1)}"],
#                                       ['Dangerous']])
# y_test = pd.DataFrame(columns=columns)

# TODO
#   Weeks similar a 3 meses
#   Eliminar hardcodeo en el nro de semanas
