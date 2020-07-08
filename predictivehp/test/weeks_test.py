from datetime import date

import pandas as pd

from predictivehp.processing.data_processing import get_data

# 4 weeks train: 3 features, 1 label
# para predecir: hay desplazamiento en una semana.

df, _, _, _ = get_data()
n_weeks = 3

weeks = [f"W {date(2017, 10, 18 - i)}" for i in [0, 7, 14][::-1]]
layers = [f"Layer_{i}" for i in range(8)]

Xt_columns = pd.MultiIndex.from_product([weeks, layers])
X_train = pd.DataFrame(columns=Xt_columns)
yt_columns = pd.MultiIndex.from_product([[f"W {date(2017, 10, 25)}"],
                                      ['Dangerous']])
y_train = pd.DataFrame(columns=yt_columns)


weeks = [f"W {date(2017, 10, 25 - i)}" for i in [0, 7, 14][::-1]]
columns = pd.MultiIndex.from_product([weeks, layers])
X_test = pd.DataFrame(columns=columns)

columns = pd.MultiIndex.from_product([[f"W {date(2017, 11, 1)}"],
                                      ['Dangerous']])
y_test = pd.DataFrame(columns=columns)

# TODO
#   Weeks similar a 3 meses
#   Eliminar hardcodeo en el nro de semanas
