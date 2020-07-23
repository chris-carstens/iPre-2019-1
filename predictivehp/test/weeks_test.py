from calendar import monthrange
from datetime import date

import pandas as pd

from predictivehp.processing.data_processing import get_data

# 5 weeks train: 4 feature weeks, 1 label week
# Final Date: 31-10-2017, de ahí hacia atrás formamos 4 semanas. La más
# cercana a Final Date es considerada label week, el resto son features
# Obs. Para predecir: hay desplazamiento en una semana en X e y.

df, _, _, _ = get_data()
layers = [f"Layer_{i}" for i in range(8)]

n_weeks = 5
f_date = date(2017, 10, 31)  # For training
i_date = date(2017, 11, 1)  # For prediction
weeks = []
yy, mm, dd = f_date.year, f_date.month, f_date.day
for _ in range(0, n_weeks):
    if dd - 7 >= 1:  # Descenso en días válido
        dd -= 7
        dt = date(yy, mm, dd + 1)
    else:
        if mm - 1 >= 1:  # Descenso en meses válido
            mm -= 1
            diff1 = dd
            diff2 = 7 - diff1
            dd = monthrange(yy, mm)[1] - diff2
            dt = date(yy, mm, dd + 1)
        else:  # Descendemos un año y nos paramos al final de éste - diff2
            yy -= 1
            mm = 12
            diff1 = dd
            diff2 = 7 - diff1
            dd = monthrange(yy, mm)[1] - diff2
            dt = date(yy, mm, dd + 1)
    weeks.append(dt)
weeks.reverse()

X_train_cols = pd.MultiIndex.from_product([weeks[:-1], layers])
y_train_cols = pd.MultiIndex.from_product([[f"{weeks[-1]}"], ['Dangerous']])
X_test_cols = pd.MultiIndex.from_product([weeks[1:], layers])
y_test_cols = pd.MultiIndex.from_product([[f"{i_date}"], ['Dangerous']])

X_train = pd.DataFrame(columns=X_train_cols)
y_train = pd.DataFrame(columns=y_train_cols)
X_test = pd.DataFrame(columns=X_test_cols)
y_test = pd.DataFrame(columns=y_test_cols)
