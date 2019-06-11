import pandas as pd
import numpy as np

from database_request import df
from statsmodels.nonparametric.kernel_density import KDEMultivariate

x = np.array(df[['x']])
y = np.array(df[['y']])
date = np.array(df[['date_ordinal']])

dens_u = KDEMultivariate(data=[x, y, date],
                         var_type='ccc',
                         bw='cv_ml')

hx, hy, ht = dens_u.bw
print(f"\nOptimal Bandwidths: \n\n"
      f"hx = {round(hx, 3)} \n"
      f"hy = {round(hy, 3)} \n"
      f"ht = {round(ht, 3)}")

print(sum(dens_u.pdf()))