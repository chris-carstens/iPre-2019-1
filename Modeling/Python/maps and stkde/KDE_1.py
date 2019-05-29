
import pandas as pd
import numpy as np

from statsmodels.nonparametric.kernel_density import KDEMultivariate

nobs = 300
np.random.seed(1234)  # Seed random generator

c1 = np.random.normal(size=(nobs, 1))
c2 = np.random.normal(2, 1, size=(nobs, 1))

dens_u = KDEMultivariate(data=[c1, c2],
                         var_type='cc',
                         bw='cv_ml')

# Para obtener los bandwidths de acuerdo al método: likelihood
# cross-validation


print(df)
print("hi", flush=False)
print(dens_u)
print(f"Optimal Bandwidths: {dens_u.bw}")
