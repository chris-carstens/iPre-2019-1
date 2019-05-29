# KDE_1:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-05-14


from statsmodels.nonparametric.kernel_density import KDEMultivariate

import numpy as np

nobs = 300
np.random.seed(1234)  # Seed random generator

c1 = np.random.normal(size=(nobs, 1))
c2 = np.random.normal(2, 1, size=(nobs, 1))

dens_u = KDEMultivariate(data=[c1, c2],
                         var_type='cc',
                         bw='cv_ml')

# Para obtener los bandwidths de acuerdo al método: likelihood
# cross-validation

print("hi", flush=False)
print(dens_u)
print(f"Optimal Bandwidths: {dens_u.bw}")
