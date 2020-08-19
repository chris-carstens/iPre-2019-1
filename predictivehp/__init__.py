import matplotlib as mpl

from .utils.seed import set_seed
from ._config import rc

# Para reproducibilidad seteamos una semilla
set_seed(seed=0)

# Actualizamos parámetros de runtime context de Matplotlib
mpl.rcParams.update(rc)
