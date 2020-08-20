"""_cmaps.py

Establece ciertas configuraciones sobre algunos cmaps de matplotlib
"""

# Author: Mauro Mendoza <msmendoza@uc.cl>

import matplotlib.colors as colors
import numpy as np


def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Trunca el colormap dado entre los valores dados.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
    minval : float
    maxval : float
    n : int
      Número de colores a extraer del cmap

    Returns
    -------
    matplotlib.colors.Colormap
    """
    cmap_name = f'trunc({cmap.name},{minval:.2f},{maxval:.2f})'
    new_cmap = colors.LinearSegmentedColormap.from_list(
        cmap_name, cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
