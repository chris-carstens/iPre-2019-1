import random
import numpy as np


def set_seed(seed=0):
    """
    Setea la semilla para hacer las corridas reproducibles

    Parameters
    ----------
    seed : int

    Returns
    -------

    """
    random.seed(seed)
    np.random.seed(seed)
