import numpy as np

from scipy.signal import convolve, convolve2d

D = np.random.randint(low=0, high=10, size=(10, 10), dtype=int)


def li_neighbors(matrix, i=1):
    """
    Calcula la cantidad de incidentes en las celdas vecinas de la
    i-ésima capa

    @param matrix:
    @param i:
    @return:
    """

    ker = np.ones(shape=matrix.shap)

    return convolve2d()
