import numpy as np

from scipy.signal import convolve, convolve2d

D = np.random.randint(low=0, high=10, size=(3, 3), dtype=int)


def li_neighbors(matrix, i=1):
    """
    Calcula la cantidad de incidentes en la i-ésima capa (capa tipo
    ProMap).

    :type matrix: np.ndarray
    :param matrix:
    :param i: int que indica la i-ésima capa a calcular
    :return: ndarray con la suma de incidentes de la i-ésima capa en
    cada celda
    """

    ker = np.ones(shape=matrix.shape, dtype=int)

    return convolve2d(matrix, ker, mode='same')


print(D, li_neighbors(D) - D, sep='\n' * 2)
