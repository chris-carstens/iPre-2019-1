import numpy as np
from math import ceil

from scipy.signal import convolve, convolve2d

D = np.random.randint(low=0, high=10, size=(10, 10), dtype=int)


def il_neighbors(matrix, i=1):
    """
    Calcula la cantidad de incidentes en la i-ésima capa (tipo
    ProMap) para cada una de las celdas en el arreglo matrix.

    :type matrix: np.ndarray
    :param matrix: ndarray con la cantidad de incidentes ocurridos en
        cada celda de la malla
    :param i: int que indica la i-ésima considerada
    :return: ndarray con la suma
        cada celda
    """

    ker1 = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )

    ker2 = np.array(
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0]]
    )

    ker3 = np.array(
        [[0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0]]
    )

    kernels = [ker1, ker2, ker3]

    return convolve2d(matrix, kernels[i - 1], mode='same')


if __name__ == '__main__':
    # print(D, il_neighbors(matrix=D, i=3), sep='\n' * 2)

    t = np.zeros((3, 3), dtype=int)
    l = np.eye(*t.shape, k=t.shape[0] // 2, dtype=int) + \
        np.eye(*t.shape, k=(t.shape[0] // 2) * -1, dtype=int)
    l = l + np.fliplr(l)

    print(l)
