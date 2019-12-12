import numpy as np

from scipy.signal import convolve2d


def diamond(dim=3):
    """
    Entrega la matriz diamante con 1s en el límite y 0s en su interior.

    :param dim: Dimensión del diamante, número impar mayor a 3
    :return: ndarray con la matriz diamate
    """

    t = np.zeros((dim, dim), dtype=int)

    l = np.eye(*t.shape, k=t.shape[0] // 2, dtype=int) + \
        np.eye(*t.shape, k=(t.shape[0] // 2) * -1, dtype=int)
    l = l + np.fliplr(l)

    # Corrección de los 2s

    l[0, t.shape[0] // 2] = 1
    l[t.shape[0] // 2, 0] = 1
    l[t.shape[0] // 2, t.shape[0] - 1] = 1
    l[t.shape[0] - 1, t.shape[0] // 2] = 1

    return l


def il_neighbors(matrix, i=1):
    """
    Calcula la cantidad de incidentes en la i-ésima capa (tipo ProMap)
    para cada una de las celdas en el arreglo matrix.

    :type matrix: np.ndarray
    :param matrix: ndarray con la cantidad de incidentes ocurridos en
        cada celda de la malla
    :param i: int que indica la i-ésima considerada
    :return: ndarray con la suma
        cada celda
    """

    kernel = diamond(2*i + 1)

    return convolve2d(matrix, kernel, mode='same')


if __name__ == '__main__':
    D = np.random.randint(low=0, high=10, size=(10, 10), dtype=int)

    print(D, il_neighbors(matrix=D, i=1), sep='\n' * 2)
