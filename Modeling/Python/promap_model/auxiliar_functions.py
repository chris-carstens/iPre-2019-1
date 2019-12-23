import math
import numpy as np


def n_semanas(total_dias, dia):
    total_semanas = total_dias // 7 + 1
    semanas_transcurridas = dia // 7 + 1
    delta = total_semanas - semanas_transcurridas
    if delta == 0:
        delta = 1
    return delta


def cells_distance(x1, y1, x2, y2, hx, hy):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    d = 1 + 2 * (math.floor(x / hx) + math.floor(y / hy))
    #print('CENTROIDE DISTANCE: ', d)
    return d


def linear_distance(a1, a2):
    linear_distance = abs(a1 - a2)
    #print('DISTANCIA LINEAL: ', linear_distance)
    return float(linear_distance)

def find_position(mgridx, mgridy, x, y, hx, hy):
    x_desplazada = mgridx - hx/2
    y_desplazada = mgridy - hy/2
    pos_x = np.where(x_desplazada <= x)[0][-1]
    pos_y = np.where(y_desplazada <= y)[1][-1]
    return pos_x, pos_y


