import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def n_semanas(total_dias, dia):
    total_semanas = total_dias // 7 + 1
    semanas_transcurridas = dia // 7 + 1
    delta = total_semanas - semanas_transcurridas
    if delta == 0:
        delta = 1
    return delta


def cells_distance(x1, y1, x2, y2, hx, hy):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    d = 1 + math.floor(dx / hx) + math.floor(dy / hy)
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

def n_celdas_pintar(xi, yi, x, y, hx, hy):
    x_sum = math.floor(abs(xi - x)/hx)
    y_sum = math.floor(abs(yi-y)/hy)
    return 1 + x_sum + y_sum

def radio_pintar(ancho_celda, bw):
    return math.ceil(bw/ancho_celda)


def diamond2(r):
    return np.add.outer(*[np.r_[:r,r:-1:-1]]*2)>=r

def square_matrix(lado):
    return np.ones((lado, lado), dtype=bool)

def limites_x(ancho_pintura, punto, malla):
    izq = punto - ancho_pintura
    if izq <0:
        izq = 0

    der = punto + ancho_pintura
    range_malla = malla.shape[0]
    if der > range_malla:
        der = range_malla
    return izq, der


def limites_y(ancho_pintura, punto, malla):
    abajo = punto - ancho_pintura
    if abajo < 0:
        abajo = 0

    up = punto + ancho_pintura
    range_malla = malla.shape[1]
    if up > range_malla:
        up = range_malla

    return abajo, up

def grafico(x, y, name_x, name_y):
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title(name_x + ' VS ' + name_y)
    plt.plot(x, y)
    plt.show()

def calcular_celdas(hx, hy, superficie):
    superficie = round(superficie)
    hx = hx/1000
    hy = hy/1000
    raiz = math.sqrt(superficie)


    """
    :param hx: en metros
    :param hy: en metros
    :param superficie: en kilemtros
    :return: numero de celdas asociadas
    """
    return round((raiz / hx) * (raiz / hy))






if __name__ == '__main__':
    a, _ = np.mgrid[0:5, 0:5]
    print(calcular_celdas(100, 100, 4170))


