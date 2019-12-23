import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from auxiliar_functions import find_position

total_dias = 30
bins = 5
bw_x = 6  # metros
bw_y = 6

x_min = 0
x_max = 10
y_min = 0
y_max = 10

hx = (x_max - x_min) / (bins)  # metros
hy = (y_max - y_min) / (bins)  # metros

d = {'x': [1, 3], 'y': [1, 3], 't': [1, 15]}
df = pd.DataFrame(data=d)

print(df['t'][0])

xx, yy = np.mgrid[x_min + hx / 2:x_max - hx / 2:bins * 1j,
         y_min + hy / 2:y_max - hy / 2:bins * 1j]

matriz_con_ceros = np.zeros((bins, bins))


def n_semanas(total_dias, dia):
    total_semanas = total_dias // 7 + 1
    semanas_transcurridas = dia // 7 + 1
    return total_semanas - semanas_transcurridas


def cells_distance(x1, y1, x2, y2):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    d = 1 + 2 * (math.floor(x / hx) + math.floor(y / hy))
    #print('CENTROIDE DISTANCE: ', d)
    return d


def linear_distance(a1, a2):
    linear_distance = abs(a1-a2)
    #print('DISTANCIA LINEAL: ', linear_distance)
    return float(linear_distance)




print('\n   ProMap   \n')
print(f'hx de {hx} metros\n')
print(f'hy de {hy} metros\n')
print(f'bandwith X: {bw_x} metros\nbandwith Y: {bw_y} metros')
print('\nBASE DE DATOS\n', df)
print()
print('X \n', xx)
print()
print('Y\n', yy)
print()

print('Calculando densidades...\n')

for k in range(len(df)):
    x, y, t = df['x'][k], df['y'][k], df['t'][k]
    for i in range(bins):
        for j in range(bins):
            elemento_x = xx[i][0]
            elemento_y = yy[0][j]
            #print(f'Punto Actual x: {elemento_x}, y: {elemento_y}')
            time_weight = 1 / n_semanas(total_dias, t)
            if linear_distance(elemento_x, x) > bw_x or linear_distance(
                    elemento_y, y) > bw_y:
            #    print('Esta celda está fuera del limite del ancho de banda')
                cell_weight = 0
                pass
            else:
                cell_weight = 1 / cells_distance(x, y, elemento_x, elemento_y)
            #print('CELL WIGHT: ', cell_weight)
            #print()
            matriz_con_ceros[i][j] += time_weight * cell_weight

print()
print(matriz_con_ceros)

#heatmap = plt.pcolormesh(xx, yy, matriz_con_ceros)
#plt.colorbar()
#plt.show()
# plt.imshow(np.flipud(matriz_con_ceros.T), extent=[x_min, x_max, y_min, y_max])
# plt.colorbar()
# plt.show()

training_data = {'x': [1, 3], 'y': [1, 3], 't': [17, 19]}
df_2 = pd.DataFrame(data=training_data)


training_matrix = np.full((bins, bins), False)

for i in range(len(df_2)):
    x, y = df_2['x'][i], df_2['y'][i]
    x_pos, y_pos = find_position(xx, yy, x, y, hx, hy)
    training_matrix[x_pos][y_pos] = True

print(training_matrix)

nodos = matriz_con_ceros.flatten()

k = np.linspace(0, nodos.max(), 50)

hits = []

for i in range(k.size):
    hits.append(np.sum((matriz_con_ceros >= k[i])*training_matrix))

area_h = []

for i in range(k.size):
    area_h.append(np.sum((matriz_con_ceros >= k[i])))



HR = [i / len(training_matrix) for i in hits]
area_percentaje = [i / len(matriz_con_ceros) for i in area_h]
PAI = [float(HR[i]) / float(area_percentaje[i]) for i in range(len(HR))]

print('HIT RATE \n')
print(HR)
print('AREA \n')
print(area_percentaje)
print('PAI \n')
print(PAI)







#hits = [np.sum(f_delitos >= k[i]) for i in range(k.size)]
# area_h = [np.sum(f_nodos >= c[i]) for i in range(c.size)]

def plot_HR():
    #results = self.calculate_HR_PAI()['HR']
    plt.xlabel('Area percentage')
    plt.ylabel('HR')
    plt.title("HR vs Area")
    plt.plot(area_percentaje, HR)
    plt.legend()
    plt.show()
    #plt.savefig("HRvsArea")


def plot_PAI():
    #results = self.calculate_HR_PAI()['PAI']
    plt.xlabel('Area percentage')
    plt.ylabel('PAI')
    plt.title("PAI vs Area")
    plt.plot(area_percentaje, PAI)
    plt.legend()
    plt.show()
    #plt.savefig("PAIvsArea")

print(area_percentaje)
print(HR)
print(PAI)

plot_HR()
plot_PAI()



