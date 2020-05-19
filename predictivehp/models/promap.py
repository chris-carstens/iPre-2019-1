"""PROMAP"""

import shutil

from models import parameters
import auxiliar_functions_promap as aux
from collections import defaultdict
from matplotlib.lines import Line2D
from processing.data_processing import *


# Observaciones
#
# 1. 3575 Incidents
# Training data 2926 incidents (January 1st - October 31st)
# Testing data 649 incidents (November 1st - December 31st)
#


def _time(fn):
    def inner_1(*args, **kwargs):
        start = time()

        fn(*args, **kwargs)

        print(f"\nFinished in {round(time() - start, 3)} sec")

    return inner_1


class Promap:
    """
    Class for a spatio-temporal using PromMap
    """

    def __init__(self, bw,
                 n: int = 1000,
                 year: str = "2017", read_files=True):
        """
        :param n: Número de registros que se piden a la database.

        :param year: Año de los registros pedidos

        :param bw: diccionario con las banwith calculadas previamente

        """

        self.data = None
        self.training_data = None  # 3000
        self.testing_data = None  # 600

        self.bw_x = bw[0]
        self.bw_y = bw[1]
        self.bw_t = bw[2]

        self.n = n
        self.year = year

        self.matriz_con_densidades = None
        self.HR = None
        self.PAI = None
        self.area_percentaje = None
        self.df = None
        self.df_testing_data = None
        self.df_training_data = None

        if read_files:
            self.df = pd.read_pickle('data.pkl')
            self.df_training_data = pd.read_pickle('training_data.pkl')
            self.df_testing_data = pd.read_pickle('testing_data.pkl')
            self.generar_df()
            self.matriz_con_densidades = np.load(
                'matriz_de_densidades.pkl.npy')

        else:
            self.data, self.training_data, self.testing_data = get_data(
                model='ProMap', n=150_000)
            self.generar_df()
            self.calcular_densidades()

    def generar_df(self):

        '''''
        Genera un dataframe en base a los x{min, max} y{min, max}.
        Recordar que cada nodo del dataframe representa el centro de cada 
        celda en la malla del mapa.
        '''''

        print("\nGenerando dataframe...\n")

        geometry = [Point(xy) for xy in zip(
            np.array(self.data[['x']]),
            np.array(self.data[['y']]))
                    ]

        self.geo_data = gpd.GeoDataFrame(self.data,  # gdf de incidentes
                                         crs=2276,
                                         geometry=geometry)

        self.geo_data.to_crs(epsg=3857, inplace=True)

        # Ahora debemos juntar los datos del geo data y los day_y

        data = defaultdict(list)

        for i in range(len(self.data)):
            data['x'].append(self.geo_data.geometry[i].x)
            data['y'].append(self.geo_data.geometry[i].y)
            data['date'].append(self.data['date'][i])
            data['y_day'].append(self.data['y_day'][i])

        data_ok = pd.DataFrame(data=data)

        self.df = data_ok

        # División en training y testing data

        self.df_training_data = self.df[
            self.data["date"].apply(lambda x: x.month) <= 10
            ]

        self.df_testing_data = self.df[
            self.data["date"].apply(lambda x: x.month) > 10
            ]

        self.x_min = parameters.dallas_limits['x_min']
        self.x_max = parameters.dallas_limits['x_max']
        self.y_min = parameters.dallas_limits['y_min']
        self.y_max = parameters.dallas_limits['y_max']

        self.hx = parameters.hx
        self.hy = parameters.hy

        self.bins_x = round(abs(self.x_max - self.x_min) / self.hx)
        self.bins_y = round(abs(self.y_max - self.y_min) / self.hy)

        print(f'\thx: {self.hx} mts, hy: {self.hy} mts')
        print(
            f'\tbw.x: {self.bw_x} mts, bw.y: {self.bw_y} mts, bw.t: {self.bw_t} dias')
        print(f'\tbins.x: {self.bins_x}, bins.y: {self.bins_y}\n')

        self.x, self.y = np.mgrid[self.x_min + self.hx / 2:self.x_max -
                                                           self.hx /
                                                           2:self.bins_x *
                                                             1j,
                         self.y_min + self.hy / 2:self.y_max - self.hy /
                                                  2:self.bins_y *
                                                    1j]

        self.total_dias_training = self.df_training_data['y_day'].max()

    def calcular_densidades(self):

        """""
        Calcula los scores de la malla en base a los delitos del self.data
        :return np.mgrid

        """""

        print('\nCalculando densidades...')
        print(
            f'\n\tNº de datos para entrenar el modelo: {len(self.df_training_data)}')
        print(f'\tNº de días usados para entrenar el modelo: '
              f'{self.total_dias_training}')
        print(
            f'\tNº de datos para testear el modelo: {len(self.df_testing_data)}')

        matriz_con_ceros = np.zeros((self.bins_x, self.bins_y))

        for k in range(len(self.df_training_data)):
            x, y, t = self.df_training_data['x'][k], \
                      self.df_training_data['y'][
                          k], \
                      self.df_training_data['y_day'][k]
            x_in_matrix, y_in_matrix = aux.find_position(self.x, self.y, x, y,
                                                         self.hx, self.hy)
            ancho_x = aux.radio_pintar(self.hx, self.bw_x)
            ancho_y = aux.radio_pintar(self.hx, self.bw_x)
            x_left, x_right = aux.limites_x(ancho_x, x_in_matrix, self.x)
            y_abajo, y_up = aux.limites_y(ancho_y, y_in_matrix, self.y)

            for i in range(x_left, x_right + 1):
                for j in range(y_abajo, y_up):
                    elemento_x = self.x[i][0]
                    elemento_y = self.y[0][j]
                    time_weight = 1 / aux.n_semanas(self.total_dias_training,
                                                    t)
                    if aux.linear_distance(elemento_x, x) > self.bw_x or \
                            aux.linear_distance(
                                elemento_y, y) > self.bw_y:

                        cell_weight = 0
                        pass
                    else:
                        cell_weight = 1 / aux.cells_distance(x, y, elemento_x,
                                                             elemento_y,
                                                             self.hx,
                                                             self.hy)

                    matriz_con_ceros[i][j] += time_weight * cell_weight

        print('\nGuardando datos...')
        np.save('matriz_de_densidades.pkl', matriz_con_ceros)

        self.matriz_con_densidades = matriz_con_ceros

    def heatmap(self, matriz, nombre_grafico):

        """
        Mostrar un heatmap de una matriz de riesgo.

        :param matriz: np.mgrid
        :return None
        """

        plt.title(nombre_grafico)
        plt.imshow(np.flipud(matriz.T),
                   extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        plt.colorbar()
        plt.show()

    def delitos_por_celda_training(self):

        delitos_agregados = 0

        """
        Calcula el nº de delitos que hay por cada celda en la matrix de
        training

        :param None
        :return None
        """

        self.training_matrix = np.zeros((self.bins_x, self.bins_y))
        for index, row in self.df_training_data.iterrows():
            x, y, t = row['x'], row['y'], row['y_day']

            if t >= (self.total_dias_training - self.bw_t):
                x_pos, y_pos = aux.find_position(self.x, self.y, x, y, self.hx,
                                                 self.hy)
                self.training_matrix[x_pos][y_pos] += 1
                delitos_agregados += 1

        print(f'\nSe han cargado: {delitos_agregados} delitos a la matriz de '
              f'training')

    def delitos_por_celda_testing(self, ventana_dias):

        delitos_agregados = 0

        """
        Calcula el nº de delitos que hay por cada celda en la matrix de
        testeo

        :param None
        :return: None
        """

        self.testing_matrix = np.zeros((self.bins_x, self.bins_y))
        for index, row in self.df_testing_data.iterrows():
            x, y, t = row['x'], row['y'], row['y_day']

            if t <= (self.total_dias_training + ventana_dias):

                x_pos, y_pos = aux.find_position(self.x, self.y, x, y, self.hx,
                                                 self.hy)
                self.testing_matrix[x_pos][y_pos] += 1
                delitos_agregados += 1


            else:
                print(f'Se han cargado: {delitos_agregados} delitos a la '
                      f'matriz de '
                      f'testeo')
                print(f'Se hará una predicción para los próximos: '
                      f'{ventana_dias} días')

                break

    def calcular_hr_and_pai(self):

        ventana_dias = 7
        self.delitos_por_celda_training()
        self.delitos_por_celda_testing(ventana_dias)

        nodos = self.matriz_con_densidades.flatten()

        k = np.linspace(0, nodos.max(), 400)

        """
        1. Solo considera las celdas que son mayor a un K
            Esto me entrega una matriz con True/False (Matriz A)
        2. La matriz de True/False la multiplico con una matriz que tiene la 
        cantidad de delitos que hay por cada celda (Matriz B)
        3. Al multiplicar A * B obtengo una matriz C donde todos los False en 
        A son 0 en B. Todos los True en A mantienen su valor en B
        4. Contar cuantos delitos quedaron luego de haber pasado este proceso.
        
        Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta
        """

        hits_n = []

        for i in range(k.size):
            hits_n.append(
                np.sum(
                    (self.matriz_con_densidades >= k[
                        i]) * self.testing_matrix))

        """
        1. Solo considera las celdas que son mayor a un K
            Esto me entrega una matriz con True/False (Matriz A)
        2. Contar cuantos celdas quedaron luego de haber pasado este proceso.
        
        Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta
        """

        area_hits = []

        for i in range(k.size):
            area_hits.append(
                np.count_nonzero((self.matriz_con_densidades >= k[
                    i]) * self.matriz_con_densidades
                                 ))

        n_delitos_testing = np.sum(self.testing_matrix)

        self.HR = [i / n_delitos_testing for i in hits_n]

        self.area_percentaje = [i / (100_000) for i in
                                area_hits]

        self.PAI = [
            0 if float(self.area_percentaje[i]) == 0 else float(self.HR[
                                                                    i]) / float(
                self.area_percentaje[i]) for i in range(len(self.HR))]

    def plot_HR(self):
        if self.HR is None:
            self.calcular_hr_and_pai()

        print('\n--- HITRATE ---\n')
        aux.grafico(self.area_percentaje, self.HR, '% Area', 'HR')

    def plot_PAI(self):
        if self.PAI is None:
            self.calcular_hr_and_pai()

        print('\n--- PAI ---\n')
        aux.grafico(self.area_percentaje, self.PAI, '% Area', 'PAI')

    def plot_delitos_meses(self):
        meses_training = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                          'Jul', 'Ago', 'Sept', 'Oct']

        meses_test = ['Nov', 'Dic']

        n_por_mes = []

        n_mes = 1
        n_datos = 0

        while n_mes <= 12:
            for x in self.data['date']:
                if x.month == n_mes:
                    n_datos += 1
            n_por_mes.append(n_datos)
            n_mes += 1
            n_datos = 0

        datos_training = n_por_mes[0:10]
        datos_test = n_por_mes[10:12]

        plt.bar(meses_training, datos_training, align='center')
        plt.bar(meses_test, datos_test, color='g', align='center')
        plt.title('Distribución de delitos por mes')
        plt.legend(['Meses de training', 'Meses de Test'])
        plt.show()

    def tabla_de_datos(self, k, hits_n, n_delitos_testing, HR,
                       area_percentaje, PAI, area_hits):
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

        d = {'k': k,
             'numero de hits identificados (n)': hits_n,
             'hits totales (N)': n_delitos_testing,
             'HR': HR,
             'celdas de area (a)': area_hits,
             'area total (A)': 100_000,
             'porcentaje de area': area_percentaje,
             'PAI': PAI}

        tabla = pd.DataFrame(data=d)

        tabla_string = tabla.to_string()

        df_split = tabla_string.split('\n')

        # Get maximum value of a single column 'y'
        maxPAI = tabla['PAI'].idxmax()
        max_pai = tabla['PAI'].max()

        columns = shutil.get_terminal_size().columns

        print(f'\nMáximo valor de PAI: {max_pai}')
        print('\n')
        print(df_split[0].center(columns))
        print(df_split[maxPAI].center(columns))
        print("\033[94m" + df_split[maxPAI + 1].center(columns) + "\033[0m")
        print(df_split[maxPAI + 2].center(columns))

        # blue '\033[94m'

        print('\n\n')

        for i in range(len(tabla)):
            if i == maxPAI + 1:
                print("\033[94m" + df_split[i].center(
                    columns) + "\033[0m")
            else:
                print(df_split[i].center(columns))

    def plot_incidents(self, i_type="real", month="October"):
        """
        Plotea los incidentes almacenados en self.data en el mes dado.
        Asegurarse que al momento de realizar el ploteo, ya se haya
        hecho un llamado al método ml_algorithm() para identificar los
        incidentes TP y FN

        :param str i_type: Tipo de incidente a plotear (e.g. TP, FN, TP & FN)
        :param str month: String con el nombre del mes que se predijo
            con ml_algorithm()
        :return:
        """

        print(f"\nPlotting {month} Incidents...")
        print("\tFiltering incidents...")

        tp_data, fn_data, data = None, None, None

        # if i_type == "TP & FN":
        #     data = gpd.GeoDataFrame(self.df)
        #     tp_data = data[self.df.TP == 1]
        #     fn_data = data[self.df.FN == 1]
        # if i_type == "TP":
        #     data = gpd.GeoDataFrame(self.df)
        #     tp_data = self.df[self.df.TP == 1]
        # if i_type == "FN":
        #     data = gpd.GeoDataFrame(self.df)
        #     fn_data = self.df[self.df.FN == 1]
        if i_type == "real":
            data = self.data[self.data.month1 == month]
            n_incidents = data.shape[0]
            print(f"\tNumber of Incidents in {month}: {n_incidents}")
        # if i_type == "pred":
        #     data = gpd.GeoDataFrame(self.df)
        #     all_hp = data[self.df[('Dangerous_pred_Oct', '')] == 1]

        print("\tReading shapefile...")
        d_streets = gpd.GeoDataFrame.from_file(
            "../Data/Streets/streets.shp")
        d_streets.to_crs(epsg=2276, inplace=True)

        print("\tRendering Plot...")
        fig, ax = plt.subplots(figsize=(20, 15))

        d_streets.plot(ax=ax,
                       alpha=0.4,
                       color="dimgrey",
                       zorder=2,
                       label="Streets")

        # if i_type == 'pred':
        #     all_hp.plot(
        #         ax=ax,
        #         markersize=2.5,
        #         color='y',
        #         marker='o',
        #         zorder=3,
        #         label="TP Incidents"
        #     )
        if i_type == "real":
            data.plot(
                ax=ax,
                markersize=10,
                color='darkorange',
                marker='o',
                zorder=3,
                label="TP Incidents"
            )
        # if i_type == "TP":
        #     tp_data.plot(
        #         ax=ax,
        #         markersize=2.5,
        #         color='red',
        #         marker='o',
        #         zorder=3,
        #         label="TP Incidents"
        #     )
        # if i_type == "FN":
        #     fn_data.plot(
        #         ax=ax,
        #         markersize=2.5,
        #         color='blue',
        #         marker='o',
        #         zorder=3,
        #         label="FN Incidents"
        #     )
        # if i_type == "TP & FN":
        #     tp_data.plot(
        #         ax=ax,
        #         markersize=2.5,
        #         color='red',
        #         marker='o',
        #         zorder=3,
        #         label="TP Incidents"
        #     )
        #     fn_data.plot(
        #         ax=ax,
        #         markersize=2.5,
        #         color='blue',
        #         marker='o',
        #         zorder=3,
        #         label="FN Incidents"
        #     )

        # Legends
        handles = [Line2D([], [],
                          marker='o',
                          color='darkorange',
                          label='Incident',
                          linestyle='None'),
                   Line2D([], [],
                          marker='o',
                          color='red',
                          label='TP Incident',
                          linestyle='None'),
                   Line2D([], [],
                          marker='o',
                          color="blue",
                          label="FN Incident",
                          linestyle='None'),
                   Line2D([], [],
                          marker='o',
                          color='y',
                          label='Predicted Incidents',
                          linestyle='None')]

        plt.legend(loc="best",
                   bbox_to_anchor=(0.1, 0.7),
                   frameon=False,
                   fontsize=13.5,
                   handles=handles)

        legends = ax.get_legend()
        for text in legends.get_texts():
            text.set_color('white')

        # Background
        ax.set_axis_off()
        fig.set_facecolor('black')
        plt.show()
        plt.close()


if __name__ == "__main__":
    st = time()
    promap = Promap(n=150_000, year="2017", bw=parameters.bw, read_files=False)
    # promap.plot_incidents()
