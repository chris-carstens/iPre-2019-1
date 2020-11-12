from calendar import month_name
from datetime import date, timedelta, datetime
from functools import reduce

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.nonparametric.kernel_density as kd
from matplotlib.lines import Line2D
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor

import predictivehp.utils._aux_functions as af
from predictivehp import d_colors

settings = kd.EstimatorSettings(efficient=True, n_jobs=8)
pd.set_option('mode.chained_assignment', None)


class MyKDEMultivariate(kd.KDEMultivariate):
    def resample(self, size, shp):
        """

        Parameters
        ----------
        size : int

        Returns
        -------
        np.hstack

        """
        # print("\nResampling...", end=" ")

        n, d = self.data.shape
        indices = np.random.randint(0, n, size)

        cov = np.diag(self.bw) ** 2
        means = self.data[indices, :]
        norm = np.random.multivariate_normal(np.zeros(d), cov, size)

        # simulated and checked points
        s_points = np.transpose(means + norm)
        c_points = af.checked_points(s_points, shp)

        # print(f"\n{size - c_points.shape[1]} invalid points found")

        if size == c_points.shape[1]:
            # print("\nfinished!")
            return s_points

        a_points = self.resample(size - c_points.shape[1], shp)

        return np.hstack((c_points, a_points))


class STKDE:
    def __init__(self, data=None,
                 shps=None, bw=None, sample_number=3600,
                 start_prediction=date(2017, 11, 1),
                 length_prediction=7, name="STKDE"):
        """
        Parameters
        ----------
        bw: np.array
          bandwidth for x, y, t
        sample_number: int
          Número de muestras de la base de datos
        start_prediction : date
          Fecha de comienzo en la ventana temporal a predecir
        length_prediction : int
          Número de días de ventana a predecir
        name : str
          Nombre predeterminado del modelo
        shps : gpd.GeoDataFrame
          GeoDataFrame que contiene información sobre la ciudad de Dallas
        """

        self.hits = None
        self.name, self.sn, self.bw = name, sample_number, bw
        self.shps = shps
        self.start_prediction = start_prediction
        self.lp = length_prediction

        self.hr, self.ap, self.pai = None, None, None
        self.f_delitos, self.f_nodos = None, None
        self.df = None
        self.f_max = None
        self.data = data
        if self.shps is not None:
            self.x_min, self.y_min, self.x_max, self.y_max = self.shps[
                'streets'].total_bounds
        else:
            delta_x = 0.1 * self.data.x.mean()
            delta_y = 0.1 * self.data.y.mean()
            self.x_min = self.data.x.min() - delta_x
            self.x_max = self.data.x.max() + delta_x
            self.y_min = self.data.y.min() - delta_y
            self.y_max = self.data.y.max() + delta_y
        # training data 3000
        # testing data  600
        # print('-' * 30)
        # print('\t\tSTKDE')
        # print(af.print_mes(self.X_months, self.X_months + 1, self.wd))

        # print('-' * 30)

    def set_parameters(self, bw):
        """

        Parameters
        ----------
        bw: np.array
            Bandwith for x,y,t

        Returns
        -------

        """
        self.bw = bw
        # Reentrenamos el modelo con nuevo bw
        if self.df is not None:
            self.fit(self.X_train, self.X_test)

    def print_parameters(self):
        """

        Returns
        -------

        """
        print('STKDE Hyperparameters')
        if self.bw is not None:
            print(f'bandwith x: {self.bw[0]} mts.')
            print(f'bandwith y: {self.bw[1]} mts.')
            print(f'bandwith t: {self.bw[2]} days\n')
        else:
            print(
                "No bandwith set. The model will automatically calculate bandwith after fit.\n")
        print()

    def fit(self, X, X_t, verbose=False):
        """
        Parameters
        ----------
        X : pd.DataFrame
          Training data.
        X_t : pd.DataFrame
          Test data.
        """
        print("\tFitting Model...") if verbose else None
        self.X_train, self.X_test = X, X_t

        self.kde = MyKDEMultivariate(
            [np.array(self.X_train[['x']]),
             np.array(self.X_train[['y']]),
             np.array(self.X_train[['y_day']])],
            'ccc', bw=self.bw)

        self.bw = self.kde.bw

    def predict(self, verbose=False):
        """
        Returns
        -------
        f_delitos_by_group : dict
          Diccionario con los valores de la función densidad evaluada en
          los puntos a predecir con llave el grupo
        f_nodos_by_group : dict
          Diccionario con los valores de la función densidad evaluada en
          la malla original con llave el grupo
        """

        print("\tMaking predictions...") if verbose else None
        if self.f_delitos is not None:
            return self.f_delitos, self.f_nodos

        stkde = self.kde
        t_training = pd.Series(self.X_train["y_day"]).to_numpy()
        if self.shps is not None:
            self.predicted_sim = stkde.resample(len(pd.Series(
                self.X_test["x"]).tolist()), self.shps["councils"])
        # noinspection PyArgumentList
        x, y, t = np.mgrid[
                  self.x_min:
                  self.x_max:100 * 1j,
                  self.y_min:
                  self.y_max:100 * 1j,
                  t_training.max():
                  t_training.max():1 * 1j
                  ]

        # pdf para nodos. checked_points filtra que los puntos estén dentro del área de dallas
        if self.shps is not None:
            f_nodos = stkde.pdf(af.checked_points(
                np.array([x.flatten(), y.flatten(), t.flatten()]),
                self.shps['councils']))
        else:
            f_nodos = stkde.pdf(
                np.array([x.flatten(), y.flatten(), t.flatten()]))

        x, y, t = \
            np.array(self.X_test['x']), \
            np.array(self.X_test['y']), \
            np.array(self.X_test['y_day'])
        ti = np.repeat(max(t_training), x.size)
        f_delitos = stkde.pdf(
            np.array([x.flatten(), y.flatten(), ti.flatten()]))

        f_max = max([f_nodos.max(), f_delitos.max()])

        # normalizar
        f_delitos = f_delitos / f_max
        f_nodos = f_nodos / f_max

        self.f_max = f_max

        self.f_delitos, self.f_nodos = f_delitos, f_nodos
        return self.f_delitos, self.f_nodos

    def score(self, x, y, t):
        """

        Parameters
        ----------
        x : float
        y : float
        t : float

        Returns
        -------
        score_pdf : float
                    Valor de la función densidad de
                    la predicción evaluada en (x,y,t)
        """
        if self.f_max is None:
            self.predict()
        score_pdf = self.kde.pdf(np.array([x, y, t])) / self.f_max
        return score_pdf

    def plot_geopdf(self, x_t, y_t, X_filtered, dallas, ax, color, label):
        if X_filtered.size > 0:

            geometry = [Point(xy) for xy in zip(x_t, y_t)]
            if dallas:
                geo_df = gpd.GeoDataFrame(X_filtered,
                                          crs=dallas.crs,
                                          geometry=geometry)
            else:
                geo_df = gpd.GeoDataFrame(X_filtered,
                                          geometry=geometry)
            geo_df.plot(ax=ax,
                        markersize=3,
                        color=color,
                        marker='o',
                        zorder=3,
                        label=label,
                        )
            plt.legend()

    def heatmap(self, c=None, show_score=True, incidences=False,
                savefig=False, fname='STKDE_heatmap.png', ap=None,
                verbose=False, **kwargs):
        """
        Parameters
        ----------
        bins : int
          Número de bins para heatmap
        ti : int
          Tiempo fijo para evaluar densidad en la predicción
        """

        print('\tPlotting Heatmap...') if verbose else None
        if self.shps is not None:
            dallas = self.shps['streets']
        else:
            dallas = None
        fig, ax = plt.subplots(figsize=[6.75] * 2)  # Sacar de _config.py
        t_training = pd.Series(self.X_train["y_day"]).to_numpy()

        x, y, t = np.mgrid[
                  self.x_min:
                  self.x_max:100 * 1j,
                  self.y_min:
                  self.y_max:100 * 1j,
                  t_training.max():
                  t_training.max():1 * 1j
                  ]
        x = x.reshape(100, 100)
        y = y.reshape(100, 100)
        t = t.reshape(100, 100)

        z = self.kde.pdf(
            np.array([x.flatten(), y.flatten(), t.flatten()]))
        if self.shps is not None:
            z_filtered = self.kde.pdf(af.checked_points(
                np.array([x.flatten(), y.flatten(), t.flatten()]),
                self.shps['councils']))
        else:
            z_filtered = self.kde.pdf(
                np.array([x.flatten(), y.flatten(), t.flatten()]))

        x_t, y_t, t_t = \
            np.array(self.X_test['x']), \
            np.array(self.X_test['y']), \
            np.array(self.X_test['y_day'])

        ti = np.repeat(max(t_training), x_t.size)

        f_delitos = self.kde.pdf(
            np.array([x_t.flatten(), y_t.flatten(), ti.flatten()]))
        max_pdf = max([f_delitos.max(), z.max()])

        # Normalize
        f_delitos = f_delitos / max_pdf
        z = z / max_pdf
        z_filtered = z_filtered / max_pdf

        if type(ap) == float or type(ap) == np.float64:
            c_array = self.c_vector
            area_h = [np.sum(z_filtered >= c_array[i]) for i in range(
                c_array.size)]
            area_percentaje = [i / len(z_filtered) for i in area_h]
            c = float(af.find_c(area_percentaje, c_array, ap))
            print('c value: ', c) if verbose else None

        elif type(ap) == list or type(ap) == np.ndarray:
            c_array = self.c_vector
            area_h = [np.sum(z_filtered >= c_array[i]) for i in range(
                c_array.size)]
            area_percentaje = [i / len(z_filtered) for i in area_h]
            c = [af.find_c(area_percentaje, c_array, i) for i in ap]
            c = sorted(list(set(c)))

            print('c values: ', c) if verbose else None

            if len(c) == 1:
                c = c[0]

        z_plot = None
        if c is None:
            z_plot = z
        elif type(c) == float:
            z_plot = z >= c
        elif type(c) == list or type(c) == np.ndarray:
            c = np.array(c).flatten()
            c = c[c > 0]
            c = c[c < 1]
            c = np.unique(c)
            c = np.sort(c)
            z_plot = np.zeros(z.size)
            for i in c:
                z_plot[z > i] += 1
            z_plot = z_plot / np.max(z_plot)

        if show_score and c is None:
            # noinspection PyUnresolvedReferences
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            # noinspection PyUnresolvedReferences
            cmap = mpl.cm.jet
            # noinspection PyUnresolvedReferences
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            c_bar = fig.colorbar(mappable, ax=ax,
                                 fraction=0.15,
                                 shrink=0.5,
                                 aspect=21.5)
            c_bar.ax.set_ylabel('Danger Score')

        if incidences:
            if dallas is not None:
                dallas.plot(ax=ax, alpha=.2, color="gray", zorder=1)
            plt.pcolormesh(x, y, z_plot.reshape(x.shape),
                           shading='gouraud',
                           alpha=.2,
                           zorder=2,
                           cmap="jet",
                           vmin=0,
                           vmax=1
                           )

            if c is None:
                self.plot_geopdf(f_delitos, np.array(self.X_test[['x']]),
                                 np.array(self.X_test[['y']]), self.X_test,
                                 dallas, ax, "red")
            elif type(c) == float:
                hits_bool = f_delitos >= c
                no_hits_bool = f_delitos < c
                self.plot_geopdf(np.array(self.X_test[['x']])[no_hits_bool],
                                 np.array(self.X_test[['y']])[no_hits_bool],
                                 self.X_test[no_hits_bool],
                                 dallas, ax, "red", "Misses")
                self.plot_geopdf(np.array(self.X_test[['x']])[hits_bool],
                                 np.array(self.X_test[['y']])[hits_bool],
                                 self.X_test[hits_bool],
                                 dallas, ax, "lime", "Hits")
            elif type(c) == list or type(c) == np.ndarray:
                c = np.array(c).flatten()
                c = c[c > 0]
                c = c[c < 1]
                c = np.unique(c)
                c = np.sort(c)
                i = 0
                for c_i in c:
                    if i == 0:
                        lvl = f_delitos <= c_i
                    elif i < c.size - 1:
                        lvl = (f_delitos <= c_i) & (f_delitos > c[i - 1])
                    else:
                        lvl = (f_delitos <= c_i) & (f_delitos > c[i - 1])
                        i += 1
                        self.plot_geopdf(np.array(self.X_test[['x']])[lvl],
                                         np.array(self.X_test[['y']])[lvl],
                                         self.X_test[lvl],
                                         dallas, ax, kwargs['colors'][i],
                                         f"Level {i}")
                        lvl = f_delitos > c_i
                    i += 1
                    self.plot_geopdf(np.array(self.X_test[['x']])[lvl],
                                     np.array(self.X_test[['y']])[lvl],
                                     self.X_test[lvl],
                                     dallas, ax, kwargs['colors'][i],
                                     f"Level {i}")

        else:
            if dallas is not None:
                dallas.plot(ax=ax, alpha=.2, color="gray", zorder=2)
            plt.pcolormesh(x, y, z_plot.reshape(x.shape),
                           shading='gouraud',
                           zorder=1,
                           cmap="jet",
                           alpha=1
                           )

        plt.title('STKDE')

        # ax.set_axis_off()
        plt.tight_layout()
        if savefig:
            plt.savefig(fname, **kwargs)
        plt.show()

    def calculate_hr(self, c=None):
        """
        Parameters
        ----------
        c : np.linspace
            Threshold de confianza para
            filtrar hotspots
        Returns
        -------
        hr_by_group: list
                    Lista con los valores del HR
                    para cada grupo
        ap_by_group: list
                    Lista con los valores del
                    Area Percentage para cada grupo
        """
        if c is None:
            c = np.linspace(0, 1, 1000)
        self.c_vector = c
        if self.f_delitos is None:
            self.predict()
        f_delitos, f_nodos = self.f_delitos, self.f_nodos
        hits = [np.sum(f_delitos >= c[i]) for i in range(c.size)]
        area_h = [np.sum(f_nodos >= c[i]) for i in range(c.size)]
        HR = [i / len(f_delitos) for i in hits]
        area_percentaje = [i / len(f_nodos) for i in area_h]
        self.hr, self.ap = HR, area_percentaje

    def calculate_pai(self, c=None):
        """
        Parameters
        ----------
        c : {float, list, np.linspace}
            Threshold de confianza para filtrar hotspots
        ap : {float, list, np.linspace}

        Returns
        -------
        pai_by_group : list
                    Lista con los valores del PAI
                    para cada grupo
        hr_by_group: list
                    Lista con los valores del HR
                    para cada grupo
        ap_by_group: list
                    Lista con los valores del
                    Area Percentage para cada grupo
        """
        if c is None:
            c = np.linspace(0, 1, 1000)

        self.c_vector = c

        if not self.hr:
            self.calculate_hr(c)
        PAI = [float(self.hr[i]) / float(self.ap[i]) if
               self.ap[i] else 0 for i in range(len(self.hr))]
        self.pai = PAI

    def validate(self, c=None, ap=None, verbose=False, area=1000):
        """
        Si inrego asp, solo calcula PAI y HR, si ingreso c, calculo
        Parameters
        ----------
        c
        area
        ap

        Returns
        -------

        """
        self.calculate_pai()

        if ap is not None:
            if type(ap) == list or type(ap) == np.ndarray:
                c = []
                for ap_i in ap:
                    c.append(af.find_c(self.ap, self.c_vector, ap_i))
            elif type(ap) == float or type(ap) == np.float64:
                c = af.find_c(self.ap, self.c_vector, ap)
        if type(c) == float or type(c) == np.float64:
            hits = self.f_delitos >= c
            h_nodos = self.f_nodos >= c
            self.hr_validated = np.sum(hits) / len(self.f_delitos)
            self.pai_validated = self.hr_validated / (
                    np.sum(h_nodos) / len(self.f_nodos))
        elif type(c) == list or type(c) == np.ndarray:
            hits = self.f_delitos < max(c)
            hits = hits > min(c)
            h_nodos = self.f_nodos < max(c)
            h_nodos = h_nodos > min(c)
            self.hr_validated = hits.size / len(self.f_delitos)
            self.pai_validated = self.hr_validated / (
                    np.sum(h_nodos) / len(self.f_nodos))

        #  if ap is not None:
        #     self.h_area = (self.hr_validated / self.pai_validated) * area

        # elif ap is None:
        dx = (self.x_max - self.x_min) / 100
        dy = (self.y_max - self.y_min) / 100
        v = self.f_nodos > c
        self.h_area = np.sum(v) * dx * dy / (10 ** 6)
        self.d_incidents = np.sum(hits)
        print("Total: ", len(self.f_delitos)) if verbose else None
        print("Hotspot area area:", self.h_area) if verbose else None
        print("Incidents detected:", self.d_incidents) if verbose else None
        print("Hit rate validated: :", self.hr_validated) if verbose else None
        print("PAI validated:", self.pai_validated) if verbose else None


class RForestRegressor(object):
    def __init__(self, data_0=None, shps=None,
                 xc_size=100, yc_size=100, n_layers=7,
                 t_history=4, start_prediction=date(2017, 11, 1),
                 length_prediction=7,
                 read_data=False, w_data=False, read_X=False, w_X=False,
                 verbose=False,
                 name='RForestRegressor'):
        """ Regressor modificado de Scipy usado para predecir delitos.

        Parameters
        ----------
        data_0 : pd.DataFrame
          Corresponde a los datos extraídos en primera instancia desde
          la Socrata API.
        shps : gpd.GeoDataFrame
          GeoDataFrame que contiene información sobre la ciudad de Dallas
        xc_size : int
          Tamaño de las celdas en x [metros]
        yc_size : int
          Tamaño de las celdas en y [metros]
        n_layers : int
          Cantidad de capas para contabilizar incidentes de celdas vecinas
        t_history : int
          Ventanas temporales consideradas 'hacia atrás' desde la fecha de
          término de entrenamiento, que serán usadas para entrenar el
          regressor
        start_prediction : date
          Fecha de comienzo en la ventana temporal a predecir
        read_data : bool
          True para leer el DataFrame con los incidentes de la Socrata
          de un archivo .pkl previamente guardado.
        read_X : bool
          True para leer el DataFrame con la información de las celdas
          en Dallas de un archivo .pkl previamente guardado.
        name : str
          Nombre especial para el regressor que aparece en los plots,
          estadísticas, etc.
        verbose : bool
          Indica si se printean las diferentes acciones del método.
          default False

        Returns
        -------
        RForestRegressor
        """
        self.name = name
        self.shps = shps
        self.xc_size, self.yc_size = xc_size, yc_size
        self.n_layers = n_layers
        self.nx, self.ny, self.hx, self.hy = [None] * 4
        self.t_history = t_history
        self.start_prediction = start_prediction
        self.length_pred = length_prediction
        self.weeks = []
        self.l_weights = None

        self.rfr = RandomForestRegressor(n_jobs=8)
        self.ap, self.hr, self.pai = [None] * 3

        start_prediction = self.start_prediction
        # current date, que corresponde al último día en la ventana
        # temporal de entrenamiento, desde donde se comenzarán a
        # armar los grupos de entrenamiento
        c_date = self.start_prediction - timedelta(days=1)
        for _ in range(self.t_history):
            c_date -= timedelta(days=7)
            self.weeks.append(c_date + timedelta(days=1))
        self.weeks.reverse()
        self.weeks.append(start_prediction)

        self.data = data_0
        self.X = None
        self.read_data, self.read_X = read_data, read_X
        self.w_data, self.w_X = w_data, w_X

        self.d_incidents = 0  # Detected incidents
        self.h_area = 0  # Hotspot area
        if self.read_X:
            self.X = pd.read_pickle('predictivehp/data/X.pkl')
        if self.read_data:
            self.data = pd.read_pickle('predictivehp/data/data.pkl')

    def set_parameters(self, t_history,
                       xc_size, yc_size, n_layers,
                       label_weights=None,
                       read_data=False, w_data=False,
                       read_X=False, w_X=False):
        """
        Setea los hiperparámetros del modelo

        Parameters
        ----------
        t_history : int
          Número de semanas hacia atrás a partir del 31-10-2017 que
          serán usadas para entrenar el modelo
        xc_size : int
          Tamaño en metros de las celdas en x de la malla
        yc_size : int
          Tamaño en metros de las celdas en y de la malla
        n_layers : int
          Número de capas para considerar en el conteo de delitos de
          celdas vecinas
        label_weights : np.ndarray
        read_data : bool
        read_X : bool
        w_data : bool
        w_X : bool
        """
        self.t_history = t_history
        self.xc_size = xc_size
        self.yc_size = yc_size
        self.n_layers = n_layers
        self.l_weights = label_weights
        self.read_data = read_data
        self.read_X = read_X
        self.w_data = w_data
        self.w_X = w_X

    def print_parameters(self):
        print('RFR Hyperparameters')
        print(f'{"Training History:":<20s}{self.t_history} weeks')
        print(f'{"xc_size:":<20s}{self.xc_size} m')
        print(f'{"yc_size:":<20s}{self.yc_size} m')
        print(f'{"n_layers:":<20s}{self.n_layers}')
        print(f'{"l_weights:":<20s}{self.l_weights}')
        print()

    def generate_data(self, verbose=False):
        """Prepara self.data a una estructura más propicia para el estudio

        Parameters
        ----------
        verbose : bool
          Indica si se printean las diferentes acciones del método.
          default False
        """
        geometry = [Point(xy) for xy in zip(np.array(self.data[['x']]),
                                            np.array(self.data[['y']]))]
        if self.shps is not None:
            self.data = gpd.GeoDataFrame(self.data, crs=2276,
                                         geometry=geometry)
            self.data.to_crs(epsg=3857, inplace=True)
        else:
            self.data = gpd.GeoDataFrame(self.data, geometry=geometry)
        self.data['Cell'] = None
        self.assign_cells()

    def generate_X(self, verbose=False):
        """
        La malla se genera de la esquina inf-izquierda a la esquina sup-derecha,
        partiendo con id = 0.

        n = i + j*n_x

        OBS.

        El numpy nd-array indexa de la esquina sup-izq a la inf-der
        [i:filas, j:columnas]

        El Pandas Dataframe comienza a indexar de la esquina inf-izq a la
        sup-der. [j, i]

        Hay que tener cuidado al momento de pasar desde la perspectiva
        matricial a la perspectiva normal de malla de ciudad, debido a las
        operaciones trasposición y luego up-down del nd-array entregan las
        posiciones reales para el pandas dataframe.

        Parameters
        ----------
        verbose : bool
          Indica si se printean las diferentes acciones del método.
          default False
        """
        print("\nGenerating dataframe...\n") \
            if verbose else None

        # Creación de la malla
        print("\tCreating mgrid...") if verbose else None
        if self.shps is not None:
            x_min, y_min, x_max, y_max = self.shps['streets'].total_bounds
        else:
            delta_x = 0.1 * self.data.x.mean()
            delta_y = 0.1 * self.data.y.mean()
            x_min = self.data.x.min() - delta_x
            x_max = self.data.x.max() + delta_x
            y_min = self.data.y.min() - delta_y
            y_max = self.data.y.max() + delta_y

        x_bins = abs(x_max - x_min) / self.xc_size
        y_bins = abs(y_max - y_min) / self.yc_size
        x, y = np.mgrid[x_min: x_max: x_bins * 1j, y_min: y_max: y_bins * 1j, ]

        # Creación del esqueleto del dataframe
        print("\tCreating dataframe columns...") if verbose else None

        X_cols = pd.MultiIndex.from_product(
            [[f"Incidents_{i}" for i in range(self.n_layers + 1)], self.weeks]
        )
        X = pd.DataFrame(columns=X_cols)

        # Creación de los parámetros para el cálculo de los índices
        print("\tFilling data...") if verbose else None
        self.nx = x.shape[0] - 1
        self.ny = y.shape[1] - 1
        self.hx = (x.max() - x.min()) / self.nx
        self.hy = (y.max() - y.min()) / self.ny

        # Nro. incidentes en la i-ésima capa de la celda (i, j)
        for week in self.weeks:
            print(f"\t\t{week}... ", end=' ') if verbose else None
            wi_date = week
            wf_date = week + timedelta(days=6)
            fil_incidents = self.data[
                (wi_date <= self.data.date) & (self.data.date <= wf_date)
                ]
            D = np.zeros((self.nx, self.ny), dtype=int)

            for _, row in fil_incidents.iterrows():
                xi, yi = row.geometry.x, row.geometry.y
                nx_i = af.n_i(xi, x.min(), self.hx)
                ny_i = af.n_i(yi, y.min(), self.hy)
                D[nx_i, ny_i] += 1

            # Actualización del pandas dataframe
            for i in range(self.n_layers + 1):
                X.loc[:, (f"Incidents_{i}", f"{week}")] = \
                    af.to_df_col(D) if i == 0 \
                        else af.to_df_col(af.il_neighbors(D, i))
            print('finished!') if verbose else None

        # Adición de las columnas 'geometry' e 'in_dallas' al data
        print("\tPreparing data for filtering...") if verbose else None
        X[('geometry', '')] = [Point(i) for i in
                               zip(x[:-1, :-1].flatten(),
                                   y[:-1, :-1].flatten())]
        X[('in_dallas', '')] = 0

        # Filtrado de celdas (llenado de la columna 'in_dallas')
        if self.shps is not None:
            X = af.filter_cells(df=X, shp=self.shps['councils'],
                                verbose=verbose)
            X.drop(columns=[('in_dallas', '')], inplace=True)

        self.X = X
        self.to_pickle('X.pkl')

    def to_pickle(self, file_name, verbose=False):
        """Genera un pickle de self.data o self.data dependiendo el nombre
        dado (data.pkl o X.pkl).

        OBS.



        luego, si self.read_X = True, no es necesario realizar un
        self.fit() o self.predict()

        Parameters
        ----------
        file_name : str
          Nombre del pickle a generar en predictivehp/data/file_name
        """
        print("\nPickling dataframe...", end=" ") if verbose \
            else None
        if file_name == "X.pkl":
            self.X.to_pickle(f"predictivehp/data/{file_name}")
        if file_name == "data.pkl":
            self.data.to_pickle(f"predictivehp/data/{file_name}")

    def assign_cells(self, verbose=False):
        """Rellena la columna 'Cell' de self.data. Asigna el número de
        celda asociado a cada incidente.

        Parameters
        ----------
        verbose : bool
          Indica si se printean las diferentes acciones del método.
          default False
        """
        print("\tAssigning cells...") if verbose else None
        if self.shps is not None:
            x_min, y_min, x_max, y_max = self.shps['streets'].total_bounds
        else:
            delta_x = 0.1 * self.data.x.mean()
            delta_y = 0.1 * self.data.y.mean()
            x_min = self.data.x.min() - delta_x
            x_max = self.data.x.max() + delta_x
            y_min = self.data.y.min() - delta_y
            y_max = self.data.y.max() + delta_y

        x_bins = abs(x_max - x_min) / self.xc_size
        y_bins = abs(y_max - y_min) / self.yc_size

        x, y = np.mgrid[x_min:x_max:x_bins * 1j, y_min:y_max:y_bins * 1j, ]

        nx = x.shape[0] - 1
        ny = y.shape[1] - 1
        hx = (x.max() - x.min()) / nx
        hy = (y.max() - y.min()) / ny
        for idx, inc in self.data.iterrows():
            xi, yi = inc.geometry.x, inc.geometry.y
            nx_i = af.n_i(xi, x.min(), hx)
            ny_i = af.n_i(yi, y.min(), hy)
            cell_idx = ny_i + ny * nx_i

            self.data.loc[idx, 'Cell'] = cell_idx

        # Dejamos la asociación inc-cell en el index de self.data
        self.data.set_index('Cell', drop=True, inplace=True)
        # self.data.to_pickle("predictivehp/data/data.pkl")

    def fit(self, X, y, verbose=False):
        """Entrena el modelo

        Parameters
        ----------
        X : pd.DataFrame
          X_train
        y : pd.DataFrame
          y_train
        verbose : bool
          Indica si se printean las diferentes acciones del método.
          default False

        Returns
        -------
        self : object
        """
        print("\tFitting Model...") if verbose else None
        self.rfr.fit(X, y.to_numpy().ravel())
        self.X[('Dangerous', '')] = y  # Sirven para determinar celdas con TP/FN
        return self

    def predict(self, X, verbose=False):
        """Predice el score de peligrosidad en cada una de las celdas
        en la malla de Dallas.

        Parameters
        ----------
        X : pd.DataFrame
          X_test for prediction
        verbose : bool
          Indica si se printean las diferentes acciones del método.
          default False

        Returns
        -------
        y : np.ndarray
          Vector de predicción que indica el score de peligrocidad en
          una celda de la malla de Dallas
        """
        print("\tMaking predictions...") if verbose else None
        y_pred = self.rfr.predict(X)
        self.X[('Dangerous_pred', '')] = y_pred / y_pred.max()
        self.X.index.name = 'Cell'
        return y_pred

    def score(self):
        return self.X[('Dangerous_pred', '')]
        # y_pred = self.predict(X_test)
        #
        # score = self.rfr.score(X_test, y_test)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        #
        # print(f"{'Score:':<10s}{score:1.5f}")
        # print(f"{'Precision:':<10s}{precision:1.5f}")
        # print(f"{'Recall:':<10s}{recall:1.5f}")

    def validate(self, c=0, ap=None, verbose=False):
        """

        Parameters
        ----------
        c: {int, float, np.ndarray, list}
        ap: {int, float, np.ndarray, list}
        """
        cells = self.X[[('geometry', ''), ('Dangerous_pred', '')]]
        cells = gpd.GeoDataFrame(cells)

        if ap is not None:
            if self.ap is None:
                self.calculate_pai(np.linspace(0, 1, 1000))

        if type(ap) in {float, np.float64}:
            c = af.find_c(self.ap, self.c_vector, ap)
            print('valor de C encontrado', c) if verbose else None

        elif type(ap) == list or type(ap) == np.ndarray:
            c = [af.find_c(self.ap, self.c_vector, i) for i in ap]
            c = sorted(list(set(c)))
            if len(c) == 1:
                c = c[0]

        if type(c) in {list, np.ndarray}:
            d_cells = cells[
                (c <= cells[('Dangerous_pred', '')]) &
                (cells[('Dangerous_pred', '')] <= c)
                ]
            cells['Hit'] = np.where(
                (c[0] <= cells[('Dangerous_pred', '')])
                & (cells[('Dangerous_pred', '')] <= c[1]),
                1, 0)

        else:
            d_cells = cells[c <= cells[('Dangerous_pred', '')]]
            cells['Hit'] = np.where(
                cells[('Dangerous_pred', '')] >= c, 1, 0
            )
        if self.read_data:
            self.data = pd.read_pickle('predictivehp/data/data.pkl')
        f_data = pd.DataFrame(
            self.data[(self.start_prediction <= self.data.date) &
                      (self.data.date <= self.start_prediction + timedelta(
                          self.length_pred))])
        f_data.columns = pd.MultiIndex.from_product([f_data.columns, ['']])
        cells.drop(columns='geometry', inplace=True)
        join_ = f_data.join(cells)

        hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])

        d_incidents = hits.shape[0]
        h_area = d_cells.shape[0] * self.xc_size * self.yc_size * (10 ** -6)

        a, A = d_cells.shape[0], cells.shape[0]

        self.d_incidents = d_incidents
        self.h_area = h_area
        self.hr_validated = self.d_incidents / f_data.shape[0]
        self.pai_validated = self.hr_validated / (a / A)

    def calculate_hr(self, c=None, verbose=False):
        """
        Parameters
        ----------
        c : {int, float, list, np.ndarray}
          Threshold de confianza para filtrar hotspots
        ap : {int, float, list, np.ndarray}
          Area percentage
        """
        # TODO
        #   .calculate_hr_onelvl(c : {float, np.ndarray, np.float64})
        if c is not None:
            if type(c) in {float, np.float64} or \
                    (type(c) == np.ndarray and c.size == 1):
                # hr = self.calculate_hr_onelvl(c)
                f_data = pd.DataFrame(
                    self.data[(self.start_prediction <= self.data.date) &
                              (
                                      self.data.date <= self.start_prediction + timedelta(
                                  self.length_pred))]
                )  # 62 Incidentes
                if 'geometry' in set(f_data.columns):
                    f_data.drop(columns='geometry', inplace=True)
                f_data.columns = pd.MultiIndex.from_product(
                    [f_data.columns, ['']]
                )
                ans = f_data.join(self.X)
                incidentsh = ans[ans[('Dangerous_pred', '')] >= c[0]]
                hr = incidentsh.shape[0] / f_data.shape[0]
                return hr
            else:
                A = self.X.shape[0]

                def a(X, c):
                    return X[X[('Dangerous_pred', '')] >= c].shape[0]

                c_arr = c
                self.c_vector = c_arr
                hr_l = []
                ap_l = []
                for c in c_arr:
                    hr_l.append(self.calculate_hr(c=np.array([c])))
                    # hr_l.append(self.calculate_hr_onelvl(c)
                    ap_l.append(a(self.X, c) / A)
                self.hr = np.array(hr_l)
                self.ap = np.array(ap_l)

    def calculate_pai(self, c=None, verbose=False):
        """
        Calcula el Predictive Accuracy Index (PAI)

        Parameters
        ----------
        c : {int, float, list, np.ndarray}
          Threshold de confianza para filtrar hotspots
        ap : {int, float, list, np.ndarray}
          Area percentage
        """

        def a(x, c):
            return x[x[('Dangerous_pred', '')] >= c].shape[0]

        A = self.X.shape[0]  # Celdas en Dallas
        if c.size == 1:
            hr = self.calculate_hr(c=c)
            ap = a(self.X, c) / A
            return hr / ap
        else:
            c_arr = c
            self.c_vector = c_arr
            hr_l = []
            ap_l = []
            pai_l = []
            for c in c_arr:
                hr = self.calculate_hr(c=np.array([c]))
                ap = a(self.X, c) / A
                hr_l.append(hr), ap_l.append(ap)
                if ap == 0:
                    pai_l.append(0)
                    continue
                pai_l.append(hr / ap)
            self.hr = np.array(hr_l)
            self.ap = np.array(ap_l)
            self.pai = np.array(pai_l)

    def heatmap(self, c=None, ap=None, incidences=False,
                savefig=False, fname='RFR_heatmap.png',
                verbose=False, **kwargs):
        """

        Parameters
        ----------
        c : {float, list, tuple}
        incidences
        savefig
        fname
        kwargs

        Returns
        -------

        """
        print('\tPlotting Heatmap...') if verbose else None
        fname = f'{fname}.png'
        if self.ap is None:
            self.calculate_pai(np.linspace(0, 1, 1000))
        if type(ap) == float or type(ap) == np.float64:
            c = af.find_c(self.ap, self.c_vector, ap)

        elif type(ap) == list or type(ap) == np.ndarray:
            c = sorted([af.find_c(self.ap, self.c_vector, i) for i in ap])

        cells = self.X[[('geometry', ''), ('Dangerous_pred', '')]]
        cells = gpd.GeoDataFrame(cells)

        if self.shps is not None:
            d_streets = self.shps['streets']

        fig, ax = plt.subplots(figsize=[6.75] * 2)
        if self.shps is not None:
            d_streets.plot(ax=ax, alpha=0.2, lw=0.3, color="w")

        if c is None:
            d_cells = cells[cells[('Dangerous_pred', '')] >= 0.0]
            d_cells.plot(ax=ax, column=('Dangerous_pred', ''), cmap='jet',
                         marker=',', markersize=0.2)  # Heatmap con rango
            cells['Hit'] = np.where(
                cells[('Dangerous_pred', '')] >= 0.0, 1, 0
            )

            # La colorbar se muestra solo cuando es necesario
            # noinspection PyUnresolvedReferences
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            # noinspection PyUnresolvedReferences
            cmap = mpl.cm.jet
            # noinspection PyUnresolvedReferences
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            c_bar = fig.colorbar(mappable, ax=ax,
                                 fraction=0.15,
                                 shrink=0.5,
                                 aspect=21.5)
            c_bar.ax.set_ylabel('Danger Score')
        elif type(c) in {float, int, np.float64}:
            d_cells = cells[cells[('Dangerous_pred', '')] >= c]
            d_cells.plot(ax=ax, marker='o', markersize=10, color='darkred',
                         alpha=0.5)  # Heatmap binario
            cells['Hit'] = np.where(
                cells[('Dangerous_pred', '')] >= c, 1, 0
            )
        else:  # c es una Iterable de c's: Asumo [c0, c1]
            c = np.array(c).flatten()
            c = c[c > 0]
            c = c[c < 1]
            c = np.unique(c)
            c = np.sort(c)

            for idx, i in enumerate(c):
                if idx == 0:
                    lvl = cells[('Dangerous_pred', '')] <= c[idx]
                else:
                    lvl = (c[idx - 1] < cells[('Dangerous_pred', '')]) & \
                          (cells[('Dangerous_pred', '')] <= c[idx])

                d_cells = cells[lvl]
                d_cells.plot(ax=ax, marker='o', markersize=3,
                             color=kwargs['colors'][idx + 1],
                             alpha=0.5)
                cells[f'D{idx + 1}'] = np.where(lvl, 1, 0)

            lvl = c[-1] < cells[('Dangerous_pred', '')]

            d_cells = cells[lvl]
            d_cells.plot(ax=ax, marker='o', markersize=3,
                         color=kwargs['colors'][c.size + 1],
                         alpha=0.5)
            cells[f'D{c.size + 1}'] = np.where(lvl, 1, 0)

            # d1_cells = cells[
            #     (0.0 <= cells[('Dangerous_pred', '')]) &
            #     (cells[('Dangerous_pred', '')] <= c[0])
            #     ]
            # d2_cells = cells[
            #     (c[0] <= cells[('Dangerous_pred', '')]) &
            #     (cells[('Dangerous_pred', '')] <= c[1])
            #     ]
            # d3_cells = cells[
            #     (c[1] <= cells[('Dangerous_pred', '')]) &
            #     (cells[('Dangerous_pred', '')] <= 1.0)
            #     ]
            # d1_cells.plot(ax=ax, marker='.', markersize=1, color='darkblue',
            #               alpha=0.1)
            # d2_cells.plot(ax=ax, marker='.', markersize=1, color='lime',
            #               alpha=0.1)
            # d3_cells.plot(ax=ax, marker='.', markersize=1, color='red',
            #               alpha=0.1)
            # cells['D1'] = np.where(
            #     (0.0 <= cells[('Dangerous_pred', '')]) &
            #     (cells[('Dangerous_pred', '')] <= c[0]), 1, 0
            # )
            # cells['D2'] = np.where(
            #     (c[0] <= cells[('Dangerous_pred', '')]) &
            #     (cells[('Dangerous_pred', '')] <= c[1]), 1, 0
            # )
            # cells['D3'] = np.where(
            #     (c[1] <= cells[('Dangerous_pred', '')]) &
            #     (cells[('Dangerous_pred', '')] <= 1.0), 1, 0
            # )

        if incidences:  # Se plotean los incidentes
            f_data = pd.DataFrame(
                self.data[(self.start_prediction <= self.data.date) &
                          (self.data.date <= self.start_prediction + timedelta(
                              self.length_pred))])
            f_data.columns = pd.MultiIndex.from_product(
                [f_data.columns, ['']]
            )
            cells.drop(columns='geometry', inplace=True)
            join_ = f_data.join(cells)

            if c is None or type(c) in {float, int, np.float64}:
                hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])
                misses = gpd.GeoDataFrame(join_[join_['Hit'] == 0])
                if not hits.empty:
                    hits.plot(ax=ax, marker='x', markersize=1,
                              color='lime',
                              label="Hits")
                if not misses.empty:
                    misses.plot(ax=ax, marker='x', markersize=1,
                                color='red',
                                label="Misses")
            else:
                for i in range(1, c.size + 2):
                    d = gpd.GeoDataFrame(join_[join_[f'D{i}'] == 1])
                    if not d.empty:
                        d.plot(
                            ax=ax, marker='x', markersize=1,
                            color=kwargs['colors'][i], label=f'Level {i}'
                        )

                # d1 = gpd.GeoDataFrame(join_[join_['D1'] == 1])
                # d2 = gpd.GeoDataFrame(join_[join_['D2'] == 1])
                # d3 = gpd.GeoDataFrame(join_[join_['D3'] == 1])
                # if not d1.empty:
                #     d1.plot(ax=ax, marker='x', markersize=0.25,
                #             color='blue', label="Level 1")
                # if not d2.empty:
                #     d2.plot(ax=ax, marker='x', markersize=0.25, color='lime',
                #             label="Level 2")
                # if not d3.empty:
                #     d3.plot(ax=ax, marker='x', markersize=0.25, color='red',
                #             label="Level 3")
            plt.legend()

        plt.title('RForestRegressor')

        plt.tight_layout()
        if savefig:
            plt.savefig(fname, dpi=200, **kwargs)
        plt.show()

    def plot_statistics(self, n=500):
        """

        :return:
        """
        c_arr = np.linspace(0, 1, n)

        def a(x, c): return x[x[('Dangerous_pred_Oct_rfr', '')] >= c].shape[0]

        ap_l, hr_l, pai_l = [], [], []
        for c in c_arr:
            A = self.X.shape[0]  # Celdas en Dallas
            ap = a(self.X, c) / A  # in [0.00, 0.25]
            hr = self.calculate_hr(c=c)
            pai = hr / ap

            ap_l.append(ap), hr_l.append(hr), pai_l.append(pai)

        ap_arr = np.array(ap_l)
        hr_arr, pai_arr = np.array(hr_l), np.array(pai_l)

        af.lineplot(
            x=c_arr, y=ap_arr,
            x_label='c',
            y_label='Area Percentage',
        )
        plt.savefig(
            fname='Statistics_ml_model/ap vs c',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=c_arr, y=hr_arr,
            x_label='c',
            y_label='Hit Rate',
        )
        plt.savefig(
            fname='Statistics_ml_model/HR vs c',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=c_arr, y=pai_arr,
            x_label='c',
            y_label='PAI',
        )
        plt.savefig(
            fname='Statistics_ml_model/PAI vs c',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=ap_arr, y=pai_arr,
            x_label='Area Percentage',
            y_label='PAI',
        )
        plt.savefig(
            fname='Statistics_ml_model/PAI vs ap',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=ap_arr, y=hr_arr,
            x_label='Area Percentage',
            y_label='Hit Rate',
        )
        plt.savefig(
            fname='Statistics_ml_model/HR vs ap',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()

    # def plot_incidents(self, i_type="real", month="October"):
    #     """
    #     Plotea los incidentes almacenados en self.data en el mes dado.
    #     Asegurarse que al momento de realizar el ploteo, ya se haya
    #     hecho un llamado al método ml_algorithm() para identificar los
    #     incidentes TP y FN
    #
    #     :param str i_type: Tipo de incidente a plotear (e.g. TP, FN, TP & FN)
    #     :param str month: String con el nombre del mes que se predijo
    #         con ml_algorithm()
    #     :return:
    #     """
    #
    #     print(f"\nPlotting {month} Incidents...")
    #     print("\tFiltering incidents...")
    #
    #     tp_data, fn_data, data = None, None, None
    #
    #     if i_type == "TP & FN":
    #         data = gpd.GeoDataFrame(self.X)
    #         tp_data = data[self.X.TP == 1]
    #         fn_data = data[self.X.FN == 1]
    #     if i_type == "TP":
    #         data = gpd.GeoDataFrame(self.X)
    #         tp_data = self.X[self.X.TP == 1]
    #     if i_type == "FN":
    #         data = gpd.GeoDataFrame(self.X)
    #         fn_data = self.X[self.X.FN == 1]
    #     if i_type == "real":
    #         data = self.data[self.data.month1 == month]
    #         n_incidents = data.shape[0]
    #         print(f"\tNumber of Incidents in {month}: {n_incidents}")
    #     if i_type == "pred":
    #         data = gpd.GeoDataFrame(self.X)
    #         all_hp = data[self.X[('Dangerous_pred_Oct', '')] == 1]
    #
    #     print("\tReading shapefile...")
    #     d_streets = gpd.GeoDataFrame.from_file(
    #         "../Data/Streets/streets.shp")
    #     d_streets.to_crs(epsg=3857, inplace=True)
    #
    #     print("\tRendering Plot...")
    #     fig, ax = plt.subplots(figsize=(20, 15))
    #
    #     d_streets.plot(ax=ax,
    #                    alpha=0.4,
    #                    color="dimgrey",
    #                    zorder=2,
    #                    label="Streets")
    #
    #     if i_type == 'pred':
    #         all_hp.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='y',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #     if i_type == "real":
    #         data.plot(
    #             ax=ax,
    #             markersize=10,
    #             color='darkorange',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #     if i_type == "TP":
    #         tp_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='red',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #     if i_type == "FN":
    #         fn_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='blue',
    #             marker='o',
    #             zorder=3,
    #             label="FN Incidents"
    #         )
    #     if i_type == "TP & FN":
    #         tp_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='red',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #         fn_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='blue',
    #             marker='o',
    #             zorder=3,
    #             label="FN Incidents"
    #         )
    #
    #     # Legends
    #     handles = [
    #         Line2D([], [],
    #                marker='o',
    #                color='darkorange',
    #                label='Incident',
    #                linestyle='None'),
    #         Line2D([], [],
    #                marker='o',
    #                color='red',
    #                label='TP Incident',
    #                linestyle='None'),
    #         Line2D([], [],
    #                marker='o',
    #                color="blue",
    #                label="FN Incident",
    #                linestyle='None'),
    #         Line2D([], [],
    #                marker='o',
    #                color='y',
    #                label='Predicted Incidents',
    #                linestyle='None')
    #     ]
    #
    #     plt.legend(loc="best",
    #                bbox_to_anchor=(0.1, 0.7),
    #                frameon=False,
    #                fontsize=13.5,
    #                handles=handles)
    #
    #     legends = ax.get_legend()
    #     for text in legends.get_texts():
    #         text.set_color('white')
    #
    #         hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])
    #         misses = gpd.GeoDataFrame(join_[join_['Hit'] == 0])
    #         if not hits.empty:
    #             hits.plot(ax=ax, marker='x', markersize=0.25, color='lime',
    #                       label="Hits")
    #         if not misses.empty:
    #             misses.plot(ax=ax, marker='x', markersize=0.25, color='r',
    #                         label="Misses")
    #
    #     ax.set_axis_off()
    #     fig.set_facecolor('black')
    #     plt.show()
    #     plt.close()

    def plot_hotspots(self):
        """
        Utiliza el método estático asociado para plotear los hotspots
        con los datos ya cargados del framework.

        :return:
        """

        data = self.X[[('geometry', ''),
                       ('Dangerous_Oct', ''),
                       ('Dangerous_pred_Oct', '')]]

        # Quitamos el nivel ''
        data = data.T.reset_index(level=1, drop=True).T

        # Creamos el data para los datos reales (1) y predichos (2).
        data1 = data[['geometry', 'Dangerous_Oct']]
        data2 = data[['geometry', 'Dangerous_pred_Oct']]

        # Filtramos las celdas detectadas como Dangerous para reducir los
        # tiempos de cómputo.
        data1_d = data1[data1['Dangerous_Oct'] == 1]
        data1_nd = data1[data1['Dangerous_Oct'] == 0]
        geodata1_d = gpd.GeoDataFrame(data1_d)
        geodata1_nd = gpd.GeoDataFrame(data1_nd)

        data2_d = data2[data2['Dangerous_pred_Oct'] == 1]
        data2_nd = data2[data2['Dangerous_pred_Oct'] == 0]
        geodata2_d = gpd.GeoDataFrame(data2_d)
        geodata2_nd = gpd.GeoDataFrame(data2_nd)

        self.plot_hotspots_s(geodata1_d, geodata1_nd)
        self.plot_hotspots_s(geodata2_d, geodata2_nd)

    @staticmethod
    def plot_ft_imp_1():
        """
        Barplot de las importancias relativas de los datos agrupados
        por meses.

        :return:
        """

        df = pd.read_pickle('rfc.pkl')
        data = [aux_df.sum()['r_importance'] for aux_df in
                [df[df['features'].isin(
                    [(f'Incidents_{i}', month_name[j]) for i in range(0, 8)])]
                 for j in range(1, 10)]]
        index = [month_name[i] for i in range(1, 10)]

        ax = pd.DataFrame(data=data, index=index, columns=['r_importance']) \
            .plot.bar(y='r_importance', color='black', width=0.25, rot=0,
                      legend=None)

        for i in range(0, 9):
            plt.text(x=i - 0.3, y=data[i] + 0.02 * max(data),
                     s=f'{data[i]:.3f}')

        plt.xlabel("Features",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=10
                   )
        plt.ylabel("Relative Importance",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=7.5
                   )
        plt.xticks(ticks=[i for i in range(0, 9)],
                   labels=[f'{month_name[i]:.3s}' for i in range(1, 10)])
        plt.tick_params(axis='both', length=0, pad=8.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')

        plt.show()

    @staticmethod
    def plot_ft_imp_2():
        """
        Barplot de las importancias relativas de los datos agrupados en
        capas.

        :return:
        """

        df = pd.read_pickle('rfc.pkl')
        data = [aux_df.sum()['r_importance'] for aux_df in
                [df[df['features'].isin(
                    [(f'Incidents_{i}', month_name[j]) for j in range(1, 10)])]
                 for i in range(0, 8)]]
        index = [i for i in range(0, 8)]

        ax = pd.DataFrame(data=data, index=index, columns=['r_importance']) \
            .plot.bar(y='r_importance', color='black', width=0.25, rot=0,
                      legend=None)

        for i in range(0, 8):
            plt.text(x=i - 0.3, y=data[i] + 0.02 * max(data),
                     s=f'{data[i]:.3f}')

        plt.xlabel("Layers",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=10
                   )
        plt.ylabel("Relative Importance",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=7.5
                   )
        plt.xticks(ticks=[i for i in range(0, 8)],
                   labels=[f'{i}' for i in range(0, 8)])
        plt.tick_params(axis='both', length=0, pad=8.5)  # Hide tick lines

        ax.spines['top'].set_visible(False)  # Hide frame
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_color('lightgray')  # Frame color
        ax.spines['left'].set_color('lightgray')

        plt.show()

    @staticmethod
    def plot_ft_imp_3():
        """
        Heatmap que combina las estadísticas de las importancias
        relativas de los datos agrupados por meses y capas.

        :return:
        """

        df = pd.read_pickle('rfc.pkl')
        df.set_index(keys='features', drop=True, inplace=True)

        data = df.to_numpy().reshape(8, 9).T
        columns = [i for i in range(0, 8)]
        index = [f'{month_name[i]:.3s}' for i in range(1, 10)]

        df = pd.DataFrame(data=data, index=index, columns=columns)

        sns.heatmap(data=df, annot=True, annot_kws={"fontsize": 9})

        plt.xlabel("Layers",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=10
                   )
        plt.ylabel("Months",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=7.5
                   )

        plt.tick_params(axis='both', length=0, pad=8.5)
        plt.yticks(rotation=0)

        plt.show()

    @staticmethod
    def plot_hotspots_s(geodata_d, geodata_nd):
        """
            Plotea las celdas de Dallas reconocidas como peligrosas o
            no-peligrosas de acuerdo al algoritmo.

            :param gpd.GeoDataFrame geodata_d: gdf con los puntos de celdas
                peligrosas
            :param gpd.GeoDataFrame geodata_nd: gdf con los puntos de celdas
                no-peligrosas
            """

        print('Reading shapefiles...')
        d_streets = gpd.GeoDataFrame.from_file(
            filename='../Data/Streets/streets.shp'
        )
        d_districts = gpd.GeoDataFrame.from_file(
            filename='../Data/Councils/councils.shp'
        )
        d_streets.to_crs(epsg=3857, inplace=True)
        d_districts.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=(20, 15))
        ax.set_facecolor('xkcd:black')

        for district, data in d_districts.groupby('DISTRICT'):
            data.plot(ax=ax,
                      color=d_colors[district],
                      linewidth=2.5,
                      edgecolor="black")

        handles = [Line2D([], [], marker='o', color='red',
                          label='Dangerous Cell',
                          linestyle='None'),
                   Line2D([], [], marker='o', color="blue",
                          label="Non-Dangerous Cell",
                          linestyle='None')]

        d_streets.plot(ax=ax,
                       alpha=0.4,
                       color="dimgrey",
                       zorder=2,
                       label="Streets")
        geodata_nd.plot(ax=ax,
                        markersize=10,
                        color='blue',
                        marker='o',
                        zorder=3,
                        label="Incidents")
        geodata_d.plot(ax=ax,
                       markersize=10,
                       color='red',
                       marker='o',
                       zorder=3,
                       label="Incidents")

        plt.legend(loc="best",
                   bbox_to_anchor=(0.1, 0.7),
                   frameon=False,
                   fontsize=13.5,
                   handles=handles)

        legends = ax.get_legend()
        for text in legends.get_texts():
            text.set_color('white')

        ax.set_axis_off()
        fig.set_facecolor('black')
        plt.show()
        plt.close()

    # def plot_joined_cells(self):
    #     """
    #
    #     :return:
    #     """
    #
    #     data_oct = pd.DataFrame(self.data[self.data.month1 == 'October'])
    #     data_oct.drop(columns='geometry', inplace=True)
    #
    #     ans = data_oct.join(other=self.X, on='Cell', how='left')
    #     ans = ans[ans[('geometry', '')].notna()]
    #
    #     gpd_ans = gpd.GeoDataFrame(ans, geometry=ans[('geometry', '')])
    #
    #     d_streets = gpd.GeoDataFrame.from_file(
    #         "../Data/Streets/streets.shp")
    #     d_streets.to_crs(epsg=3857, inplace=True)
    #
    #     fig, ax = plt.subplots(figsize=(20, 15))
    #
    #     d_streets.plot(ax=ax,
    #                    alpha=0.4,
    #                    color="dimgrey",
    #                    zorder=2,
    #                    label="Streets")
    #
    #     gpd_ans.plot(
    #         ax=ax,
    #         markersize=10,
    #         color='red',
    #         marker='o',
    #         zorder=3,
    #         label="Joined Incidents"
    #     )
    #
    #     handles = [
    #         Line2D([], [],
    #                marker='o',
    #                color='red',
    #                label='Joined Incidents',
    #                linestyle='None'),
    #     ]
    #
    #     plt.legend(loc="best",
    #                bbox_to_anchor=(0.1, 0.7),
    #                frameon=False,
    #                fontsize=13.5,
    #                handles=handles)
    #
    #     legends = ax.get_legend()
    #     for text in legends.get_texts():
    #         text.set_color('white')
    #
    #         hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])
    #         misses = gpd.GeoDataFrame(join_[join_['Hit'] == 0])
    #         if not hits.empty:
    #             hits.plot(ax=ax, marker='x', markersize=0.25, color='lime',
    #                       label="Hits")
    #         if not misses.empty:
    #             misses.plot(ax=ax, marker='x', markersize=0.25, color='r',
    #                         label="Misses")
    #
    #     ax.set_axis_off()
    #     fig.set_facecolor('black')
    #     plt.show()
    #     plt.close()


class ProMap:
    def __init__(self, data=None, read_density=False,
                 bw_x=400, bw_y=400, bw_t=7, length_prediction=7,
                 tiempo_entrenamiento=None,
                 start_prediction=date(2017, 11, 1),
                 km2=1_000, name='ProMap', shps=None, verbose=False):

        """
        Modelo Promap
        Parameters
        ----------
        n_datos: int
            indica el nº de datos que se usarán para entrenar el modelo
        read_density: bool
            True si se va a leer una matriz de densidades
        hx: int
            Ancho en x de las celdas en metros
        hy: int
            Ancho en y de las celdas en metros
        bw_x: float
            Ancho de banda en x
        bw_y: float
            Ancho de banda en y
        bw_t: float
            Ancho de banda en t
        length_prediction: int
            Número de días que se quiere predecir
        tiempo_entrenamiento: int
            Número de días que se quiern entrenar al modelo.
            Si es None, se usa bw_t
        start_prediction: datetime
            Fecha desde donde comienza la predicción
        name: str
            Nombre del modelo
        shps: gpd.GeoDataFrame
            GeoDataFrame que contiene información sobre la ciudad de Dallas
        """
        # DATA
        self.data = data
        self.start_prediction = start_prediction
        self.X, self.y = None, None
        self.shps = shps
        self.read_density = read_density

        # MAP
        self.bw_x, self.bw_y, self.bw_t = bw_x, bw_y, bw_t

        if self.shps is not None:
            self.x_min, self.y_min, self.x_max, self.y_max = self.shps[
                'streets'].total_bounds
        else:
            delta_x = 0.1 * self.data.x.mean()
            delta_y = 0.1 * self.data.y.mean()
            self.x_min = self.data.x.min() - delta_x
            self.x_max = self.data.x.max() + delta_x
            self.y_min = self.data.y.min() - delta_y
            self.y_max = self.data.y.max() + delta_y

        # MODEL
        self.name = name
        self.lp = length_prediction

        self.hr, self.pai, self.ap = None, None, None

    def set_parameters(self, bw=None, hx=None, hy=None, read_density=False,
                       verbose=False):
        """
        Setea los hiperparámetros del modelo Promap
        Parameters
        ----------
        bw: list or np.array
            anchos de banda en x, y, t
        hx: int
            ancho de la celda en metros, en x
        hy: int
            ancho de la celda en metros, en y
        -------
        """

        if bw:
            self.bw_x, self.bw_y, self.bw_t = bw
        if hx and hy:
            self.hx, self.hy = hx, hy
            self.bins_x = int(round(abs(self.x_max - self.x_min) / self.hx))
            self.bins_y = int(round(abs(self.y_max - self.y_min) / self.hy))
        self.read_density = read_density

    def print_parameters(self):
        """
        Imprime los parametros del modelo
        """

        print('ProMap Hyperparameters')
        print(f'bandwith x: {self.bw_x} mts')
        print(f'bandwith y: {self.bw_y} mts')
        print(f'bandwith t: {self.bw_t} days')
        print(f'hx: {self.hx} mts')
        print(f'hy: {self.hy} mts')
        print()

    def create_grid(self, verbose=False):

        """
        Genera una malla en base a los x{min, max} y{min, max}.
        Recordar que cada nodo de la malla representa el centro de cada
        celda en la malla del mapa.

        """

        print("\nGenerating grid...\n") \
            if verbose else None

        delta_x = self.hx / 2
        delta_y = self.hy / 2

        self.xx, self.yy = np.mgrid[
                           self.x_min + delta_x:self.x_max - delta_x:self.bins_x * 1j,
                           self.y_min + delta_y:self.y_max - delta_y:self.bins_y * 1j
                           ]

    def fit(self, X, y, verbose=False):

        """
        Se encarga de generar la malla en base a los datos del modelo.
        También se encarga de filtrar los puntos que no están dentro de dallas.
        Parameters
        ----------
        X: pd.dataframe
            Son los datos de entrenamiento (x_point, y_point)
        y: pd.dataframe
            Son los datos para el testeo (x_point, y_point)
        """

        # print('Fitting Promap...')

        self.create_grid()
        self.X = X
        self.y = y
        self.dias_train = self.X['y_day'].max()

        points = np.array([self.xx.flatten(), self.yy.flatten()])
        print("\tFitting ProMap...\n") \
            if verbose else None

        if self.shps is not None:
            # self.cells_in_map = af.checked_points_pm(points, self.shps[
            # 'councils'])  #
            self.cells_in_map = 141337
        else:
            cells_x = abs(self.x_min - self.x_max) // self.hx
            cells_y = abs(self.y_min - self.y_max) // self.hy
            self.cells_in_map = cells_x * cells_y

        self.load_test_matrix()

    def predict(self, verbose=False):

        """
        Calula el score en cada celda de la malla de densidades.
        Parameters
        ----------

        """

        if self.read_density:
            self.prediction = np.load(
                'predictivehp/data/prediction.npy')


        else:
            print("\tPredicting...\n") \
                if verbose else None
            self.prediction = np.zeros((self.bins_x, self.bins_y))
            # print('\nEstimando densidades...')
            # print(
            #     f'\n\tNº de datos para entrenar el modelo: {len(self.X)}')
            # print(
            #     f'\tNº de días usados para entrenar el modelo: {self.dias_train}')
            # print(
            #     f'\tNº de datos para testear el modelo: {len(self.y)}')

            ancho_x = af.radio_pintar(self.hx, self.bw_x)
            ancho_y = af.radio_pintar(self.hy, self.bw_y)

            for k in range(len(self.X)):
                x, y, t = self.X['x_point'][k], self.X['y_point'][k], \
                          self.X['y_day'][k]
                x_in_matrix, y_in_matrix = af.find_position(self.xx, self.yy,
                                                            x, y,
                                                            self.hx, self.hy)
                x_left, x_right = af.limites_x(ancho_x, x_in_matrix, self.xx)
                y_abajo, y_up = af.limites_y(ancho_y, y_in_matrix, self.yy)

                for i in range(x_left, x_right):
                    for j in range(y_abajo, y_up):
                        elem_x = self.xx[i][0]
                        elem_y = self.yy[0][j]
                        time_weight = 1 / af.n_semanas(self.dias_train, t)

                        if af.linear_distance(elem_x, x) > self.bw_x or \
                                af.linear_distance(elem_y, y) > self.bw_y:
                            cell_weight = 0

                        else:
                            cell_weight = 1 / af.cells_distance(x, y, elem_x,
                                                                elem_y,
                                                                self.hx,
                                                                self.hy)

                        self.prediction[i][j] += time_weight * cell_weight

            self.prediction = self.prediction / self.prediction.max()

            # print('\nGuardando datos...')
            np.save('predictivehp/data/prediction.npy', self.prediction)

    def load_train_matrix(self):

        """
        Ubica los delitos en la matriz de entrenamiento.
        """

        self.training_matrix = np.zeros((self.bins_x, self.bins_y))
        for index, row in self.X.iterrows():
            x, y, t = row['x_point'], row['y_point'], row['y_day']

            if t >= (self.dias_train - self.bw_t):
                x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                self.hx,
                                                self.hy)
                self.training_matrix[x_pos][y_pos] += 1

    def load_test_matrix(self):

        """
        Ubica los delitos en la matriz de testeo

        Parameters
        ----------
        ventana_dias: int
            Cantidad de días que se quieren predecir.
        """

        self.testing_matrix = np.zeros((self.bins_x, self.bins_y))

        for index, row in self.y.iterrows():
            x, y, t = row['x_point'], row['y_point'], row['y_day']

            if t <= (self.dias_train + self.lp):
                x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                self.hx,
                                                self.hy)
                self.testing_matrix[x_pos][y_pos] += 1

            else:
                break

    def calculate_hr(self, c=None, verbose=False):
        """
        Calcula el hr (n/N)
        Parameters
        ----------
        c: np.ndarray
            Vector que sirve para analizar el mapa en cada punto
        """

        print("\tCalculando HR...\n") \
            if verbose else None

        self.load_train_matrix()
        self.load_test_matrix()

        # 1. Solo considera las celdas que son mayor a un K
        #     Esto me entrega una matriz con True/False (Matriz A)
        # 2. La matriz de True/False la multiplico con una matriz que tiene la
        # cantidad de delitos que hay por cada celda (Matriz B)
        # 3. Al multiplicar A * B obtengo una matriz C donde todos los False en
        # A son 0 en B. Todos los True en A mantienen su valor en B
        # 4. Contar cuantos delitos quedaron luego de haber pasado este proceso.

        # Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta

        if c is None:
            self.c_vector = np.linspace(0, 1, 1000)

        self.c_vector = c

        hits_n = []

        for i in range(c.size):
            hits_n.append(
                np.sum(
                    (self.prediction >= c[
                        i]) * self.testing_matrix))

        # 1. Solo considera las celdas que son mayor a un K
        #     Esto me entrega una matriz con True/False (Matriz A)
        # 2. Contar cuantos celdas quedaron luego de haber pasado este proceso.

        # Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta

        area_hits = []

        for i in range(c.size):
            area_hits.append(
                np.count_nonzero(self.prediction >= c[i]))

        n_delitos_testing = np.sum(self.testing_matrix)

        self.hr = [i / n_delitos_testing for i in hits_n]

        self.ap = [1 if j > 1 else j for j in [i / self.cells_in_map for
                                               i in area_hits]]

    def calculate_pai(self, c=None, verbose=False):

        """
        Calcula el PAI (n/N) / (a/A)
        Parameters
        ----------
        c: np.ndarray
            Vector que sirve para analizar el mapa en cada punto
        """

        print("\tCalculando PAI...\n") \
            if verbose else None

        if c is None:
            self.c_vector = np.linspace(0, 1, 1000)
        else:
            self.c_vector = c

        if not self.hr:
            self.calculate_hr(self.c_vector)

        self.pai = [
            0 if float(self.ap[i]) == 0
            else float(self.hr[i]) / float(self.ap[i])
            for i in range(len(self.ap))]

        # if type(ap) == float or type(ap) == np.float64:
        #     print('PAI: ', af.find_hr_pai(self.pai, self.ap, ap))
        #
        # elif type(ap) == list or type(ap) == np.ndarray:
        #     pais = [af.find_hr_pai(self.pai, self.ap, i) for i in ap]
        #     for index, value in enumerate(pais):
        #         print(f'AP: {ap[index]} PAI: {value}')

    def plot_geopdf(self, dallas, ax, color, label, level):

        geometry = [Point(xy) for xy in zip(
            np.array(self.y[self.y['captured'] == level][['x_point']]),
            np.array(self.y[self.y['captured'] == level][['y_point']]))
                    ]

        if self.shps is not None:
            geo_df = gpd.GeoDataFrame(self.y[self.y['captured'] ==
                                             level],
                                      crs=dallas.crs,
                                      geometry=geometry)
        else:
            geo_df = gpd.GeoDataFrame(self.y[self.y['captured'] ==
                                             level],
                                      geometry=geometry)

        geo_df.plot(ax=ax,
                    markersize=3,
                    color=color,
                    marker='o',
                    zorder=3,
                    label=label,
                    )
        plt.legend()

    def heatmap(self, c=None, show_score=True, incidences=False,
                savefig=False, fname=f'Promap_heatmap.png', ap=None, **kwargs):
        """
        Mostrar un heatmap de una matriz de riesgo.

        Parameters
        ----------
        c: float
            Representa el valor desde donde quiero ver los scores en el mapa
        nombre_grafico: str
            nombre del gráfico
        incidentes: bool
            True para mostrar cuntos incidentes se han capturado
        -------

        """
        if self.shps is not None:
            dallas = self.shps['streets']
        else:
            dallas = None
        # dallas.crs = 2276
        # dallas.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=[6.75] * 2)

        if type(ap) == float or type(ap) == np.float64:
            c = af.find_c(self.ap, self.c_vector, ap)


        elif type(ap) == list or type(ap) == np.ndarray:
            c = [af.find_c(self.ap, self.c_vector, i) for i in ap]
            c = sorted(list(set(c)))
            if len(c) == 1:
                c = c[0]

        matriz = None

        if c is None:
            matriz = self.prediction

        elif type(c) == float or type(c) == np.float64:
            matriz = np.where(self.prediction >= c, 1, 0)
        elif type(c) == list or type(c) == np.ndarray:

            c = np.array(c).flatten()
            c = c[c > 0]
            c = c[c < 1]
            c = np.unique(c)
            c = np.sort(c)
            matriz = np.zeros((self.bins_x, self.bins_y))
            for i in c:
                matriz[self.prediction > i] += 1
            matriz = matriz / np.max(matriz)

        if show_score and c is None:
            # noinspection PyUnresolvedReferences
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            # mpl.cm.set_array(np.ndarray(c))
            # noinspection PyUnresolvedReferences
            cmap = mpl.cm.jet
            # noinspection PyUnresolvedReferences
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            c_bar = fig.colorbar(mappable, ax=ax,
                                 fraction=0.15,
                                 shrink=0.5,
                                 aspect=21.5)
            c_bar.ax.set_ylabel('Danger Score')

        if incidences:

            self.y['captured'] = 0

            plt.imshow(np.flipud(matriz.T),
                       extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                       cmap='jet',
                       # vmin=0, vmax=1
                       alpha=0.3, interpolation=None,
                       vmin=0, vmax=1)

            if self.shps is not None:
                dallas.plot(ax=ax, alpha=0.2, lw=0.3, color="w")

            if c is None:
                self.y['captured'] = 1
                self.plot_geopdf(dallas, ax, color='lime',
                                 label="Hits", level=1)

            if type(c) == float or type(c) == np.float64:
                for index, row in self.y.iterrows():
                    x, y, t = row['x_point'], row['y_point'], row['y_day']

                    if t <= (self.dias_train + self.lp):
                        x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                        self.hx,
                                                        self.hy)
                        if self.prediction[x_pos][y_pos] >= c:
                            self.y['captured'][index] = 1
                    else:
                        break
                if c != 0.0:
                    self.plot_geopdf(dallas, ax, color='red',
                                     label="Misses", level=0)
                self.plot_geopdf(dallas, ax, color='lime',
                                 label="Hits", level=1)

            elif type(c) == list or type(c) == np.ndarray:
                for index_c, c_i in enumerate(c, start=1):
                    for index, row in self.y.iterrows():
                        x, y, t = row['x_point'], row['y_point'], row['y_day']

                        if t <= (self.dias_train + self.lp):
                            x_pos, y_pos = af.find_position(self.xx, self.yy, x,
                                                            y,
                                                            self.hx,
                                                            self.hy)
                            if self.prediction[x_pos][y_pos] >= c_i:
                                self.y['captured'][index] = index_c
                        else:
                            break

                for index in range(len(c) + 1):
                    self.plot_geopdf(dallas, ax, kwargs['colors'][index + 1],
                                     label=f'Level {index + 1}',
                                     level=index)


        else:
            plt.imshow(np.flipud(matriz.T),
                       extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                       cmap='jet',
                       interpolation=None)
            if self.shps is not None:
                dallas.plot(ax=ax, alpha=0.2, lw=0.3, color="w")

        plt.title('ProMap')
        plt.tight_layout()
        if savefig:
            plt.savefig(fname, **kwargs)
        plt.show()

    def score(self):

        """
        Entrega la matriz de riesgo.
        Returns: np.mgrid
        -------

        """
        return self.prediction

    def validate(self, c=0, ap=None, verbose=False):

        self.load_test_matrix()
        self.calculate_pai()

        if type(ap) == float or type(ap) == np.float64:
            c = af.find_c(self.ap, self.c_vector, ap)


        elif type(ap) == list or type(ap) == np.ndarray:
            c = [af.find_c(self.ap, self.c_vector, i) for i in ap]
            c = sorted(list(set(c)))
            if len(c) == 1:
                c = c[0]

        if type(c) != list:
            hits = np.sum(
                (self.prediction >= c) * self.testing_matrix)
            hp_area = np.count_nonzero(self.prediction >= c)

        else:
            c_min, c_max = min(c), max(c)

            aux = np.where(self.prediction >= c_min, self.prediction, 0)
            aux = np.where(aux <= c_max, self.prediction, 0)
            aux = np.where(aux != 0, True, False)

            hits = np.sum(aux * self.testing_matrix)
            hp_area = np.count_nonzero(aux)

        hp_area = min(hp_area, self.cells_in_map)

        self.total_incidents = np.sum(self.testing_matrix)
        self.d_incidents = int(hits)
        self.h_area = hp_area * self.hx * self.hy * 10 ** -6
        self.hr_validated = self.d_incidents / self.total_incidents
        self.pai_validated = self.hr_validated / (hp_area / self.cells_in_map)

    def calculate_ap_c(self):

        self.c_vector = np.linspace(0, 1, 1000)

        area_hits = []

        for i in range(self.c_vector.size):
            area_hits.append(
                np.count_nonzero(self.prediction >= self.c_vector[i]))

        self.ap = [1 if j > 1 else j for j in [i / self.cells_in_map for
                                               i in area_hits]]


class Model:
    def __init__(self, models=None, data=None, shps=None, verbose=False):
        """Supraclase Model"""
        self.models = [] if not models else models
        self.data = data
        self.shps = shps

        self.set_parameters()

    def prepare_stkde(self):
        """

        Returns
        -------

        """
        stkde = list(filter(lambda m: m.name == "STKDE", self.models))[0]

        data = self.data.copy(deep=True)
        geometry = [Point(xy) for xy in zip(np.array(data[['x']]),
                                            np.array(data[['y']]))]

        if self.shps is not None:
            data = gpd.GeoDataFrame(data, crs=2276, geometry=geometry)
            #Paso de sistema de pies a metros
            data.to_crs(epsg=3857, inplace=True)
        else:
            data = gpd.GeoDataFrame(data, geometry=geometry)

        data['x'] = data['geometry'].apply(lambda x: x.x)
        data['y'] = data['geometry'].apply(lambda x: x.y)
        data.loc[:, 'y_day'] = data["date"].apply(
            lambda x: x.timetuple().tm_yday
        )

        # data = data.sample(n=stkde.sn, replace=False, random_state=0)
        # data.sort_values(by=['date'], inplace=True)
        # data.reset_index(drop=True, inplace=True)

        # División en training data (X_train) y testing data (y)
        X_train = data[data["date"] < stkde.start_prediction]
        X_test = data[data["date"] >= stkde.start_prediction]
        X_test = X_test[
            X_test["date"] < stkde.start_prediction
            + timedelta(days=stkde.lp)]
        return X_train, X_test

    def prepare_promap(self):
        promap = list(filter(lambda m: m.name == "ProMap", self.models))[0]
        df = self.data.copy(deep=True)
        # print("\nGenerando dataframe...")

        geometry = [Point(xy) for xy in zip(
            np.array(df[['x']]),
            np.array(df[['y']]))
                    ]

        if self.shps is not None:
            geo_data = gpd.GeoDataFrame(df, crs=2276, geometry=geometry)
            geo_data.to_crs(epsg=3857, inplace=True)
        else:
            geo_data = gpd.GeoDataFrame(df, geometry=geometry)

        df['x_point'] = geo_data['geometry'].x
        df['y_point'] = geo_data['geometry'].y

        df.loc[:, 'y_day'] = df["date"].apply(
            lambda x: x.timetuple().tm_yday
        )

        # División en training y testing data

        X = df[df["date"] < promap.start_prediction]
        y = df[df["date"] >= promap.start_prediction]
        y = y[y["date"] < promap.start_prediction + timedelta(days=promap.lp)]

        return X, y

    def prepare_rfr(self, mode='train', label='default', verbose=False):
        """Prepara el set de datos correspondiente para entrenar RFR y
        predecir para un set dado

        Parameters
        ----------
        mode : str
            Tipo de X, y a retornar. Elegir entre {'train', 'test'}
        label : str
            Establece la forma en la que se generará la label:
            {'default', 'weighted'}. En el caso de 'weighted', se usa
            el atributo .l_weights de la clase para ponderar las
            labels asociadas
        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
        """
        # y en 'label_weights' es una sola columna que corresponde a la
        # suma ponderada de las columnas (considerar division por número
        # de celdas en las capas [o distancia])
        # [('Incidents_i', self.model.weeks[-2])] for i in range(8)
        rfr = [m for m in self.models if m.name == 'RForestRegressor'][0]

        if 'geometry' not in rfr.data.columns:
            rfr.generate_data()
        if rfr.X is None:  # Sin las labels generadas
            rfr.generate_X(verbose)

        if mode == 'train':
            # print("\nPreparing Training Data for RFR...")
            # First three weeks of October
            X = rfr.X.loc[
                :,
                reduce(lambda a, b: a + b,
                       [[(f'Incidents_{i}', week)]
                        for i in range(rfr.n_layers)
                        for week in rfr.weeks[:-2]]
                       )
                ]
            # Last week of October
            # y = self.model.X_train.loc[:, [('Incidents_0', self.model.weeks[-2])]]
            y = rfr.X.loc[
                :, [(f'Incidents_{i}', rfr.weeks[-2])
                    for i in range(rfr.n_layers)]
                ]
            if label == 'default':
                # Cualquier valor != 0 en la fila produce que la celda sea
                # 'Dangerous' = 1
                y[('Dangerous', '')] = y.T.any().astype(int)
            else:
                if rfr.l_weights is not None:
                    w = rfr.l_weights
                else:
                    w = np.array([1 / (l + 1)
                                  for l in range(rfr.n_layers)])
                y[('Dangerous', '')] = y.dot(w)  # Ponderación con los pesos
            y = y[('Dangerous', '')]  # Hace el .drop() del resto de las cols

        else:
            # print("Preparing Testing Data for RFR...")
            X = rfr.X.loc[  # Nos movemos una semana adelante
                :,
                reduce(lambda a, b: a + b,
                       [[(f'Incidents_{i}', week)]
                        for i in range(rfr.n_layers)
                        for week in rfr.weeks[1:-1]]
                       )
                ]
            y = rfr.X.loc[
                :, [(f'Incidents_{i}', rfr.weeks[-1])
                    for i in range(rfr.n_layers)]
                ]
            if label == 'default':
                y[('Dangerous', '')] = y.T.any().astype(int)
            else:
                if rfr.l_weights is not None:
                    w = rfr.l_weights
                else:
                    w = np.array([1 / (l + 1)
                                  for l in range(rfr.n_layers)])
                y[('Dangerous', '')] = y.dot(w)
            y = y[('Dangerous', '')]
        return X, y

    def prepare_data(self, verbose=False):
        dict_ = {}
        for m in self.models:
            m.verbose = True
            if m.name == 'STKDE':
                dict_['STKDE'] = self.prepare_stkde()
            elif m.name == 'ProMap':
                dict_['ProMap'] = self.prepare_promap()
            else:  # RFR
                if m.read_X:
                    m.X = pd.read_pickle('predictivehp/data/X.pkl')
                dict_['RForestRegressor'] = self.prepare_rfr(verbose=verbose)
        return dict_

    def add_model(self, m, verbose=False):
        """Añade el modelo a self.models

        Parameters
        ----------
        m : {STKDE, RForestRegressor, ProMap}
        """
        print(f"\t{m.name} model added") if verbose else None
        self.models.append(m)

    def set_parameters(self, m_name='', **kwargs):
        if not m_name:  # Se setean todos los hyperparameters
            for m in self.models:
                if m.name == 'STKDE':
                    m.set_parameters(bw=[700, 1000, 25])
                if m.name == 'ProMap':
                    m.set_parameters(bw=[1500, 1100, 35], hx=100,
                                     hy=100,
                                     read_density=False)
                if m.name == 'RForestRegressor':
                    m.set_parameters(t_history=4,
                                     xc_size=100, yc_size=100, n_layers=7,
                                     label_weights=None,
                                     read_data=False, read_X=False,
                                     w_data=True, w_X=True)
        else:
            for m in self.models:
                if m.name == m_name:
                    m.set_parameters(**kwargs)
                    break

    def print_parameters(self):
        """
        Printea los hiperparámetros de cada uno de los modelos activos
        en self.stkde, self.promap, self.rfr

        """
        for m in self.models:
            m.print_parameters()

    def fit(self, data_p=None, verbose=False, **kwargs):
        if data_p is None:
            data_p = self.prepare_data(verbose=verbose)
        for m in self.models:
            m.fit(*data_p[m.name], verbose=verbose, **kwargs)

    def predict(self, verbose=False):
        for m in self.models:
            if m.name == 'RForestRegressor':
                X_test = self.prepare_rfr(mode='test', label='default')[0]
                m.predict(X_test)
                continue
            m.predict(verbose=verbose)

    def validate(self, c=None, ap=None, verbose=False):
        """
        Calcula la cantidad de incidentes detectados para los hotspots
        afines.

        nro incidentes, area x cada c

        Parameters
        ----------
        c : {float, list, np.ndarray}
          Umbral de score
        """
        if ap is not None:
            for m in self.models:
                m.validate(ap=ap, verbose=verbose)
        elif c is not None:
            for m in self.models:
                m.validate(c=c, verbose=verbose)
        # for m in self.models:
        #   m.validate(c=c, ap=ap)

    def detected_incidences(self):
        for m in self.models:
            print(f"{m.name}: {m.d_incidents}")

    def hotspot_area(self):
        for m in self.models:
            print(f"{m.name}: {m.h_area} km^2")

    def hr_validated(self):
        for m in self.models:
            print(f"{m.name}: {m.hr_validated}")

    def pai_validated(self):
        for m in self.models:
            print(f"{m.name}: {m.pai_validated}")

    def store(self, file_name='model.data'):
        pass

    # def plot_heatmap(self, c=0, show_score=True, incidences=False,
    #                  savefig=False, fname='', **kwargs):
    #     """
    #
    #     Parameters
    #     ----------
    #     c : {int, float, list}
    #       Si un solo elemento es especificado, luego celdas con un
    #       score superior a c son consideradas parte del hotspot. El
    #       resto quedan de color transparente.
    #
    #       Si es una lista, tenemos hotpots con scores entre c[0] y c[1],
    #       el resto de las celdas quedan transparentes.
    #     incidences : bool
    #       Determina si se plotean o no los incidentes en el heatmap.
    #
    #       Green marker para hits, Red marker para misses.
    #     """
    #     for m in self.models:
    #         m.heatmap(c=c, show_score=show_score, incidences=incidences,
    #                   savefig=savefig, fname=fname, **kwargs)


def create_model(data=None, shps=None,
                 start_prediction=date(2017, 11, 1), length_prediction=7,
                 use_stkde=False, use_promap=False, use_rfr=False):
    """

    Parameters
    ----------
    data
    shps
    start_prediction
    length_prediction
    use_stkde
    use_promap
    use_rfr

    Returns
    -------
    Model
    """
    m = Model(data=data.copy(deep=True))
    m.shps = shps

    if use_promap:
        promap = ProMap(data=data.copy(deep=True), shps=shps,
                        start_prediction=start_prediction,
                        length_prediction=length_prediction)
        m.add_model(promap)
    if use_rfr:
        rfr = RForestRegressor(data_0=data.copy(deep=True), shps=shps,
                               start_prediction=start_prediction,
                               length_prediction=length_prediction
                               )
        m.add_model(rfr)
    if use_stkde:
        stkde = STKDE(data=data.copy(deep=True), shps=shps,
                      start_prediction=start_prediction,
                      length_prediction=length_prediction)
        m.add_model(stkde)
    return m


if __name__ == '__main__':
    pass
