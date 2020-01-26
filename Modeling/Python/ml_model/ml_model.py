"""
ml_model.py:
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 30-08-19
Pontifical Catholic University of Chile

Notes

-
"""

import pandas as pd
import numpy as np
import datetime
from calendar import month_name
from time import time

import geopandas as gpd
from shapely.geometry import Point
from fiona.crs import from_epsg

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

from sodapy import Socrata
import credentials as cre

from aux_functions import *
from parameters import dallas_limits
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class Framework:
    def __init__(self, n=1000, year="2017", read_df=True):
        self.n, self.year = n, year

        self.data = None
        self.df = None

        self.x, self.y = None, None
        self.nx, self.ny, self.hx, self.hy = None, None, None, None

        m_dict = {month_name[i]: None for i in range(1, 13)}
        self.incidents = {
            'Incidents': m_dict,
            'NC Incidents_1': m_dict,
            'NC Incidents_2': m_dict,
            'NC Incidents_3': m_dict,
            'NC Incidents_4': m_dict,
            'NC Incidents_5': m_dict,
            'NC Incidents_6': m_dict,
            'NC Incidents_7': m_dict,
        }

        if read_df:
            st = time()

            print("\nReading pickle dataframe...", end=" ")
            self.df = pd.read_pickle('df.pkl')
            print(f"finished! ({time() - st:3.1f} sec)")
        else:
            self.get_data()
            self.generate_df()

    @timer
    def get_data(self):
        """
        Obtención de datos a partir de la Socrata API.

        Por ahora se está realizando un filtro para obtener solo  incidentes
        asociados a robos residenciales

        :return:
        """

        print("\nRequesting data...")

        with Socrata(cre.socrata_domain,
                     cre.API_KEY_S,
                     username=cre.USERNAME_S,
                     password=cre.PASSWORD_S) as client:
            # Actualmente estamos filtrando por robos a domicilios

            where = \
                f"""
                    year1 = {self.year}
                    and date1 is not null
                    and time1 is not null
                    and x_coordinate is not null
                    and y_cordinate is not null
                    and offincident = 'BURGLARY OF HABITATION - FORCED ENTRY'
                """  #  571000 max. 09/07/2019

            results = client.get(cre.socrata_dataset_identifier,
                                 where=where,
                                 order="date1 ASC",
                                 limit=self.n,
                                 content_type='json')

            df = pd.DataFrame.from_records(results)

            print(f"\n\t{df.shape[0]} records successfully retrieved!")

            # DB Cleaning & Formatting

            df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(
                lambda x: float(x))
            df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(
                lambda x: float(x))
            df.loc[:, 'date1'] = df['date1'].apply(
                lambda x: datetime.datetime.strptime(
                    x.split('T')[0], '%Y-%m-%d')
            )
            df.loc[:, 'y_day'] = df["date1"].apply(
                lambda x: x.timetuple().tm_yday
            )

            df.rename(columns={'x_coordinate': 'x',
                               'y_cordinate': 'y',
                               'date1': 'date'},
                      inplace=True)

            df.sort_values(by=['date'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            self.data = df

    @timer
    def generate_df(self):
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

        :return: Pandas Dataframe con la información
        """

        print("\nGenerating dataframe...\n")

        # Creación de la malla
        print("\tCreating mgrid...")

        x_bins = abs(dallas_limits['x_max'] - dallas_limits['x_min']) / 100
        y_bins = abs(dallas_limits['y_max'] - dallas_limits['y_min']) / 100

        self.x, self.y = np.mgrid[
                         dallas_limits['x_min']:
                         dallas_limits['x_max']:x_bins * 1j,
                         dallas_limits['y_min']:
                         dallas_limits['y_max']:y_bins * 1j,
                         ]

        # Creación del esqueleto del dataframe
        print("\tCreating dataframe columns...")

        months = [month_name[i] for i in range(1, 13)]
        columns = pd.MultiIndex.from_product(
            [['Incidents_0', 'Incidents_1', 'Incidents_2', 'Incidents_3',
              'Incidents_4', 'Incidents_5', 'Incidents_6', 'Incidents_7'],
             months]
        )

        self.df = pd.DataFrame(columns=columns)

        # Creación de los parámetros para el cálculo de los índices
        print("\tFilling df...")

        self.nx = self.x.shape[0] - 1
        self.ny = self.y.shape[1] - 1

        self.hx = (self.x.max() - self.x.min()) / self.nx
        self.hy = (self.y.max() - self.y.min()) / self.ny

        # Manejo de los puntos de incidentes para poder trabajar en (x, y)
        # y realizar Feature Engineering
        geometry = [Point(xy) for xy in zip(
            np.array(self.data[['x']]),
            np.array(self.data[['y']]))
                    ]
        self.data = gpd.GeoDataFrame(self.data,  # gdf de incidentes
                                     crs=2276,
                                     geometry=geometry)

        self.data.to_crs(epsg=3857, inplace=True)
        fig, ax = plt.subplots(figsize=(20, 15))
        fig.set_facecolor('black')
        ax.set_facecolor('xkcd:black')
        self.data.plot(ax=ax, markersize=15, color='red', marker='o')
        plt.show()

        # Nro. incidentes en la i-ésima capa de la celda (i, j)
        for month in [month_name[i] for i in range(1, 13)]:
            print(f"\t\t{month}... ", end=' ')

            fil_incidents = self.data[self.data.month1 == month]
            D = np.zeros((self.nx, self.ny), dtype=int)

            for _, row in fil_incidents.iterrows():
                xi, yi = row.geometry.x, row.geometry.y

                nx_i = n_i(xi, self.x.min(), self.hx)
                ny_i = n_i(yi, self.y.min(), self.hy)
                D[nx_i, ny_i] += 1

            # Actualización del pandas dataframe
            self.df.loc[:, ('Incidents_0', month)] = to_df_col(D)
            self.df.loc[:, ('Incidents_1', month)] = \
                to_df_col(il_neighbors(matrix=D, i=1))
            self.df.loc[:, ('Incidents_2', month)] = \
                to_df_col(il_neighbors(matrix=D, i=2))
            self.df.loc[:, ('Incidents_3', month)] = \
                to_df_col(il_neighbors(matrix=D, i=3))
            self.df.loc[:, ('Incidents_4', month)] = \
                to_df_col(il_neighbors(matrix=D, i=4))
            self.df.loc[:, ('Incidents_5', month)] = \
                to_df_col(il_neighbors(matrix=D, i=5))
            self.df.loc[:, ('Incidents_6', month)] = \
                to_df_col(il_neighbors(matrix=D, i=6))
            self.df.loc[:, ('Incidents_7', month)] = \
                to_df_col(il_neighbors(matrix=D, i=7))

            print('finished!')

        # Adición de las columnas 'geometry' e 'in_dallas' al df
        print("\tPreparing df for filtering...")

        self.df['geometry'] = [Point(i) for i in
                               zip(self.x[:-1, :-1].flatten(),
                                   self.y[:-1, :-1].flatten())]

        # print(self.df.geometry.head(200))

        self.df['in_dallas'] = 0

        # Llenado de la columna 'in_dallas'
        self.df = filter_cells(self.df)
        self.df.drop(columns=[('in_dallas', '')], inplace=True)

        # Garbage recollection
        del self.data, self.incidents, self.x, self.y

    @timer
    def ml_algorithm(self, f_importance=False, pickle=False):
        """
        Produce la predicción de acuerdo a los datos entregados, utilizando
        un approach de machine learning con clasificador RandomForest (rfc) y
        entrega el output asociado.

        :param f_importance: True para imprimir estadísticas
            asociadas a los features utilizados para entrenar el classifier
        :param pickle: True si se quiere generar un pickle de las estadísticas
            asociadas al entrenamiento del classifier
        """

        print("\nInitializing...")

        # Preparación del input para el algoritmo

        print("\n\tPreparing input...")

        x_ft = self.df.loc[
               :,
               [('Incidents_0', month_name[i]) for i in range(1, 10)] +
               [('Incidents_1', month_name[i]) for i in range(1, 10)] +
               [('Incidents_2', month_name[i]) for i in range(1, 10)] +
               [('Incidents_3', month_name[i]) for i in range(1, 10)] +
               [('Incidents_4', month_name[i]) for i in range(1, 10)] +
               [('Incidents_5', month_name[i]) for i in range(1, 10)] +
               [('Incidents_6', month_name[i]) for i in range(1, 10)] +
               [('Incidents_7', month_name[i]) for i in range(1, 10)]
               ]
        x_lbl = self.df.loc[
                :,
                [('Incidents_0', 'October'), ('Incidents_1', 'October'),
                 ('Incidents_2', 'October'), ('Incidents_3', 'October'),
                 ('Incidents_4', 'October'), ('Incidents_5', 'October'),
                 ('Incidents_6', 'October'), ('Incidents_7', 'October')]
                ]

        x_lbl[('Dangerous', '')] = x_lbl.T.any().astype(int)
        x_lbl = x_lbl[('Dangerous', '')]

        # Algoritmo

        print("\tRunning algorithms...")

        rfc = RandomForestClassifier(n_jobs=8)
        rfc.fit(x_ft, x_lbl.to_numpy().ravel())
        x_pred_rfc = rfc.predict(x_ft)

        # dtc = DecisionTreeClassifier()
        # dtc.fit(x_ft, x_lbl.to_numpy().ravel())
        # rbf_svm = SVC()
        # rbf_svm.fit(x_ft, x_lbl)
        # x_pred_dtc = dtc.predict(x_ft)
        # x_pred_rbf_svm = rbf_svm.predict(x_ft)

        if f_importance:
            cols = pd.Index(['features', 'r_importance'])
            rfc_fi_df = pd.DataFrame(columns=cols)
            rfc_fi_df['features'] = x_ft.columns.to_numpy()
            rfc_fi_df['r_importance'] = rfc.feature_importances_

            if pickle:
                rfc_fi_df.to_pickle('rfc.pkl')

            # print('\n', rfc_fi_df)

        print("\n\tx\n")

        self.df[('Dangerous_Oct', '')] = x_lbl
        self.df[('Dangerous_pred_Oct', '')] = x_pred_rfc

        rfc_score = rfc.score(x_ft, x_lbl)
        # dtc_score = dtc.score(x_ft, x_lbl)
        # rbf_svm_score = rbf_svm.score(x_ft, x_lbl)

        rfc_precision = precision_score(x_lbl, x_pred_rfc)
        # dtc_precision = precision_score(x_lbl, x_pred_dtc)
        # rbf_svm_precision = precision_score(x_lbl, x_pred_rbf_svm)

        rfc_recall = recall_score(x_lbl, x_pred_rfc)
        # dtc_recall = recall_score(x_lbl, x_pred_dtc)
        # rbf_svm_recall = recall_score(x_lbl, x_pred_rbf_svm)

        print(
            f"""
    rfc score           {rfc_score:1.3f}
    rfc precision       {rfc_precision:1.3f}
    rfc recall          {rfc_recall:1.3f}
        """
            # dtc score           {dtc_score:1.3f}
            # dtc precision       {dtc_precision:1.3f}
            # dtc recall          {dtc_recall:1.3f}

            # rbf_svm score       {rbf_svm_score:1.9f}
            # rbf_svm precision   {rbf_svm_precision:1.9f}
            # rbf_svm recall      {rbf_svm_recall:1.9f}
            #         """
        )

        print("\n\ty\n")

        y_ft = self.df.loc[
               :,
               [('Incidents_0', month_name[i]) for i in range(2, 11)] +
               [('Incidents_1', month_name[i]) for i in range(2, 11)] +
               [('Incidents_2', month_name[i]) for i in range(2, 11)] +
               [('Incidents_3', month_name[i]) for i in range(2, 11)] +
               [('Incidents_4', month_name[i]) for i in range(2, 11)] +
               [('Incidents_5', month_name[i]) for i in range(2, 11)] +
               [('Incidents_6', month_name[i]) for i in range(2, 11)] +
               [('Incidents_7', month_name[i]) for i in range(2, 11)]
               ]
        y_lbl = self.df.loc[
                :,
                [('Incidents_0', 'November'), ('Incidents_1', 'November'),
                 ('Incidents_2', 'November'), ('Incidents_3', 'November'),
                 ('Incidents_4', 'November'), ('Incidents_5', 'November'),
                 ('Incidents_6', 'November'), ('Incidents_7', 'November')]
                ]

        y_lbl[('Dangerous', '')] = y_lbl.T.any().astype(int)
        y_lbl = y_lbl[('Dangerous', '')]

        y_pred_rfc = rfc.predict(y_ft)
        # y_pred_dtc = dtc.predict(y_ft)
        # y_pred_rbf_svm = rbf_svm.predict(y_ft)

        rfc_score = rfc.score(y_ft, y_lbl.to_numpy().ravel())
        # dtc_score = dtc.score(y_ft, y_lbl.to_numpy().ravel())
        # rbf_svm_score = rbf_svm.score(y_ft, y_lbl)

        rfc_precision = precision_score(y_lbl, y_pred_rfc)
        # dtc_precision = precision_score(y_lbl, y_pred_dtc)
        # rbf_svm_precision = precision_score(y_lbl, y_pred_rbf_svm)

        rfc_recall = recall_score(y_lbl, y_pred_rfc)
        # dtc_recall = recall_score(y_lbl, y_pred_dtc)
        # rbf_svm_recall = recall_score(y_lbl, y_pred_rbf_svm)

        print(
            f"""
    rfc score           {rfc_score:1.3f}
    rfc precision       {rfc_precision:1.3f}
    rfc recall          {rfc_recall:1.3f}
            """
            # dtc score           {dtc_score:1.3f}
            # dtc precision       {dtc_precision:1.3f}
            # dtc recall          {dtc_recall:1.3f}

            # rbf_svm score       {rbf_svm_score:1.3f}
            # rbf_svm precision   {rbf_svm_precision:1.3f}
            # rbf_svm recall      {rbf_svm_recall:1.3f}
            #         """
        )

        # Confusion Matrix

        # print("\tComputando matrices de confusión...", end="\n\n")
        #
        # c_matrix_x = confusion_matrix(
        #         x_lbl[('Dangerous', '')], x_predict[:, 0]
        # )
        #
        # print(c_matrix_x, end="\n\n")
        #
        # c_matrix_y = confusion_matrix(
        #         y_lbl[('Dangerous', '')], y_predict[:, 0]
        # )
        #
        # print(c_matrix_y)

    @timer
    def df_to_pickle(self, file_name='df.pkl'):
        """
        Genera un pickle de self.df

        :param str file_name: Nombre del pickle a generar
        :return: pickle de self.df
        """

        print("\nPickling dataframe...", end=" ")
        self.df.to_pickle(file_name)


if __name__ == "__main__":
    # TODO
    #       - Comparar entre etiqueta 1 y 3
    #       - Mejorar feature engineering con medidas del modelo ProMap
    #       - Pensar implementación de HR/PAI
    #       - Comparación de rendimiento Bin. Class vs Multi. Class

    fwork = Framework(n=150000, year="2017", read_df=False)
    # fwork.ml_algorithm(f_importance=False, pickle=False)

    # aux_df = fwork.df
    #
    # X1 = aux_df.loc[:,
    #       [('Incidents', month_name[i]) for i in range(1, 10)] +
    #       [('NC Incidents', month_name[i]) for i in range(1, 10)]
    #      ].to_numpy()
    #
    # y1 = aux_df.loc[:,
    #       [('Incidents', 'October'), ('NC Incidents', 'October')]
    #      ]
    #
    # y1[('Dangerous', '')] = ((y1[('Incidents', 'October')] != 0) |
    #                         (y1[('NC Incidents', 'October')] != 0)) \
    #     .astype(int)
    # y1.drop([('Incidents', 'October'), ('NC Incidents', 'October')],
    #         axis=1,
    #         inplace=True)
    #
    # y1 = y1.to_numpy().ravel()

    # bc = BaggingClassifier(RandomForestClassifier(n_jobs=8), n_jobs=8)
    # bc.fit(X1, y1)

    # print(
    #     f"""
    # bc score           {bc.score(X1, y1):1.3f}
    # bc precision       {precision_score(y1, bc.predict(X1)):1.3f}
    # bc recall          {recall_score(y1, bc.predict(X1)):1.3f}
    #         """
    # )
    #
    # abc = AdaBoostClassifier()
    # abc.fit(X1, y1)
    #
    # print(
    #     f"""
    # abc score           {abc.score(X1, y1):1.3f}
    # abc precision       {precision_score(y1, abc.predict(X1)):1.3f}
    # abc recall          {recall_score(y1, abc.predict(X1)):1.3f}
    #     """
    # )

    # X2 = aux_df.loc[:,
    #        [('Incidents', month_name[i]) for i in range(2, 11)] +
    #        [('NC Incidents', month_name[i]) for i in range(2, 11)]
    #      ].to_numpy()
    #
    # y2 = aux_df.loc[:,
    #         [('Incidents', 'November')] + [('NC Incidents', 'November')]]
    # y2[('Dangerous', '')] = \
    #     ((y2[('Incidents', 'November')] != 0) |
    #      (y2[('NC Incidents', 'November')] != 0)).astype(int)
    # y2.drop([('Incidents', 'November'), ('NC Incidents', 'November')],
    #         axis=1, inplace=True)
    #
    # y2 = y2.to_numpy().ravel()
