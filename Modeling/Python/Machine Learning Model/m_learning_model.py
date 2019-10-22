# m_learning_model.py:
# - Python version: 3.7.1
# - Author: Mauro S. Mendoza Elguera
# - Institution: Pontifical Catholic University of Chile
# - Date: 2019-08-30

import pandas as pd
import numpy as np
import datetime
from calendar import month_name

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sodapy import Socrata
import credentials as cre

from aux_functions import n_i, id_i, nc_incidents, to_df_col
from parameters import dallas_limits

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class Framework:
    def __init__(self, n=1000, year="2017"):
        self.n, self.year = n, year

        self.data = self.get_data()
        self.df = None

        self.x, self.y = None, None
        self.nx, self.ny, self.hx, self.hy = None, None, None, None

        m_dict = {month_name[i]: None for i in range(1, 13)}
        self.incidents = {
            'Incidents': m_dict,
            'NC Incidents': m_dict
        }

        self.generate_df()

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

            return df

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

        # Creación de la malla

        x_bins = abs(dallas_limits['x_max'] - dallas_limits['x_min']) / 100
        y_bins = abs(dallas_limits['y_max'] - dallas_limits['y_min']) / 100

        self.x, self.y = np.mgrid[
                         dallas_limits['x_min']:
                         dallas_limits['x_max']:x_bins * 1j,
                         dallas_limits['y_min']:
                         dallas_limits['y_max']:y_bins * 1j,
                         ]

        # Creación del esqueleto del dataframe

        months = [month_name[i] for i in range(1, 13)]
        cols = pd.MultiIndex.from_product(
                [['Incidents', 'NC Incidents'], months]
        )

        self.df = pd.DataFrame(columns=cols)

        # Creación de los parámetros para el cálculo de los índices

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
        self.data.to_crs(epsg=3857, inplace=True)  # Cambio del m. de referencia

        # Nro. incidentes en la celda(i, j) + Nro. incidentes en celdas vecinas

        for month in [month_name[i] for i in range(1, 13)]:
            fil_incidents = self.data[self.data.month1 == month]

            D = np.zeros((self.nx, self.ny), dtype=int)

            for index, row in fil_incidents.iterrows():
                xi, yi = row.geometry.x, row.geometry.y

                nx_i = n_i(xi, self.x.min(), self.hx)
                ny_i = n_i(yi, self.y.min(), self.hy)

                D[nx_i, ny_i] += 1

            # Actualización del diccionario con las matrices

            self.incidents['Incidents'][month] = D
            self.incidents['NC Incidents'][month] = nc_incidents(D)

            # Actualización del pandas dataframe

            self.df.loc[:, ('Incidents', month)] = to_df_col(D)
            self.df.loc[:, ('NC Incidents', month)] = to_df_col(D)

    def ml_p_algorithm(self):
        """
        Produce la predicción de acuerdo a los datos entregados, utilizando
        un approach de machine learning con clasificador RandomForest

        :return:
        """

        pass


if __name__ == "__main__":
    #%%
    fwork = Framework(n=150000, year="2017")

    # Preparación de los input para el algoritmo

    x_ft = fwork.df.loc[:,
           [('Incidents', month_name[i]) for i in range(1, 10)] +
           [('NC Incidents', month_name[i]) for i in range(1, 10)]
           ]

    x_lbl = fwork.df.loc[:,
            [('Incidents', 'October')] + [('NC Incidents', 'October')]
            ]

    y_ft = fwork.df.loc[:,
           [('Incidents', month_name[i]) for i in range(2, 11)] +
           [('NC Incidents', month_name[i]) for i in range(2, 11)]
           ]
    y_lbl = fwork.df.loc[:,
            [('Incidents', 'November')] + [('NC Incidents', 'November')]
            ]

    #%%
    # Algoritmo

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_ft, x_lbl)

    x_predict = clf.predict(x_ft)
    y_predict = clf.predict(y_ft)
