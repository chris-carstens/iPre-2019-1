"""
data_processing.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 10-05-20
Pontifical Catholic University of Chile

"""

import datetime
from functools import reduce

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sodapy import Socrata

import predictivehp.credentials as cre
import predictivehp.models.parameters as prm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class PreProcessing:
    def __init__(self, models=None, df=None, year=2017, n=150000):
        """

        Parameters
        ----------
        models : {None, list}
          Lista de modelos
        df
        year
        n
        """
        self.models = models
        self.stkde = None
        self.promap = None
        self.rfr = None
        if models:
            self.define_models()
        if df is not None:
            self.df = df
        else:
            self.df = self.get_data(year=year, n=n)

    @staticmethod
    def get_data(year=2017, n=150000):
        # print("\nRequesting data...")
        with Socrata(cre.socrata_domain,
                     cre.API_KEY_S,
                     username=cre.USERNAME_S,
                     password=cre.PASSWORD_S) as client:
            query = \
                f"""
                select
                    incidentnum,
                    year1,
                    date1,
                    month1,
                    time1,
                    x_coordinate,
                    y_cordinate,
                    offincident
                where
                    year1 = {year}
                    and date1 is not null
                    and time1 is not null
                    and x_coordinate is not null
                    and y_cordinate is not null
                    and offincident = 'BURGLARY OF HABITATION - FORCED ENTRY'
                order by date1
                limit
                    {n}
                """

            results = client.get(cre.socrata_dataset_identifier,
                                 query=query,
                                 content_type='json')
            df = pd.DataFrame.from_records(results)
            # print("\n"
            #       f"\tn = {n} incidents requested  Year = {year}"
            #       "\n"
            #       f"\t{df.shape[0]} incidents successfully retrieved!")

            # DB Cleaning & Formatting
            for col in ['x_coordinate', 'y_cordinate']:
                df.loc[:, col] = df[col].apply(
                    lambda x: float(x))
            df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(
                lambda x: float(x))
            df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(
                lambda x: float(x))
            df.loc[:, 'date1'] = df['date1'].apply(  # OJO AL SEPARADOR ' '
                lambda x: datetime.datetime.strptime(
                    x.split(' ')[0], '%Y-%m-%d')
            )
            df.loc[:, 'date1'] = df["date1"].apply(lambda x: x.date())

            df = df[['x_coordinate', 'y_cordinate', 'date1', 'month1']]
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

    @staticmethod
    def shps_processing(s_shp='', c_shp='', cl_shp=''):
        """

        Parameters
        ----------
        s_shp
        c_shp
        cl_shp

        Returns
        -------

        """
        streets, councils, c_limits = [None, ] * 3
        shps = {}
        if s_shp:
            streets = gpd.read_file(filename=s_shp)
            streets.crs = 2276
            streets.to_crs(epsg=3857, inplace=True)
        if c_shp:
            councils = gpd.read_file(filename=c_shp)
            councils.crs = 2276
            councils.to_crs(epsg=3857, inplace=True)
        if cl_shp:
            c_limits = gpd.read_file(filename=cl_shp)
            c_limits.crs = 2276
            c_limits.to_crs(epsg=3857, inplace=True)

        shps['streets'], shps['councils'], shps['c_limits'] = \
            streets, councils, c_limits

        return shps

    def define_models(self):
        for model in self.models:
            self.stkde = model if model.name == 'STKDE' else self.stkde
            self.promap = model if model.name == 'ProMap' else self.promap
            self.rfr = model if model.name == 'RForestRegressor' else self.rfr

    def preparing_data(self, model, **kwargs):
        if model not in [m.name for m in self.models]:
            print("Model not found!\n")
            return None
        if "STKDE" in model:
            return self.prepare_stkde()
        elif "ProMap" in model:
            return self.prepare_promap()
        return self.prepare_rfr(**kwargs)

    def prepare_stkde(self):
        df = self.df
        df = df.sample(n=self.stkde.sn, replace=False, random_state=250499)
        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # División en training data (X) y testing data (y)
        X = df[df["date"] <= self.stkde.start_prediction]
        X = X[X["date"].apply(
            lambda x: x.month) >= self.stkde.md - self.stkde.X_months]
        y = df[df["date"] > self.stkde.start_prediction]
        predict_groups = {
            f"group_{i}": {'t1_data': [], 't2_data': [], 'STKDE': None}
            for i in range(1, self.stkde.ng + 1)}
        days = prm.days_year[self.stkde.md - 1:]
        days = reduce(lambda a, b: a + b, days)
        # Time 1 Data for building STKDE models : 1 Month
        group_n = 1
        for i in range(1, len(days))[::self.stkde.wd]:
            predict_groups[f"group_{group_n}"]['t1_data'] = \
                days[i - 1:i - 1 + prm.days_by_month[self.stkde.md]]
            group_n += 1
            if group_n > self.stkde.ng:
                break
            # Time 2 Data for Prediction            : 1 Week
        group_n = 1
        for i in range(1, len(days))[::self.stkde.wd]:
            predict_groups[f"group_{group_n}"]['t2_data'] = \
                days[i - 1 + prm.days_by_month[self.stkde.md]:i - 1 +
                                                              prm.days_by_month[
                                                                  self.stkde.md] + self.stkde.wd]
            group_n += 1
            if group_n > self.stkde.ng:
                break
        # Time 1 Data for building STKDE models : 1 Month
        for group in predict_groups:
            predict_groups[group]['t1_data'] = \
                df[df['date'].apply(lambda x:
                                    predict_groups[group]['t1_data'][0]
                                    <= x <=
                                    predict_groups[group]['t1_data'][-1])]
        # Time 2 Data for Prediction            : 1 Week
        for group in predict_groups:
            predict_groups[group]['t2_data'] = \
                df[df['date'].apply(lambda x:
                                    predict_groups[group]['t2_data'][0]
                                    <= x <=
                                    predict_groups[group]['t2_data'][-1])]
        return df, X, y, predict_groups

    def prepare_promap(self):

        df = self.df

        if len(df) >= self.promap.n:
            # print(f'\nEligiendo {self.promap.n} datos...')
            df = df.sample(n=self.promap.n,
                           replace=False,
                           random_state=250499)
            df.sort_values(by=['date'], inplace=True)
            df.reset_index(drop=True, inplace=True)

        # print("\nGenerando dataframe...")

        geometry = [Point(xy) for xy in zip(
            np.array(df[['x']]),
            np.array(df[['y']]))
                    ]

        geo_data = gpd.GeoDataFrame(df,  # gdf de incidentes
                                    crs=2276,
                                    geometry=geometry)

        geo_data.to_crs(epsg=3857, inplace=True)

        df['x_point'] = geo_data['geometry'].x
        df['y_point'] = geo_data['geometry'].y

        # División en training y testing data

        X = df[df["date"] <= self.promap.start_prediction]
        y = df[df["date"] > self.promap.start_prediction]

        return X, y

    def prepare_rfr(self, mode='train', label='default'):
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
        if self.rfr.X is None:
            self.rfr.generate_df()

        if mode == 'train':
            # print("\nPreparing Training Data for RFR...")
            # First three weeks of October
            X = self.rfr.X.loc[
                :,
                reduce(lambda a, b: a + b,
                       [[(f'Incidents_{i}', week)]
                        for i in range(self.rfr.n_layers)
                        for week in self.rfr.weeks[:-2]]
                       )
                ]
            # Last week of October
            # y = self.model.X.loc[:, [('Incidents_0', self.model.weeks[-2])]]
            y = self.rfr.X.loc[
                :, [(f'Incidents_{i}', self.rfr.weeks[-2])
                    for i in range(self.rfr.n_layers)]
                ]
            if label == 'default':
                # Cualquier valor != 0 en la fila produce que la celda sea
                # 'Dangerous' = 1
                y[('Dangerous', '')] = y.T.any().astype(int)
            else:
                if self.rfr.l_weights is not None:
                    w = self.rfr.l_weights
                else:
                    w = np.array([1 / (l + 1)
                                  for l in range(self.rfr.n_layers)])
                y[('Dangerous', '')] = y.dot(w)  # Ponderación con los pesos
            y = y[('Dangerous', '')]  # Hace el .drop() del resto de las cols

        else:
            # print("Preparing Testing Data for RFR...")
            # Nos movemos una semana adelante
            X = self.rfr.X.loc[
                :,
                reduce(lambda a, b: a + b,
                       [[(f'Incidents_{i}', week)]
                        for i in range(self.rfr.n_layers)
                        for week in self.rfr.weeks[1:-1]]
                       )
                ]
            y = self.rfr.X.loc[
                :, [(f'Incidents_{i}', self.rfr.weeks[-1])
                    for i in range(self.rfr.n_layers)]
                ]
            if label == 'default':
                y[('Dangerous', '')] = y.T.any().astype(int)
            else:
                if self.rfr.l_weights is not None:
                    w = self.rfr.l_weights
                else:
                    w = np.array([1 / (l + 1)
                                  for l in range(self.rfr.n_layers)])
                y[('Dangerous', '')] = y.dot(w)
            y = y[('Dangerous', '')]
        return X, y


if __name__ == '__main__':
    pass
