"""
data_processing.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 10-05-20
Pontifical Catholic University of Chile

"""

import datetime

import geopandas as gpd
import pandas as pd
from sodapy import Socrata

import predictivehp.credentials as cre
import predictivehp.models.parameters as prm
from functools import reduce

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def get_data(year=2017, n=150000, s_shp='', c_shp='', cl_shp=''):
    """
    Obtiene los datos de la Socrata API

    :param int year: Año de la db (e.g. 2017)
    :param int n: Número máximo de muestras a extraer de la db
    :param str s_shp: path al archivo streets.shp
    :param str c_shp: path al archivo councils.shp
    :param str cl_shp: path al archivo citylimits.shp
    :return:
    """
    streets, councils, c_limits = [None, ] * 3

    if s_shp:
        streets = gpd.read_file(filename=s_shp)
        streets.crs = 2276
        streets.to_crs(epsg=3857, inplace=True)
    if c_shp:
        councils = gpd.read_file(filename=c_shp)
        councils.crs = 2276
        councils.to_crs(epsg=3857, inplace=True)
    if cl_shp:
        c_limits = gpd.read_file(filename=cl_shp) if cl_shp else None
        c_limits.crs = 2276
        c_limits.to_crs(epsg=3857, inplace=True)

    print("\nRequesting data...")
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
        print("\n"
              f"\tn = {n} incidents requested  Year = {year}"
              "\n"
              f"\t{df.shape[0]} incidents successfully retrieved!")

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

        df = df[['x_coordinate', 'y_cordinate', 'date1', 'month1']]
        df.loc[:, 'y_day'] = df["date1"].apply(
            lambda x: x.timetuple().tm_yday
        )
        df.loc[:, 'date1'] = df["date1"].apply(
            lambda x: x.date()
        )
        df.rename(columns={'x_coordinate': 'x',
                           'y_cordinate': 'y',
                           'date1': 'date'},
                  inplace=True)

        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df, streets, councils, c_limits


class pre_processing:

    def __init__(self, model, df=None, year=2017, n=150000, s_shp='', c_shp='', cl_shp=''):

        self.model = model
        if self.df:
            self.df = df
        else:
            self.df = get_data(year, n, s_shp, c_shp, cl_shp)

    def preparing_data(self):
        if self.model.name == "STKDE":
            return self.prepare_stkde()
        elif self.model.name == "Promap":
            return self.prepare_promap()
        return self.prepare_rfr()

    def prepare_stkde(self):
        df = self.df
        df = df.sample(n=self.model.sn, replace=False, random_state=250499)
        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # División en training data (X) y testing data (y)
        X = df[df["date"].apply(lambda x: x.month) <= self.model.md]
        X = X[X["date"].apply(
            lambda x: x.month) >= self.model.md - self.model.X_months]
        y = df[df["date"].apply(lambda x: x.month) > self.model.X_months]
        predict_groups = {
            f"group_{i}": {'t1_data': [], 't2_data': [], 'STKDE': None}
            for i in range(1, self.model.ng + 1)}
        days = prm.days_year[self.model.md - 1:]
        days = reduce(lambda a, b: a + b, days)
        # Time 1 Data for building STKDE models : 1 Month
        group_n = 1
        for i in range(1, len(days))[::self.model.wd]:
            predict_groups[f"group_{group_n}"]['t1_data'] = \
                days[i - 1:i - 1 + prm.days_by_month[self.model.md]]
            group_n += 1
            if group_n > self.model.ng:
                break
            # Time 2 Data for Prediction            : 1 Week
        group_n = 1
        for i in range(1, len(days))[::self.model.wd]:
            predict_groups[f"group_{group_n}"]['t2_data'] = \
                days[i - 1 + prm.days_by_month[self.md]:i - 1 +
                                                        prm.days_by_month[
                                                            self.model.md] + self.model.wd]
            group_n += 1
            if group_n > self.model.ng:
                break
        # Time 1 Data for building STKDE models : 1 Month
        for group in predict_groups:
            predict_groups[group]['t1_data'] = \
                df[df['date'].apply(lambda x:
                                    predict_groups[group]['t1_data'][0]
                                    <= x.date() <=
                                    predict_groups[group]['t1_data'][-1])]
        # Time 2 Data for Prediction            : 1 Week
        for group in predict_groups:
            predict_groups[group]['t2_data'] = \
                df[df['date'].apply(lambda x:
                                    predict_groups[group]['t2_data'][0]
                                    <= x.date() <=
                                    predict_groups[group]['t2_data'][-1])]
        return df, X, y, predict_groups

    def prepare_promap(self):
        pass

    def prepare_rfr(self):
        pass

if __name__ == '__main__':
    pass
