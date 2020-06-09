"""
data_processing.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 10-05-20
Pontifical Catholic University of Chile

"""

import datetime
import predictivehp.credentials as cre
from sodapy import Socrata

from predictivehp.aux_functions import *

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

    streets = gpd.read_file(filename=s_shp) if s_shp else None
    councils = gpd.read_file(filename=c_shp) if c_shp else None
    c_limits = gpd.read_file(filename=cl_shp) if cl_shp else None

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

        df.rename(columns={'x_coordinate': 'x',
                           'y_cordinate': 'y',
                           'date1': 'date'},
                  inplace=True)

        df.sort_values(by=['date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df, streets, councils, c_limits


def grades_to_meters(point):
    lat, lon = point

    ans = gpd.GeoDataFrame(
        geometry=[Point((lat, lon))],
        crs=4326,
    )
    ans.to_crs(epsg=3857, inplace=True)

    value = ans.geometry[0]

    return value.x, value.y


def get_limits_shp(s_shp=''):
    dll = gpd.read_file('./../data/streets.shp')
    dll.crs = 2276  # Source en ft
    dll.to_crs(epsg=3857, inplace=True)
    return dll


if __name__ == '__main__':
    pass
