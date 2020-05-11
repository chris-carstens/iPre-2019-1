"""
data_processing.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 10-05-20
Pontifical Catholic University of Chile

"""

from aux_functions import *

import datetime
import credentials as cre
from sodapy import Socrata
from parameters import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def get_data(model='STKDE', year=2017, n=150000):
    """
    Obtiene los datos de la Socrata API

    :param str model: Uno entre 'STKDE', 'ProMap, 'ML'
    :param int year: Año de la db (e.g. 2017)
    :param int n: Número máximo de muestras a extraer de la db
    :return:
    """

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

        if model == 'STKDE' or model == 'ProMap':
            # Reducción del tamaño de la DB
            if n >= 3600:
                df = df.sample(n=3600,
                               replace=False,
                               random_state=250499)
                df.sort_values(by=['date'], inplace=True)
                df.reset_index(drop=True, inplace=True)

            # División en training data (X) y testing data (y)
            X = df[df["date"].apply(lambda x: x.month) <= 10]
            y = df[df["date"].apply(lambda x: x.month) > 10]

            if model == 'STKDE':
                predict_groups = {f"group_{i}": {'t1_data': [], 't2_data': [], 'STKDE': None} for i in range(1, 9)}

                # Time 1 Data for building STKDE models : 1 Month
                group_n = 1
                for i in range(1, len(days_oct_nov_dic))[::7]:
                    predict_groups[f"group_{group_n}"]['t1_data'] = \
                        days_oct_nov_dic[i - 1:i - 1 + days_oct]

                    group_n += 1
                    if group_n > 8:
                        break
                # Time 2 Data for Prediction            : 1 Week
                group_n = 1
                for i in range(1, len(days_oct_nov_dic))[::7]:
                    predict_groups[f"group_{group_n}"]['t2_data'] = \
                        days_oct_nov_dic[i - 1 + days_oct:i - 1 + days_oct + 7]

                    group_n += 1
                    if group_n > 8:
                        break

                # Time 1 Data for building STKDE models : 1 Month
                for group in predict_groups:
                    predict_groups[group]['t1_data'] = \
                        df[df['date'].apply(
                            lambda x:
                            predict_groups[group]['t1_data'][0]
                            <= x.date() <=
                            predict_groups[group]['t1_data'][-1]
                        )
                        ]
                # Time 2 Data for Prediction            : 1 Week
                for group in predict_groups:
                    predict_groups[group]['t2_data'] = \
                        df[df['date'].apply(
                            lambda x:
                            predict_groups[group]['t2_data'][0]
                            <= x.date() <=
                            predict_groups[group]['t2_data'][-1]
                        )
                        ]
                return df, X, y, predict_groups
            return df, X, y
        return df


if __name__ == '__main__':
    pass
