# part_1:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-05-14

import pandas as pd
from datetime import datetime
from sodapy import Socrata

import credentials as cre

pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

with Socrata(cre.socrata_domain,
             cre.API_KEY_S,
             username=cre.USERNAME_S,
             password=cre.PASSWORD_S) as client:
    query = \
        """
    select
        incidentnum,
        geocoded_column,
        year1,
        date1,
        time1,
        x_coordinate,
        y_cordinate
    where
        geocoded_column is not null
        and year1 = 2014
        and date1 is not null
        and time1 is not null
        and x_coordinate is not null
        and y_cordinate is not null
    order by date1
    limit
        1000
    """  #  530000 max. 11/04

    results = client.get(cre.socrata_dataset_identifier,
                         query=query,
                         content_type='json')
    df = pd.DataFrame.from_records(results)

# DB Cleaning & Formatting


df.loc[:, 'x_coordinate'] = df['x_coordinate'].apply(lambda x: float(x))
df.loc[:, 'y_cordinate'] = df['y_cordinate'].apply(lambda x: float(x))
df.loc[:, 'date1'] = df['date1'].apply(
        lambda x: datetime.strptime(x.split('T')[0], '%Y-%m-%d'))

df_t = df[['x_coordinate', 'y_cordinate', 'date1']]
df = df[['x_coordinate', 'y_cordinate']]

df.rename(columns={'x_coordinate': 'x',
                   'y_cordinate': 'y'},
          inplace=True)
df_t.rename(columns={'x_coordinate': 'x',
                     'y_cordinate': 'y',
                     'date1': 'date'},
            inplace=True)

print(df_t)
