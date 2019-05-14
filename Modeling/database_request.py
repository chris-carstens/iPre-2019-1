# part_1:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-05-14

import pandas as pd
from sodapy import Socrata

import credentials as cre

with Socrata(cre.socrata_domain,
             cre.API_KEY_S,
             username=cre.USERNAME_S,
             password=cre.PASSWORD_S) as client:
    query = """
select
    incidentnum,
    geocoded_column,
    date1,
    time1,
    x_coordinate,
    y_cordinate
where
    geocoded_column is not null
    and date1 is not null
    and time1 is not null
    and x_coordinate is not null
    and y_cordinate is not null
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

df = df[['x_coordinate', 'y_cordinate']]

df.rename(columns={'x_coordinate': 'x', 'y_cordinate': 'y'}, inplace=True)
