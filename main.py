"""
main.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 11-05-20
Pontifical Catholic University of Chile

"""

from predictivehp.models.models import *
from predictivehp.models.parameters import *


stkde = STKDE(n=1000, year='2017')
rfr = RForestRegressor(n=1000, year='2017', read_df=False, read_data=False)
pm = ProMap(n=150_000, year="2017", bw=bw, read_files=False)


if __name__ == '__main__':
    pass
