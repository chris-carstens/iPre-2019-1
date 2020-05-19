"""
main.py
Python Version: 3.8.1

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 11-05-20
Pontifical Catholic University of Chile

"""

from predictivehp.models.models import *

stkde = STKDE(n=1000, year='2017')
rfr = RForestRegressor()
pm = ProMap()

if __name__ == '__main__':
    pass
