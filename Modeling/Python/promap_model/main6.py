# main promap:
# - Python version: 3.7
# - Author: Francisco Tobar
# - Date: 2019-07-14


import params
from time import time
from promap import Promap

st = time()

dallas_promap = Promap(n=150_000,
                     year="2017",
                     bw=params.bw)


dallas_promap.plot_HR()
dallas_promap.plot_PAI()

print(f"\nTotal time: {round((time() - st) / 60, 3)} min")