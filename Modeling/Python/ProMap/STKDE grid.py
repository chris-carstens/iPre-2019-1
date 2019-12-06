# STKDE grid:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-07-14

import numpy as np
import parameters as params

from pyevtk.hl import gridToVTK
from time import time
from STKDE2 import Framework

st = time()

bins = 100

dallas_stkde = Framework(n=150000,
                     year="2016",
                     bw=params.bw)

# Coordinates

print("\nCreating 3D grid...")

x, y, t = np.mgrid[
          np.array(dallas_stkde.testing_data[['x']]).min():
          np.array(dallas_stkde.testing_data[['x']]).max():bins * 1j,
          np.array(dallas_stkde.testing_data[['y']]).min():
          np.array(dallas_stkde.testing_data[['y']]).max():bins * 1j,
          np.array(dallas_stkde.testing_data[['y_day']]).min():
          np.array(dallas_stkde.testing_data[['y_day']]).max():60 * 1j
          ]

print("\nEstimating densities...")

d = dallas_stkde.kde.pdf(
        np.vstack([
            x.flatten(),
            y.flatten(),
            t.flatten()
        ])).reshape((100, 100, 60))

print("\nExporting 3D grid...")

gridToVTK("STKDE grid",
          x, y, t,
          pointData={"density": d,
                     "y_day": t})

print(f"\nTotal time: {round((time() - st) / 60, 3)} min")
