"""
mpl_plot_3.py

Ploteo de un heatmap que muestra la importancia de los datos de acuerdo
a las capas/meses.

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 20-12-19
Pontifical Catholic University of Chile
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from calendar import month_name

df = pd.read_pickle('rfc.pkl')
df.set_index(keys='features', drop=True, inplace=True)

data = df.to_numpy().reshape(8, 9).T
columns = [i for i in range(0, 8)]
index = [f'{month_name[i]:.3s}' for i in range(1, 10)]

df = pd.DataFrame(data=data, index=index, columns=columns)

sbn.heatmap(data=df, annot=True, annot_kws={"fontsize": 9})

plt.xlabel("Layers",
           fontdict={'fontsize': 12.5,
                     'fontweight': 'bold',
                     'family': 'serif'},
           labelpad=10
           )
plt.ylabel("Months",
           fontdict={'fontsize': 12.5,
                     'fontweight': 'bold',
                     'family': 'serif'},
           labelpad=7.5
           )

plt.tick_params(axis='both', length=0, pad=8.5)
plt.yticks(rotation=0)

plt.show()
