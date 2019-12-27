"""
mpl_plot_2.py

iPre - Big Data para Criminología
Created by Mauro S. Mendoza Elguera at 20-12-19
Pontifical Catholic University of Chile
"""

import pandas as pd
import matplotlib.pyplot as plt

from calendar import month_name

df = pd.read_pickle('rfc.pkl')
data = [aux_df.sum()['r_importance'] for aux_df in [df[df['features'].isin(
    [(f'Incidents_{i}', month_name[j]) for j in range(1, 10)])]
                                                    for i in range(0, 8)]]
index = [i for i in range(0, 8)]

ax = pd.DataFrame(data=data, index=index, columns=['r_importance']) \
    .plot.bar(y='r_importance', color='black', width=0.25, rot=0,
              legend=None)

for i in range(0, 8):
    plt.text(x=i - 0.3, y=data[i] + 0.02 * max(data), s=f'{data[i]:.3f}')

plt.xlabel("Layers",
           fontdict={'fontsize': 12.5,
                     'fontweight': 'bold',
                     'family': 'serif'},
           labelpad=10
           )
plt.ylabel("Relative Importance",
           fontdict={'fontsize': 12.5,
                     'fontweight': 'bold',
                     'family': 'serif'},
           labelpad=7.5
           )
plt.xticks(ticks=[i for i in range(0, 8)],
           labels=[f'{i}' for i in range(0, 8)])
plt.tick_params(axis='both', length=0, pad=8.5)  # Hide tick lines

ax.spines['top'].set_visible(False)  # Hide frame
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_color('lightgray')  # Frame color
ax.spines['left'].set_color('lightgray')

plt.show()
