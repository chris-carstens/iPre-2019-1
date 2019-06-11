# Histogram:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-06-11

from database_request import df
import datetime
import numpy as np
import matplotlib.pyplot as plt

months = [i for i in range(1, 13)]

fig, ax = plt.subplots()
bins = np.arange(1, 14)
ax.hist(df["date"].apply(lambda x: x.month),
        bins=bins,
        edgecolor="k",
        align='left')
ax.set_xticks(bins[:-1])
ax.set_xticklabels(
        [datetime.date(1900, i, 1).strftime('%b') for i in bins[:-1]])

n = 50000
plt.title(f"Database Request, n = {n}",
          fontdict={'fontsize': 20,
                    'fontweight': 'bold'},
          pad=20)
plt.savefig(f"histogram_{n}.pdf", format='pdf')
plt.show()
