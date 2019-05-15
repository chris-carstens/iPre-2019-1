import numpy as np
from sklearn.metrics import confusion_matrix
from string import ascii_uppercase
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

y_verd = np.array([0,1,1])

y_pred = np.array([0,2,1])

conm = confusion_matrix(y_verd, y_pred)
print(conm)

columnas = ['Clase %s'%(i) for i in list(ascii_uppercase)[0:len(np.unique(y_pred))]]
df_cm = pd.DataFrame(conm,index=columnas, columns=columnas)

grafica = sns.heatmap(df_cm,cmap='Pastel1', annot=True)

grafica.set(xlabel = 'Verdadero', ylabel = 'Predicción')

plt.show()

