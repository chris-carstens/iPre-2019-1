import numpy as np
import pickle
from tqdm.auto import tqdm
import pandas as pd
import math
import shutil

# training_data = pd.read_pickle('training_data.pkl')
# testing_data = pd.read_pickle('testing_data.pkl')

# print(training_data)
# print()
# print(testing_data)

a = np.mgrid[0:5,0:5]



x_min = 0
x_max = 11
y_min = 0
y_max = 10


dy = y_max - y_min
dx = x_max - x_min
print()


hx = 2
hy = 2

bins_x = round((x_max - x_min) / (hx))
bins_y = round((y_max - y_min) / (hy))






x, y = np.mgrid[x_min + hx/2:x_max - hx/2:bins_x *1j,
                 y_min + hy/2:y_max - hy/2:bins_y *1j]



print()



d = {'x': [4, 3], 'y': [1, 4], 't': [1, 12]}







a = [1,2,3]

c = 0.0

b = "1/"+str(len(a))

c = b

print(c)



