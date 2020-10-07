import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])

aux = a-3.5
print(aux)
print(np.abs(aux))

print(b[np.argmin(np.abs(aux))])



