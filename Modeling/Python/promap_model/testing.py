import numpy as np

x, y = np.mgrid[0:5,0:5]

print(x)

mayor = x>2

print(mayor)

b = np.sum(mayor)
print(b)

training_matrix = np.full((5, 5), False)
print(training_matrix)

print(mayor*training_matrix)