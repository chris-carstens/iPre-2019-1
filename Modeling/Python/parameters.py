# parameters:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-07-12

import numpy as np
import pandas as pd

bw = np.array([1577.681, 1167.16, 35.549])
# bw_2 = np.array([1025.263, 1045.335, 51.045])   # menos usual

if __name__ == "__main__":
    df = pd.DataFrame([1, 2, 3, 4, 5])
    print(df.sample(n=3, replace=False, random_state=1))
