# parameters:
# - Python version: 3.7
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-07-12

import numpy as np

from calendar import monthrange
from datetime import date



# Optimal Bandwidths

#bw = np.array([1577.681, 1167.16, 35.549])

bw = {'x': 1577.681,'y': 1167.16}

dallas_limits = {'x_min': -10804957.65128928, 'x_max': -10735466.29163222,
                 'y_min': 3840201.8325116523, 'y_max': 3900214.267184315}

bins = 100
# ------------------------------------------------------------------------------

# Oct - Nov - Dic

# w_day_oct, days_oct = monthrange(2016, 10)
# w_day_nov, days_nov = monthrange(2016, 11)
# w_day_dic, days_dic = monthrange(2016, 12)

# days_oct_nov_dic = [date(2016, 10, i) for i in range(1, days_oct + 1)] + \
#                    [date(2016, 11, i) for i in range(1, days_nov + 1)] + \
#                    [date(2016, 12, i) for i in range(1, days_dic + 1)]

# predict_groups = {
#     'group_1': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_2': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_3': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_4': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_5': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_6': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_7': {'t1_data': [], 't2_data': [], 'STKDE': None},
#     'group_8': {'t1_data': [], 't2_data': [], 'STKDE': None}
# }

# Time 1 Data for building STKDE models : 1 Month

# group_n = 1
# for i in range(1, len(days_oct_nov_dic))[::7]:
#     predict_groups[f"group_{group_n}"]['t1_data'] = \
#         days_oct_nov_dic[i - 1:i - 1 + days_oct]
#
#     group_n += 1
#     if group_n > 8:
#         break

# Time 2 Data for Prediction            : 1 Week

# group_n = 1
# for i in range(1, len(days_oct_nov_dic))[::7]:
#     predict_groups[f"group_{group_n}"]['t2_data'] = \
#         days_oct_nov_dic[i - 1 + days_oct:i - 1 + days_oct + 7]
#
#     group_n += 1
#     if group_n > 8:
#         break

# ------------------------------------------------------------------------------

# if __nme__ == "__main__":
#     for i in predict_groups['group_2']['t1_data']:
#         print(i)
