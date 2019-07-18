# Time1_incidents:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-07-18

from calendar import monthrange

# Oct - Nov - Dic

w_day_oct, days_oct = monthrange(2016, 10)
w_day_nov, days_nov = monthrange(2016, 11)
w_day_dic, days_dic = monthrange(2016, 12)

days_oct_nov_dic = [i for i in range(1, days_oct + 1)] + \
                   [i for i in range(1, days_nov + 1)] + \
                   [i for i in range(1, days_dic + 1)]

predict_groups = {
    'group_1': {'t1_data': [], 't2_data': []},
    'group_2': {'t1_data': [], 't2_data': []},
    'group_3': {'t1_data': [], 't2_data': []},
    'group_4': {'t1_data': [], 't2_data': []},
    'group_5': {'t1_data': [], 't2_data': []},
    'group_6': {'t1_data': [], 't2_data': []},
    'group_7': {'t1_data': [], 't2_data': []},
    'group_8': {'t1_data': [], 't2_data': []}
}

# Time 1 Data for building STKDE models : 1 Month

group_n = 1
for i in range(1, len(days_oct_nov_dic))[::7]:
    predict_groups[f"group_{group_n}"]['t1_data'] = \
        days_oct_nov_dic[i - 1:i - 1 + days_oct]

    group_n += 1
    if group_n > 8:
        break

# Time 2 Data for Prediction            : 1 Week

