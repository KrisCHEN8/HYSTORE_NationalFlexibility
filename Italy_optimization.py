import os
import pandas as pd
from predictive_optimization import PredictiveOptimizerCVXPY  # noqa: E501
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

solver = 'CVXPY'

coutry = 'IT'      # Change with other coutries' abbreviations, e.g. ESP, SE, AUT  # noqa: E501
pickle_path = './national_zones/pickled_df'
df_agg = pd.read_pickle(os.path.join(pickle_path, f'df_agg_{coutry}.pkl'))

df_demand = pd.read_pickle(os.path.join(pickle_path, 'df_demand.pkl'))


time_series = pd.date_range(start='2022-01-01 00:00:00', periods=8760, freq='1h')  # noqa: E501

# Input data
D_H = df_demand['Italy_heating_kWh'].values * 1e-6  # be consistant with the country in line 7  # noqa: E501
D_C = df_demand['Italy_cooling_kWh'].values * 1e-6  # be consistant with the country in line 7  # noqa: E501


df_demand.index = time_series

df_agg.index = time_series

COP_df = pd.read_pickle(os.path.join(pickle_path, 'COP_2022_df.pkl'))
EER_df = pd.read_pickle(os.path.join(pickle_path, 'EER_2022_df.pkl'))

COP_df.index = time_series
EER_df.index = time_series

COP_df = COP_df.applymap(lambda x: 0 if isinstance(x, str) else x)
EER_df = EER_df.applymap(lambda x: 0 if isinstance(x, str) else x)

Cm_dict_ave = {
    'Cm_h': 254761.18 * 1e-6,   # GWh
    'Cm_c': 815840.28 * 1e-6    # GWh
}

Cm_dict_70p = {
    'Cm_h': 521523.24 * 1e-6,   # GWh
    'Cm_c': 1989612.33 * 1e-6   # GWh
}

Cm_dict_50p = {
    'Cm_h': 372516.60 * 1e-6,   # GWh
    'Cm_c': 1421151.66 * 1e-6   # GWh
}

if solver == 'Pyomo':
    print('Pyomo not available for now.')

elif solver == 'CVXPY':
    # Cm_ave
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['IT'], EER_df['IT'], Cm_dict_ave, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results.to_excel('./res/IT/results_IT_aveCm.xlsx', index=True)

    # Cm_70%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['IT'], EER_df['IT'], Cm_dict_70p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results.to_excel('./res/IT/results_IT_70PCm.xlsx', index=True)

    # Cm_50%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['IT'], EER_df['IT'], Cm_dict_50p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results.to_excel('./res/IT/results_IT_50PCm.xlsx', index=True)
