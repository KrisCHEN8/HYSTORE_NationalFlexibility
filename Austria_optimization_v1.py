import os
import pandas as pd
from predictive_optimization_v1 import PredictiveOptimizerCVXPY  # noqa: E501
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

solver = 'CVXPY'

coutry = 'AUT'      # Change with other coutries' abbreviations, e.g. ESP, SE, AUT  # noqa: E501
pickle_path = './national_zones/pickled_df'
df_agg = pd.read_pickle(os.path.join(pickle_path, f'df_agg_{coutry}.pkl'))

df_demand = pd.read_pickle(os.path.join(pickle_path, 'df_demand.pkl'))


time_series = pd.date_range(start='2022-01-01 00:00:00', periods=8760, freq='1h')  # noqa: E501

# Input data
D_H = df_demand['Austria_heating_kWh'].values * 0.001  # be consistant with the country in line 7  # noqa: E501
D_C = df_demand['Austria_cooling_kWh'].values * 0.001  # be consistant with the country in line 7  # noqa: E501


df_demand.index = time_series

df_agg.index = time_series

COP_df = pd.read_pickle(os.path.join(pickle_path, 'COP_2022_df.pkl'))
EER_df = pd.read_pickle(os.path.join(pickle_path, 'EER_2022_df.pkl'))

COP_df.index = time_series
EER_df.index = time_series

hours = 12

Cm_dict_ave = {
    'Cm_h_PCM': 157664.92 * 0.001 * hours,   # MWh
    'Cm_c_PCM': 1138.73 * 0.001 * hours,    # MWh
    'Cm_h_TCM': 157664.92 * 0.001 * hours,   # MWh
    'Cm_c_TCM': 1138.73 * 0.001 * hours    # MWh
}

Cm_dict_70p = {
    'Cm_h_PCM': 255468.26 * 0.001 * hours,   # MWh
    'Cm_c_PCM': 2202.34 * 0.001 * hours,   # MWh
    'Cm_h_TCM': 255468.26 * 0.001 * hours,   # MWh
    'Cm_c_TCM': 2202.34 * 0.001 * hours   # MWh
}

Cm_dict_50p = {
    'Cm_h_PCM': 182477.33 * 0.001 * hours,   # MWh
    'Cm_c_PCM': 1573.10 * 0.001 * hours,   # MWh
    'Cm_h_TCM': 182477.33 * 0.001 * hours,   # MWh
    'Cm_c_TCM': 1573.10 * 0.001 * hours   # MWh
}

if solver == 'Pyomo':
    print('Pyomo not available for now.')

elif solver == 'CVXPY':
    # Cm_ave
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['AUT'], EER_df['AUT'], Cm_dict_ave, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_v1/AUT/results_AUT_aveCm_V1_{hours}.xlsx', index=True)

    # Cm_70%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['AUT'], EER_df['AUT'], Cm_dict_70p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_v1/AUT/results_AUT_70PCm_V1_{hours}.xlsx', index=True)

    # Cm_50%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['AUT'], EER_df['AUT'], Cm_dict_50p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_v1/AUT/results_AUT_50PCm_V1_{hours}.xlsx', index=True)
