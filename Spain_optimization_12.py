import os
import pandas as pd
from predictive_optimization import PredictiveOptimizerCVXPY  # noqa: E501
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

solver = 'CVXPY'

coutry = 'ESP'      # Change with other coutries' abbreviations, e.g. ESP, SE, AUT  # noqa: E501
pickle_path = './national_zones/pickled_df'
df_agg = pd.read_pickle(os.path.join(pickle_path, f'df_agg_{coutry}.pkl'))

df_demand = pd.read_pickle(os.path.join(pickle_path, 'df_demand.pkl'))


time_series = pd.date_range(start='2022-01-01 00:00:00', periods=8760, freq='1h')  # noqa: E501

# Input data
D_H = df_demand['Spain_heating_kWh'].values * 0.001  # be consistant with the country in line 7 (MWh)  # noqa: E501
D_C = df_demand['Spain_cooling_kWh'].values * 0.001  # be consistant with the country in line 7 (MWh)  # noqa: E501


df_demand.index = time_series

df_agg.index = time_series

COP_df = pd.read_pickle(os.path.join(pickle_path, 'COP_2022_df.pkl'))
EER_df = pd.read_pickle(os.path.join(pickle_path, 'EER_2022_df.pkl'))

COP_df.index = time_series
EER_df.index = time_series

Cm_dict_ave = {
    'Cm_h': 308836.95 * 0.001 * 12,   # MWh
    'Cm_c': 146419.18 * 0.001 * 12    # MWh
}

Cm_dict_70p = {
    'Cm_h': 669284.33 * 0.001 * 12,   # MWh
    'Cm_c': 485496.01 * 0.001 * 12   # MWh
}

Cm_dict_50p = {
    'Cm_h': 478060.24 * 0.001 * 12,   # MWh
    'Cm_c': 346782.87 * 0.001 * 12   # MWh
}

if solver == 'PSO':
    print('Pyomo is not available.')

elif solver == 'CVXPY':
    # Cm_ave
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['ESP'], EER_df['ESP'], Cm_dict_ave, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel('./res_12/ESP/results_ESP_aveCm_12.xlsx', index=True)

    # Cm_70%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['ESP'], EER_df['ESP'], Cm_dict_70p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel('./res_12/ESP/results_ESP_70PCm_12.xlsx', index=True)

    # Cm_50%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, 12, COP_df['ESP'], EER_df['ESP'], Cm_dict_50p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1])
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel('./res_12/ESP/results_ESP_50PCm_12.xlsx', index=True)