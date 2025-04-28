import os
import pandas as pd
from predictive_optimization import PredictiveOptimizerCVXPY  # noqa: E501
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

solver = 'CVXPY'

coutry = 'SE'      # Change with other coutries' abbreviations, e.g. ESP, SE, AUT  # noqa: E501
pickle_path = './national_zones/pickled_df'
df_agg = pd.read_pickle(os.path.join(pickle_path, f'df_agg_{coutry}.pkl'))
df_emission = pd.read_pickle(os.path.join(pickle_path, f'emission_{coutry}_2022.pkl'))
df_demand = pd.read_pickle(os.path.join(pickle_path, 'df_demand.pkl'))

time_series = pd.date_range(start='2022-01-01 00:00:00', periods=8760, freq='1h')  # noqa: E501

# Input data
D_H = df_demand['Sweden_heating_kWh'].values * 0.001  # be consistant with the country in line 7  # noqa: E501
D_C = df_demand['Sweden_cooling_kWh'].values * 0.001  # be consistant with the country in line 7  # noqa: E501


df_demand.index = time_series
df_emission.index = time_series
df_agg.index = time_series

COP_df = pd.read_pickle(os.path.join(pickle_path, 'COP_2022_df.pkl'))
EER_df = pd.read_pickle(os.path.join(pickle_path, 'EER_2022_df.pkl'))

COP_df.index = time_series
EER_df.index = time_series

hours = 1

heating_season = df_demand[df_demand['Sweden_heating_kWh'] > 0]
avg_heating_demand = heating_season['Sweden_heating_kWh'].mean() * 0.001
mean_COP = COP_df[coutry].mean()
thermal_capacity_heating = avg_heating_demand * mean_COP

cooling_season = df_demand[df_demand['Sweden_cooling_kWh'] > 0]
avg_cooling_demand = heating_season['Sweden_cooling_kWh'].mean() * 0.001
mean_EER = EER_df[coutry].mean()
thermal_capacity_cooling = avg_cooling_demand * mean_EER

Cm_dict_ave = {
    'Cm_h_PCM': thermal_capacity_heating * hours,    # MWh
    'Cm_c_PCM': thermal_capacity_cooling * hours,    # MWh
    'Cm_h_TCM': thermal_capacity_heating * hours,    # MWh
    'Cm_c_TCM': thermal_capacity_cooling * hours     # MWh
}

Cm_dict_70p = {
    'Cm_h_PCM': thermal_capacity_heating * hours,   # MWh
    'Cm_c_PCM': thermal_capacity_cooling * hours,   # MWh
    'Cm_h_TCM': thermal_capacity_heating * hours,   # MWh
    'Cm_c_TCM': thermal_capacity_cooling * hours   # MWh
}

Cm_dict_50p = {
    'Cm_h_PCM': thermal_capacity_heating * hours,   # MWh
    'Cm_c_PCM': thermal_capacity_cooling * hours,   # MWh
    'Cm_h_TCM': thermal_capacity_heating * hours,   # MWh
    'Cm_c_TCM': thermal_capacity_cooling * hours    # MWh
}

lambda_value = 1

if solver == 'Pyomo':
    print('Pyomo not available for now.')

elif solver == 'CVXPY':
    # Cm_ave
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, df_emission, 12, COP_df['SE'], EER_df['SE'], Cm_dict_ave, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1], lambda_value)
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_multiobj/SE/results_SE_aveCm_{hours}_lambda{lambda_value}.xlsx', index=True)

    # Cm_70%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, df_emission, 12, COP_df['SE'], EER_df['SE'], Cm_dict_70p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1], lambda_value)
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_multiobj/SE/results_SE_70PCm_{hours}_lambda{lambda_value}.xlsx', index=True)

    # Cm_50%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, df_emission, 12, COP_df['SE'], EER_df['SE'], Cm_dict_50p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1], lambda_value)
    df_results.index = time_series
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_multiobj/SE/results_SE_50PCm_{hours}_lambda{lambda_value}.xlsx', index=True)
