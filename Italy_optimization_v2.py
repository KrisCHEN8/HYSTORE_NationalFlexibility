import os
import pandas as pd
from predictive_optimization_v2 import PredictiveOptimizerCVXPY  # noqa: E501
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

solver = 'CVXPY'

coutry = 'IT'      # Change with other coutries' abbreviations, e.g. ESP, SE, AUT  # noqa: E501
pickle_path = './national_zones/pickled_df'
df_agg = pd.read_pickle(os.path.join(pickle_path, f'df_agg_{coutry}.pkl'))
df_emission = pd.read_pickle(os.path.join(pickle_path, 'emission.pkl'))

df_demand = pd.read_pickle(os.path.join(pickle_path, 'df_demand.pkl'))


time_series = pd.date_range(start='2022-01-01 00:00:00', periods=8760, freq='1h')  # noqa: E501

# Input data
D_H = df_demand['Italy_heating_kWh'].values * 0.001  # be consistant with the country in line 7 (MWh)  # noqa: E501
D_C = df_demand['Italy_cooling_kWh'].values * 0.001  # be consistant with the country in line 7 (MWh)  # noqa: E501


df_demand.index = time_series

df_agg.index = time_series

COP_df = pd.read_pickle(os.path.join(pickle_path, 'COP_2022_df.pkl'))
EER_df = pd.read_pickle(os.path.join(pickle_path, 'EER_2022_df.pkl'))

COP_df.index = time_series
EER_df.index = time_series

hours = 1

heating_season = df_demand[df_demand['Italy_heating_kWh'] > 0]
avg_heating_demand = heating_season['Italy_heating_kWh'].mean() * 0.001
mean_COP = COP_df[coutry].mean()
thermal_capacity_heating = avg_heating_demand * mean_COP

cooling_season = df_demand[df_demand['Italy_cooling_kWh'] > 0]
avg_cooling_demand = heating_season['Italy_cooling_kWh'].mean() * 0.001
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

if solver == 'PSO':
    print('Pyomo is not available.')

elif solver == 'CVXPY':
    lambda_value = 0
    # Cm_ave
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, df_emission, 12, COP_df[coutry], EER_df[coutry], Cm_dict_ave, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1], lambda_value)[0]
    date_range = pd.date_range(start=time_series[0], periods=len(df_results), freq='1h')
    df_results.index = date_range
    df_results = df_results.loc[:, [
                'x_PCM_h',
                'y_PCM_h',
                'x_PCM_c',
                'y_PCM_c',
                'SoC_PCM_h',
                'SoC_PCM_c',
                'surplus',
                'D_H',
                'D_C',
                'x_TCM_h',
                'y_TCM_h',
                'x_TCM_c',
                'y_TCM_c',
                'SoC_TCM_h',
                'SoC_TCM_c'
    ]]
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_V2/IT/results_IT_aveCm_V2_{hours}_lambda{lambda_value}.xlsx', index=True)   # noqa: E501

    # Cm_70%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, df_emission, 12, COP_df[coutry], EER_df[coutry], Cm_dict_70p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1], lambda_value)[0]
    df_results.index = date_range
    df_results = df_results.loc[:, [
                'x_PCM_h',
                'y_PCM_h',
                'x_PCM_c',
                'y_PCM_c',
                'SoC_PCM_h',
                'SoC_PCM_c',
                'surplus',
                'D_H',
                'D_C',
                'x_TCM_h',
                'y_TCM_h',
                'x_TCM_c',
                'y_TCM_c',
                'SoC_TCM_h',
                'SoC_TCM_c'
    ]]
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_V2/IT/results_IT_70PCm_V2_{hours}_lambda{lambda_value}.xlsx', index=True)   # noqa: E501

    # Cm_50%
    optimizer = PredictiveOptimizerCVXPY(D_H, D_C, df_agg, df_emission, 12, COP_df[coutry], EER_df[coutry], Cm_dict_50p, 'surplus_RES')  # noqa: E501
    df_results = optimizer.opt(time_series[0], time_series[-1], lambda_value)[0]
    df_results.index = date_range
    df_results = df_results.loc[:, [
                'x_PCM_h',
                'y_PCM_h',
                'x_PCM_c',
                'y_PCM_c',
                'SoC_PCM_h',
                'SoC_PCM_c',
                'surplus',
                'D_H',
                'D_C',
                'x_TCM_h',
                'y_TCM_h',
                'x_TCM_c',
                'y_TCM_c',
                'SoC_TCM_h',
                'SoC_TCM_c'
    ]]
    df_results['actual_load'] = df_agg['Actual load'].values
    heating = - df_results['x_TCM_h'] - df_results['x_PCM_h']  # noqa: E501
    cooling = - df_results['x_TCM_c'] - df_results['x_PCM_c']  # noqa: E501
    df_results['modified_load'] = df_results['actual_load'] + heating + cooling
    df_results['surplus_optimized'] = df_results['surplus'] - (df_results['y_TCM_h'] + df_results['y_PCM_h'] + df_results['y_TCM_c'] + df_results['y_PCM_c'])  # noqa: E501
    df_results.to_excel(f'./res_V2/IT/results_IT_50PCm_V2_{hours}_lambda{lambda_value}.xlsx', index=True)   # noqa: E501
