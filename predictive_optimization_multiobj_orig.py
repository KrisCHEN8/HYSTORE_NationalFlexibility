import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm  # optional, for a progress bar
import os
import matplotlib as mpl


class PredictiveOptimizerCVXPY:
    def __init__(self, D_H, D_C, df_simplified_calc, df_emission, horizon, COP, EER, Cm_dict, optimization_obj):
        self.Cm_TCM_h = Cm_dict['Cm_h_TCM']
        self.Cm_TCM_c = Cm_dict['Cm_c_TCM']
        self.Cm_PCM_h = Cm_dict['Cm_h_PCM']
        self.Cm_PCM_c = Cm_dict['Cm_c_PCM']
        self.eta_TCM_c_dis = 0.5
        self.eta_TCM_h_dis = 1.1
        self.eta_TCM_ch = 1.0
        self.eta_PCM = 0.7
        self.SoC_TCM_max = 80.0
        self.SoC_TCM_min = 15.0
        self.SoC_PCM_max = 75.0
        self.SoC_PCM_min = 20.0
        self.f_loss_PCM = (self.SoC_PCM_max - self.SoC_PCM_min) / 24.0
        self.D_H = D_H
        self.D_C = D_C
        self.df = df_simplified_calc.copy()
        self.T = horizon
        self.dt = 1
        self.alpha = 2.5
        self.obj = optimization_obj
        self.k = -7
        self.df['D_H'] = self.D_H
        self.df['D_C'] = self.D_C
        self.emission = df_emission.copy()
        self.COP = COP
        self.EER = EER
        self.SoC_PCM_h_init = [self.SoC_PCM_min]
        self.SoC_PCM_c_init = [self.SoC_PCM_min]
        self.SoC_TCM_h_init = [self.SoC_TCM_min]
        self.SoC_TCM_c_init = [self.SoC_TCM_min]

    def opt(self, t_start, t_end, lambda_value):
        df_results = pd.DataFrame()

        while t_start < t_end:
            time_series = pd.date_range(start=t_start, periods=self.T, freq='1h')

            # PCM Variables
            SoC_PCM_h_init = self.SoC_PCM_h_init[-1]
            SoC_PCM_c_init = self.SoC_PCM_c_init[-1]

            PCM_disc_c = cp.Variable(self.T, nonneg=True)
            PCM_disc_h = cp.Variable(self.T, nonneg=True)
            PCM_char_c = cp.Variable(self.T, nonneg=True)
            PCM_char_h = cp.Variable(self.T, nonneg=True)
            SoC_PCM_h = cp.Variable(self.T + 1, pos=True)
            SoC_PCM_c = cp.Variable(self.T + 1, pos=True)
            u_h = cp.Variable(self.T, boolean=True)
            u_c = cp.Variable(self.T, boolean=True)
            allocated_surplus_h = cp.Variable(self.T, nonneg=True)
            allocated_surplus_c = cp.Variable(self.T, nonneg=True)
            epsilon_h = cp.Variable(self.T)
            epsilon_c = cp.Variable(self.T)

            constraints = [
                SoC_PCM_h[0] == SoC_PCM_h_init,
                SoC_PCM_c[0] == SoC_PCM_c_init
            ]

            eer = self.EER[time_series].values
            cop = self.COP[time_series].values
            co2 = self.emission.loc[time_series, 'emission_factor'].values

            for t in range(self.T):
                constraints += [
                    epsilon_c[t] >= cp.exp(self.k * (SoC_PCM_c[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)),
                    epsilon_h[t] >= cp.exp(self.k * (SoC_PCM_h[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)),
                    SoC_PCM_h[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_h[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_c[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_c[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_c[t + 1] == SoC_PCM_c[t] + 100 * ((PCM_char_c[t] * eer[t] - PCM_disc_c[t] * eer[t]) / self.Cm_PCM_c) * self.eta_PCM - self.f_loss_PCM * (1 - epsilon_c[t]),
                    SoC_PCM_h[t + 1] == SoC_PCM_h[t] + 100 * ((PCM_char_h[t] * cop[t] - PCM_disc_h[t] * cop[t]) / self.Cm_PCM_h) * self.eta_PCM - self.f_loss_PCM * (1 - epsilon_h[t])
                ]

            surplus = self.df.loc[time_series, self.obj].values
            d_h = self.df.loc[time_series, 'D_H'].values
            d_c = self.df.loc[time_series, 'D_C'].values

            cumulative_future_demand_c = [sum(self.df.loc[time_series[t]: time_series[t] + timedelta(hours=self.T), 'D_C']) for t in range(self.T)]
            cumulative_future_demand_h = [sum(self.df.loc[time_series[t]: time_series[t] + timedelta(hours=self.T), 'D_H']) for t in range(self.T)]

            for t in range(self.T):
                constraints += [
                    allocated_surplus_h[t] + allocated_surplus_c[t] <= surplus[t],
                    allocated_surplus_h[t] <= cumulative_future_demand_h[t] * u_h[t],
                    allocated_surplus_c[t] <= cumulative_future_demand_c[t] * u_c[t],
                    PCM_char_h[t] <= allocated_surplus_h[t],
                    PCM_char_c[t] <= allocated_surplus_c[t],
                    PCM_disc_c[t] <= d_c[t] * (1 - u_c[t]),
                    PCM_disc_h[t] <= d_h[t] * (1 - u_h[t])
                ]

            cooling_weight = []
            heating_weight = []

            for t in range(self.T):
                cooling_weight.append(np.maximum(1, d_c[t] / (d_h[t] + 1e-4)))  # Weight for cooling  # noqa: E501
                heating_weight.append(np.maximum(1, d_h[t] / (d_c[t] + 1e-4)))  # Weight for heating  # noqa: E501

            f_demand_weight = cp.sum(d_h - cp.multiply(PCM_disc_h, heating_weight)) + cp.sum(d_c - cp.multiply(PCM_disc_c, cooling_weight))
            f_surplus_pcm = cp.sum(surplus - cp.multiply(PCM_char_h, heating_weight) - cp.multiply(PCM_char_c, cooling_weight)) + f_demand_weight
            f_carbon_pcm = cp.sum(cp.multiply(PCM_disc_h + PCM_disc_c, co2))

            objective = cp.Minimize(f_surplus_pcm - lambda_value * f_carbon_pcm + 1e9 * (cp.sum(epsilon_c) + cp.sum(epsilon_h)))
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)

            results = {
                'x_PCM_h': PCM_disc_h.value,
                'y_PCM_h': PCM_char_h.value,
                'x_PCM_c': PCM_disc_c.value,
                'y_PCM_c': PCM_char_c.value,
                'SoC_PCM_h': SoC_PCM_h.value[:-1],
                'SoC_PCM_c': SoC_PCM_c.value[:-1],
                'surplus': surplus,
                'D_H': d_h,
                'D_C': d_c,
                'epsilon_c': epsilon_c.value,
                'epsilon_h': epsilon_h.value
            }

            df_results_pcm = pd.DataFrame(results)
            self.SoC_PCM_h_init.append(SoC_PCM_h.value[-1])
            self.SoC_PCM_c_init.append(SoC_PCM_c.value[-1])

            print(f'{t_start} PCM optimization finished')

            # TCM Optimization
            SoC_TCM_h_init = self.SoC_TCM_h_init[-1]
            SoC_TCM_c_init = self.SoC_TCM_c_init[-1]

            TCM_disc_c = cp.Variable(self.T, nonneg=True)
            TCM_disc_h = cp.Variable(self.T, nonneg=True)
            TCM_char_c = cp.Variable(self.T, nonneg=True)
            TCM_char_h = cp.Variable(self.T, nonneg=True)
            SoC_TCM_h = cp.Variable(self.T + 1, pos=True)
            SoC_TCM_c = cp.Variable(self.T + 1, pos=True)
            u_h = cp.Variable(self.T, boolean=True)
            u_c = cp.Variable(self.T, boolean=True)
            allocated_surplus_h = cp.Variable(self.T, nonneg=True)
            allocated_surplus_c = cp.Variable(self.T, nonneg=True)

            constraints = [
                SoC_TCM_h[0] == SoC_TCM_h_init,
                SoC_TCM_c[0] == SoC_TCM_c_init
            ]

            eer = self.EER[time_series].values
            cop = self.COP[time_series].values
            co2 = self.emission.loc[time_series, 'emission_factor'].values

            surplus = np.round(self.df.loc[time_series, self.obj].values - (df_results_pcm['y_PCM_c'].values + df_results_pcm['y_PCM_h'].values), 2)
            d_h = self.df.loc[time_series, 'D_H'].values - df_results_pcm['x_PCM_h'].values
            d_c = self.df.loc[time_series, 'D_C'].values - df_results_pcm['x_PCM_c'].values

            cumulative_future_demand_c = [sum(self.df.loc[time_series[t]: time_series[t] + timedelta(hours=self.T), 'D_C']) for t in range(self.T)]
            cumulative_future_demand_h = [sum(self.df.loc[time_series[t]: time_series[t] + timedelta(hours=self.T), 'D_H']) for t in range(self.T)]

            for t in range(self.T):
                constraints += [
                    SoC_TCM_h[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_h[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_c[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] == SoC_TCM_c[t] + 100 * (((TCM_char_c[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_c[t] * self.eta_TCM_c_dis) * eer[t]) / self.Cm_TCM_c),
                    SoC_TCM_h[t + 1] == SoC_TCM_h[t] + 100 * (((TCM_char_h[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_h[t] * self.eta_TCM_h_dis) * cop[t]) / self.Cm_TCM_h),
                    allocated_surplus_h[t] + allocated_surplus_c[t] <= surplus[t],
                    allocated_surplus_h[t] <= cumulative_future_demand_h[t] * u_h[t],
                    allocated_surplus_c[t] <= cumulative_future_demand_c[t] * u_c[t],
                    TCM_char_h[t] <= allocated_surplus_h[t],
                    TCM_char_c[t] <= allocated_surplus_c[t],
                    TCM_disc_c[t] <= d_c[t] * (1 - u_c[t]),
                    TCM_disc_h[t] <= d_h[t] * (1 - u_h[t])
                ]

            cooling_weight = []
            heating_weight = []

            for t in range(self.T):
                cooling_weight.append(np.maximum(1, d_c[t] / (d_h[t] + 1e-4)))  # Weight for cooling  # noqa: E501
                heating_weight.append(np.maximum(1, d_h[t] / (d_c[t] + 1e-4)))  # Weight for heating  # noqa: E501

            f_demand_weight = cp.sum(d_h - cp.multiply(TCM_disc_h, heating_weight)) + cp.sum(d_c - cp.multiply(TCM_disc_c, cooling_weight))
            f_surplus_tcm = cp.sum(surplus - cp.multiply(TCM_char_h, heating_weight) - cp.multiply(TCM_char_c, cooling_weight)) + f_demand_weight
            f_carbon_tcm = cp.sum(cp.multiply(TCM_disc_h + TCM_disc_c, co2))

            objective = cp.Minimize(f_surplus_tcm - lambda_value * f_carbon_tcm)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)

            results = {
                'x_TCM_h': TCM_disc_h.value,
                'y_TCM_h': TCM_char_h.value,
                'x_TCM_c': TCM_disc_c.value,
                'y_TCM_c': TCM_char_c.value,
                'SoC_TCM_h': SoC_TCM_h.value[:-1],
                'SoC_TCM_c': SoC_TCM_c.value[:-1],
            }

            df_results_tcm = pd.DataFrame(results)
            new_rows = pd.concat([df_results_pcm, df_results_tcm], axis=1)
            df_results = pd.concat([df_results, new_rows], ignore_index=True, axis=0)

            self.SoC_TCM_h_init.append(SoC_TCM_h.value[-1])
            self.SoC_TCM_c_init.append(SoC_TCM_c.value[-1])

            print(f'{t_start} TCM optimization finished')

            t_start += timedelta(hours=self.T)

        return df_results


if __name__ == '__main__':
    # — parameters —
    lambdas = np.linspace(0, 1.5, 6)  # from 0 (surplus only) to 10
    T = 12
    results = []

    coutry = 'IT'      # Change with other countries' abbreviations, e.g. ESP, SE, AUT
    pickle_path = './national_zones/pickled_df'
    df_agg = pd.read_pickle(os.path.join(pickle_path, f'df_agg_{coutry}.pkl'))
    df_demand = pd.read_pickle(os.path.join(pickle_path, 'df_demand.pkl'))
    df_emission = pd.read_pickle(os.path.join(pickle_path, f'emission_{coutry}_2022.pkl'))

    time_series = pd.date_range(start='2022-01-01 00:00:00', periods=8760, freq='1h')
    df_demand.index = df_emission.index = df_agg.index = time_series

    COP_df = pd.read_pickle(os.path.join(pickle_path, 'COP_2022_df.pkl'))
    EER_df = pd.read_pickle(os.path.join(pickle_path, 'EER_2022_df.pkl'))
    COP_df.index = EER_df.index = time_series

    # compute thermal capacities
    heating_season = df_demand[df_demand['Italy_heating_kWh'] > 0]
    avg_heating_demand = heating_season['Italy_heating_kWh'].mean() * 0.001
    mean_COP = COP_df[coutry].mean()
    thermal_capacity_heating = avg_heating_demand * mean_COP

    cooling_season = df_demand[df_demand['Italy_cooling_kWh'] > 0]
    avg_cooling_demand = heating_season['Italy_cooling_kWh'].mean() * 0.001
    mean_EER = EER_df[coutry].mean()
    thermal_capacity_cooling = avg_cooling_demand * mean_EER

    hours = 1
    Cm_dict_ave = {
        'Cm_h_PCM': thermal_capacity_heating * hours,
        'Cm_c_PCM': thermal_capacity_cooling * hours,
        'Cm_h_TCM': thermal_capacity_heating * hours,
        'Cm_c_TCM': thermal_capacity_cooling * hours
    }

    # Sweep over lambdas
    for λ in tqdm(lambdas, desc="Sweeping λ"):
        optimizer = PredictiveOptimizerCVXPY(
            D_H=df_demand['Italy_heating_kWh'].values * 0.001,
            D_C=df_demand['Italy_cooling_kWh'].values * 0.001,
            df_simplified_calc=df_agg,
            df_emission=df_emission,
            horizon=T,
            COP=COP_df[coutry],
            EER=EER_df[coutry],
            Cm_dict=Cm_dict_ave,
            optimization_obj='surplus_RES'
        )
        df_res = optimizer.opt(time_series[0], time_series[-1], lambda_value=λ)
        df_res.index = time_series
        df_res['actual_load'] = df_agg['Actual load'].values
        heating = - df_res['x_TCM_h'] - df_res['x_PCM_h']  # noqa: E501
        cooling = - df_res['x_TCM_c'] - df_res['x_PCM_c']  # noqa: E501
        df_res['modified_load'] = df_res['actual_load'] + heating + cooling
        df_res['surplus_optimized'] = df_res['surplus'] - (df_res['y_TCM_h'] + df_res['y_PCM_h'] + df_res['y_TCM_c'] + df_res['y_PCM_c'])  # noqa: E501

        # compute CO2 and surplus
        co2 = ((df_res[['x_PCM_h','x_PCM_c','x_TCM_h','x_TCM_c']].sum(axis=1) *
                df_emission['emission_factor']).sum())
        surplus = df_res['surplus_optimized'].sum()
        results.append((λ, co2, surplus))

    # --- build results DataFrame and save ---
    df_pareto = pd.DataFrame(results, columns=['lambda', 'co2', 'surplus'])
    df_pareto.to_csv('./res_multiobj/pareto_results.csv', index=False)
    print("Saved Pareto results to pareto_results.csv")

    # --- plot with continuous colormap for λ ---
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=lambdas.min(), vmax=lambdas.max())

    plt.figure(figsize=(10, 10))
    sc = plt.scatter(
        df_pareto['co2'],
        df_pareto['surplus'],
        c=df_pareto['lambda'],
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolor='k'
    )
    plt.ylabel('Total Surplus After Storage [MWh]')
    plt.xlabel('Total CO₂ Emissions reduction [kgCO₂ eq]')
    plt.title('Pareto Front results')
    plt.grid(True, linestyle='--', alpha=0.6)

    cbar = plt.colorbar(sc)
    cbar.set_label('λ values')

    plt.tight_layout()
    plt.savefig('./res_multiobj/pareto_front_within1.png', dpi=400)
    plt.show()
