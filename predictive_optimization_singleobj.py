import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import timedelta


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

    def enforce_min_duration(self, z, duration, T):
        constraints = []
        for t in range(T - duration + 1):
            prev = z[t - 1] if t > 0 else 0
            constraints.append(z[t] - prev <= cp.sum(z[t:t + duration]))
        return constraints

    def opt(self, t_start, t_end):
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
            z_char_h = cp.Variable(self.T, boolean=True)
            z_disc_h = cp.Variable(self.T, boolean=True)
            z_char_c = cp.Variable(self.T, boolean=True)
            z_disc_c = cp.Variable(self.T, boolean=True)

            M = 1e3

            constraints = [
                SoC_PCM_h[0] == SoC_PCM_h_init,
                SoC_PCM_c[0] == SoC_PCM_c_init
            ]

            eer = self.EER[time_series].values
            cop = self.COP[time_series].values

            for t in range(self.T):
                constraints += [
                    epsilon_c[t] >= cp.exp(self.k * (SoC_PCM_c[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)),
                    epsilon_h[t] >= cp.exp(self.k * (SoC_PCM_h[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)),
                    SoC_PCM_h[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_h[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_c[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_c[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_c[t + 1] == SoC_PCM_c[t] + 100 * ((PCM_char_c[t] * eer[t] - PCM_disc_c[t] * eer[t]) / self.Cm_PCM_c) * self.eta_PCM - self.f_loss_PCM * (1 - epsilon_c[t]),
                    SoC_PCM_h[t + 1] == SoC_PCM_h[t] + 100 * ((PCM_char_h[t] * cop[t] - PCM_disc_h[t] * cop[t]) / self.Cm_PCM_h) * self.eta_PCM - self.f_loss_PCM * (1 - epsilon_h[t]),
                    PCM_char_h[t] <= M * z_char_h[t],
                    PCM_disc_h[t] <= M * z_disc_h[t],
                    PCM_char_c[t] <= M * z_char_c[t],
                    PCM_disc_c[t] <= M * z_disc_c[t]
                ]

            surplus = self.df.loc[time_series, self.obj].values
            d_h = self.df.loc[time_series, 'D_H'].values
            d_c = self.df.loc[time_series, 'D_C'].values

            cumulative_future_demand_c = [sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_C']) for t in range(self.T)]
            cumulative_future_demand_h = [sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_H']) for t in range(self.T)]

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

            constraints += self.enforce_min_duration(z_char_h, 3, self.T)
            constraints += self.enforce_min_duration(z_disc_h, 2, self.T)
            constraints += self.enforce_min_duration(z_char_c, 3, self.T)
            constraints += self.enforce_min_duration(z_disc_c, 2, self.T)

            f_surplus_pcm = cp.sum(surplus - (PCM_char_c + PCM_char_h))

            objective = cp.Minimize(f_surplus_pcm + 1e9 * (cp.sum(epsilon_c) + cp.sum(epsilon_h)))
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
            z_char_h = cp.Variable(self.T, boolean=True)
            z_disc_h = cp.Variable(self.T, boolean=True)
            z_char_c = cp.Variable(self.T, boolean=True)
            z_disc_c = cp.Variable(self.T, boolean=True)

            M = 1e3
            constraints = [
                SoC_TCM_h[0] == SoC_TCM_h_init,
                SoC_TCM_c[0] == SoC_TCM_c_init
            ]

            eer = self.EER[time_series].values
            cop = self.COP[time_series].values

            surplus = np.round(self.df.loc[time_series, self.obj].values - (df_results_pcm['y_PCM_c'].values + df_results_pcm['y_PCM_h'].values), 2)
            d_h = self.df.loc[time_series, 'D_H'].values - df_results_pcm['x_PCM_h'].values
            d_c = self.df.loc[time_series, 'D_C'].values - df_results_pcm['x_PCM_c'].values

            cumulative_future_demand_c = [sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_C']) for t in range(self.T)]
            cumulative_future_demand_h = [sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_H']) for t in range(self.T)]

            for t in range(self.T):
                constraints += [
                    SoC_TCM_h[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_h[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_c[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] == SoC_TCM_c[t] + 100 * (((TCM_char_c[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_c[t] * self.eta_TCM_c_dis) * eer[t]) / self.Cm_TCM_c),
                    SoC_TCM_h[t + 1] == SoC_TCM_h[t] + 100 * (((TCM_char_h[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_h[t] * self.eta_TCM_h_dis) * cop[t]) / self.Cm_TCM_h),
                    TCM_char_h[t] <= M * z_char_h[t],
                    TCM_disc_h[t] <= M * z_disc_h[t],
                    TCM_char_c[t] <= M * z_char_c[t],
                    TCM_disc_c[t] <= M * z_disc_c[t],
                    allocated_surplus_h[t] + allocated_surplus_c[t] <= surplus[t],
                    allocated_surplus_h[t] <= cumulative_future_demand_h[t] * u_h[t],
                    allocated_surplus_c[t] <= cumulative_future_demand_c[t] * u_c[t],
                    TCM_char_h[t] <= allocated_surplus_h[t],
                    TCM_char_c[t] <= allocated_surplus_c[t],
                    TCM_disc_c[t] <= d_c[t] * (1 - u_c[t]),
                    TCM_disc_h[t] <= d_h[t] * (1 - u_h[t])
                ]

            constraints += self.enforce_min_duration(z_char_h, 5, self.T)
            constraints += self.enforce_min_duration(z_disc_h, 3, self.T)
            constraints += self.enforce_min_duration(z_char_c, 5, self.T)
            constraints += self.enforce_min_duration(z_disc_c, 3, self.T)

            f_surplus_TCM = cp.sum(surplus - (TCM_char_c + TCM_char_h))

            objective = cp.Minimize(f_surplus_TCM)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.COPT, verbose=False)

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
