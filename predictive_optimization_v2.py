import pandas as pd
import cvxpy as cp
from datetime import timedelta


class PredictiveOptimizerCVXPY:
    def __init__(self,
                 D_H,                   # Heating demand in electric [GWh]
                 D_C,                   # Cooling demand in electric [GWh]
                 df_simplified_calc,    # Input dataframe containing surplus_all and surplus_RES  # noqa: E501
                 horizon,               # Optimization horizon
                 COP,                   # Hourly COP values
                 EER,                   # Hourly EER values
                 Cm_dict,               # The capacity dictionary of capacities for TCM and PCM  # noqa: E501
                 optimization_obj):     # Optimization objective, either surplus_all or surplus_RES  # noqa: E501

        self.Cm_TCM_h = Cm_dict['Cm_h_TCM']                # Maximum energy storage capacity for TCM heating [MWh_thermal]  # noqa: E501
        self.Cm_TCM_c = Cm_dict['Cm_c_TCM']                # Maximum energy storage capacity for TCM cooling [MWh_thermal]  # noqa: E501
        self.Cm_PCM_h = Cm_dict['Cm_h_PCM']                # Maximum energy storage capacity for PCM heating [MWh_thermal]  # noqa: E501
        self.Cm_PCM_c = Cm_dict['Cm_c_PCM']                # Maximum energy storage capacity for PCM heating [MWh_thermal]  # noqa: E501
        self.eta_TCM_c_dis = 0.5                       # Efficiency of the TCM storage during cooling discharge [-]  # noqa: E501
        self.eta_TCM_h_dis = 1.1                       # Efficiency of the TCM storage during heating discharge [-]  # noqa: E501
        self.eta_TCM_ch = 1.0                          # Efficiency of the TCM storage during charge for both heating and cooling [-]  # noqa: E501
        self.eta_PCM = 0.8                             # Efficiency of the PCM storage [-]  # noqa: E501
        self.SoC_TCM_max = 80.0                        # State of Charge max of the TCM storage [%]  # noqa: E501
        self.SoC_TCM_min = 15.0                        # State of Charge min of the TCM storage [%]  # noqa: E501
        self.SoC_PCM_max = 75.0                        # State of Charge max of the PCM storage [%]  # noqa: E501
        self.SoC_PCM_min = 20.0                        # State of Charge min of the PCM storage [%]  # noqa: E501
        self.f_loss_PCM = (
            self.SoC_PCM_max - self.SoC_PCM_min
        ) / 24.0                                       # Maximum loss coefficient for PCM storage (100% of SoC over 24 hours)  # noqa: E501
        self.D_H = D_H                                 # demand for heating (in electric)  # noqa: E501
        self.D_C = D_C                                 # demand for cooling (in electric)  # noqa: E501
        self.df = df_simplified_calc                   # input dataframe
        self.T = horizon                               # prediction horizon
        self.dt = 1                                    # time step, 1h
        self.alpha = 2.5                               # Conversion factor for TCM [-] (COP of high temperature heat pump for charging TCM) # noqa: E501
        self.obj = optimization_obj                    # The objective of optimization (surplus_all or surplus_RES)  # noqa: E501
        self.k = 7                                     # Constant coefficient for calculating the loss of PCM storage    # noqa: E501
        self.df['D_H'] = self.D_H
        self.df['D_C'] = self.D_C
        self.COP = COP
        self.EER = EER
        self.SoC_PCM_h_init = [self.SoC_PCM_min]
        self.SoC_PCM_c_init = [self.SoC_PCM_min]
        self.SoC_TCM_h_init = [self.SoC_TCM_min]
        self.SoC_TCM_c_init = [self.SoC_TCM_min]

    def opt(self, t_start, t_end):
        df_results = pd.DataFrame()
        SoC_PCM_h_init = self.SoC_PCM_h_init[-1]
        SoC_PCM_c_init = self.SoC_PCM_c_init[-1]
        SoC_TCM_h_init = self.SoC_TCM_h_init[-1]
        SoC_TCM_c_init = self.SoC_TCM_c_init[-1]

        while t_start < t_end:
            time_series = pd.date_range(start=t_start, periods=self.T, freq='1h')

            # Cooling Optimization

            # CVXPY Variables
            PCM_disc_c = cp.Variable(self.T, nonneg=True)  # Cooling PCM discharge [MWh]
            PCM_char_c = cp.Variable(self.T, nonneg=True)  # Cooling PCM charge [MWh]
            SoC_PCM_c = cp.Variable(self.T + 1, pos=True)  # Cooling PCM SoC
            pcm_c_active = cp.Variable(self.T, boolean=True)
            pcm_c_loss = cp.Variable(self.T, nonneg=True)

            TCM_disc_c = cp.Variable(self.T, nonneg=True)  # Cooling TCM discharge [MWh]
            TCM_char_c = cp.Variable(self.T, nonneg=True)  # Cooling TCM charge [MWh]
            SoC_TCM_c = cp.Variable(self.T + 1, pos=True)  # Cooling TCM SoC

            allocated_surplus_pcm = cp.Variable(self.T, nonneg=True)  # Surplus allocated to cooling
            allocated_surplus_tcm = cp.Variable(self.T, nonneg=True)  # Surplus allocated to cooling
            u_c_pcm = cp.Variable(self.T, boolean=True)  # Binary variable for cooling charge/discharge
            u_c_tcm = cp.Variable(self.T, boolean=True)  # Binary variable for cooling charge/discharge

            constraints = [
                SoC_PCM_c[0] == SoC_PCM_c_init,
                SoC_TCM_c[0] == SoC_TCM_c_init
            ]

            eer = self.EER[time_series].values
            surplus = self.df.loc[time_series, self.obj].values
            d_c = self.df.loc[time_series, 'D_C'].values

            cumulative_future_demand_c = [
                sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_C'])
                for t in range(self.T)
            ]

            for t in range(self.T):
                constraints += [
                    pcm_c_loss[t] == pcm_c_active[t] * self.f_loss_PCM,
                    SoC_PCM_c[t+1] - self.SoC_PCM_min <= pcm_c_active[t] * 1e4,
                    SoC_PCM_c[t+1] - self.SoC_PCM_min >= (1 - pcm_c_active[t]) * -1e4,
                    SoC_PCM_c[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_c[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_c[t + 1] == SoC_PCM_c[t] + 100 * ((PCM_char_c[t] * eer[t] - PCM_disc_c[t] * eer[t]) / self.Cm_PCM_c) * self.eta_PCM - pcm_c_loss[t],  # noqa: E501
                    SoC_TCM_c[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_c[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] == SoC_TCM_c[t] + 100 * (((TCM_char_c[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_c[t] * self.eta_TCM_c_dis) * self.alpha) / self.Cm_TCM_c),   # noqa: E501
                    allocated_surplus_pcm[t] + allocated_surplus_tcm[t] <= surplus[t],
                    allocated_surplus_pcm[t] <= cumulative_future_demand_c[t] * u_c_pcm[t],
                    allocated_surplus_tcm[t] <= cumulative_future_demand_c[t] * u_c_tcm[t],
                    PCM_char_c[t] <= allocated_surplus_pcm[t],
                    TCM_char_c[t] <= allocated_surplus_tcm[t],
                    PCM_disc_c[t] + TCM_disc_c[t] <= d_c[t],
                    PCM_disc_c[t] <= d_c[t] * (1-u_c_pcm[t]),
                    TCM_disc_c[t] <= d_c[t] * (1-u_c_tcm[t])
                ]

            objective = cp.Minimize(cp.sum(d_c - (10 * PCM_disc_c + TCM_disc_c)) +
                                    cp.sum(surplus - (10 * PCM_char_c + TCM_char_c)))

            problem = cp.Problem(objective, constraints)
            problem.solve(solver='CPLEX', verbose=False)

            # Extract cooling-related results
            cooling_results = {
                'x_PCM_c': PCM_disc_c.value,
                'y_PCM_c': PCM_char_c.value,
                'x_TCM_c': TCM_disc_c.value,
                'y_TCM_c': TCM_char_c.value,
                'SoC_PCM_c': SoC_PCM_c.value[:-1],
                'SoC_TCM_c': SoC_TCM_c.value[:-1],
                'surplus': surplus,
                'D_C': d_c
            }
            df_results_cooling = pd.DataFrame(cooling_results)

            # Update initial conditions for heating optimization
            self.SoC_PCM_c_init.append(SoC_PCM_c.value[-1])
            self.SoC_TCM_c_init.append(SoC_TCM_c.value[-1])

            # **Heating Optimization**

            PCM_disc_h = cp.Variable(self.T, nonneg=True)  # Heating PCM discharge [MWh]
            PCM_char_h = cp.Variable(self.T, nonneg=True)  # Heating PCM charge [MWh]
            SoC_PCM_h = cp.Variable(self.T + 1, pos=True)  # Heating PCM SoC
            pcm_h_active = cp.Variable(self.T, boolean=True)
            pcm_h_loss = cp.Variable(self.T, nonneg=True)

            TCM_disc_h = cp.Variable(self.T, nonneg=True)  # Heating TCM discharge [MWh]
            TCM_char_h = cp.Variable(self.T, nonneg=True)  # Heating TCM charge [MWh]
            SoC_TCM_h = cp.Variable(self.T + 1, pos=True)  # Heating TCM SoC

            allocated_surplus_pcm = cp.Variable(self.T, nonneg=True)  # Surplus allocated to cooling
            allocated_surplus_tcm = cp.Variable(self.T, nonneg=True)  # Surplus allocated to cooling
            u_h_pcm = cp.Variable(self.T, boolean=True)  # Binary variable for cooling charge/discharge
            u_h_tcm = cp.Variable(self.T, boolean=True)  # Binary variable for cooling charge/discharge

            constraints = [
                SoC_PCM_h[0] == SoC_PCM_h_init,
                SoC_TCM_h[0] == SoC_TCM_h_init
            ]

            cop = self.COP[time_series].values
            d_h = self.df.loc[time_series, 'D_H'].values
            surplus = surplus - df_results_cooling['y_PCM_c'].values - df_results_cooling['y_TCM_c'].values

            cumulative_future_demand_h = [
                sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_H'])
                for t in range(self.T)
            ]

            for t in range(self.T):
                constraints += [
                    pcm_h_loss[t] == pcm_h_active[t] * self.f_loss_PCM,
                    SoC_PCM_h[t + 1] - self.SoC_PCM_min <= pcm_h_active[t] * 1e4,
                    SoC_PCM_h[t + 1] - self.SoC_PCM_min >= (1 - pcm_h_active[t]) * -1e4,
                    SoC_PCM_h[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_h[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_h[t + 1] == SoC_PCM_h[t] + 100 * ((PCM_char_h[t] * cop[t] - PCM_disc_h[t] * cop[t]) / self.Cm_PCM_h) * self.eta_PCM - pcm_h_loss[t],  # noqa: E501
                    SoC_TCM_h[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_h[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_h[t + 1] == SoC_TCM_h[t] + 100 * (((TCM_char_h[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_h[t] * self.eta_TCM_h_dis) * self.alpha) / self.Cm_TCM_h),  # noqa: E501,
                    allocated_surplus_pcm[t] + allocated_surplus_tcm[t] <= surplus[t],
                    allocated_surplus_pcm[t] <= cumulative_future_demand_h[t] * u_h_pcm[t],
                    allocated_surplus_tcm[t] <= cumulative_future_demand_h[t] * u_h_tcm[t],
                    PCM_char_h[t] <= allocated_surplus_pcm[t],
                    TCM_char_h[t] <= allocated_surplus_tcm[t],
                    PCM_disc_h[t] + TCM_disc_h[t] <= d_h[t],
                    PCM_disc_h[t] <= d_h[t] * (1 - u_h_pcm[t]),
                    TCM_disc_h[t] <= d_h[t] * (1 - u_h_tcm[t])
                ]

            objective = cp.Minimize(cp.sum(d_h - (10 * PCM_disc_h + TCM_disc_h)) +
                                    cp.sum(surplus - (10 * PCM_char_h + TCM_char_h)))

            problem = cp.Problem(objective, constraints)
            problem.solve(solver='CPLEX', verbose=False)

            # Extract heating-related results
            heating_results = {
                'x_PCM_h': PCM_disc_h.value,
                'y_PCM_h': PCM_char_h.value,
                'x_TCM_h': TCM_disc_h.value,
                'y_TCM_h': TCM_char_h.value,
                'SoC_PCM_h': SoC_PCM_h.value[:-1],
                'SoC_TCM_h': SoC_TCM_h.value[:-1],
                'D_H': d_h
            }
            df_results_heating = pd.DataFrame(heating_results)

            # Update initial conditions for the next time step
            self.SoC_PCM_h_init.append(SoC_PCM_h.value[-1])
            self.SoC_TCM_h_init.append(SoC_TCM_h.value[-1])

            # Combine results
            df_results = pd.concat([df_results, pd.concat([df_results_cooling, df_results_heating], axis=1)], ignore_index=True)

            print(f'{t_start} optimization finished')
            t_start += timedelta(hours=self.T)

        return df_results
