import pandas as pd
import numpy as np
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

        self.Cm_TCM_h = Cm_dict['Cm_h']                # Maximum energy storage capacity for TCM heating [MWh_thermal]  # noqa: E501
        self.Cm_TCM_c = Cm_dict['Cm_c']                # Maximum energy storage capacity for TCM cooling [MWh_thermal]  # noqa: E501
        self.Cm_PCM_h = Cm_dict['Cm_h']                # Maximum energy storage capacity for PCM heating [MWh_thermal]  # noqa: E501
        self.Cm_PCM_c = Cm_dict['Cm_c']                # Maximum energy storage capacity for PCM heating [MWh_thermal]  # noqa: E501
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

        while t_start < t_end:
            time_series = pd.date_range(start=t_start, periods=self.T, freq='1h')  # noqa: E501

            # CVXPY Variables
            PCM_disc_c = cp.Variable(self.T, nonneg=True)     # Cooling PCM discharge in electric [MWh] # noqa: E501
            PCM_disc_h = cp.Variable(self.T, nonneg=True)     # Heating PCM discharge in electric [MWh]  # noqa: E501
            PCM_char_c = cp.Variable(self.T, nonneg=True)     # Heating PCM charge in electric [MWh] # noqa: E501
            PCM_char_h = cp.Variable(self.T, nonneg=True)     # Heating PCM charge in electric [MWh] # noqa: E501
            SoC_PCM_h = cp.Variable(self.T + 1, pos=True)     # SoC of Heating PCM   # noqa: E501
            SoC_PCM_c = cp.Variable(self.T + 1, pos=True)     # SoC of Cooling PCM   # noqa: E501
            u_h = cp.Variable(self.T, boolean=True)             # Binary variable, 1 for charge and 0 for discharge  # noqa: E501
            u_c = cp.Variable(self.T, boolean=True)             # Binary variable, 1 for charge and 0 for discharge  # noqa: E501
            allocated_surplus_h = cp.Variable(self.T, nonneg=True)    # the surplus allocated for heating  # noqa: E501
            allocated_surplus_c = cp.Variable(self.T, nonneg=True)    # the surplus allocated for cooling  # noqa: E501
            pcm_h_active = cp.Variable(self.T, boolean=True)
            pcm_c_active = cp.Variable(self.T, boolean=True)
            pcm_h_loss = cp.Variable(self.T)
            pcm_c_loss = cp.Variable(self.T)

            # Initial SoC conditions
            constraints = [
                SoC_PCM_h[0] == SoC_PCM_h_init,
                SoC_PCM_c[0] == SoC_PCM_c_init
            ]

            eer = self.EER[time_series].values
            cop = self.COP[time_series].values

            threshold = self.SoC_PCM_min

            for t in range(self.T):
                constraints += [
                    # Logic for pcm_h_loss: 0 when SoC_PCM_h < threshold, otherwise f_loss_PCM  # noqa: E501
                    pcm_h_loss[t] == pcm_h_active[t] * self.f_loss_PCM,  # exact pcm loss  # noqa: E501

                    # Logic for pcm_c_loss: 0 when SoC_PCM_c < threshold, otherwise f_loss_PCM  # noqa: E501
                    pcm_c_loss[t] == pcm_c_active[t] * self.f_loss_PCM,  # exact pcm loss  # noqa: E501

                    # Big-M constraints to link binary variables with thresholds  # noqa: E501
                    SoC_PCM_h[t+1] - threshold <= pcm_h_active[t] * (1e4),  # If above threshold, active = 1  # noqa: E501
                    SoC_PCM_h[t+1] - threshold >= (1 - pcm_h_active[t]) * (-1e4),       # If below threshold, active = 0  # noqa: E501

                    SoC_PCM_c[t+1] - threshold <= pcm_c_active[t] * (1e4),  # If above threshold, active = 1  # noqa: E501
                    SoC_PCM_c[t+1] - threshold >= (1 - pcm_c_active[t]) * (-1e4)        # If below threshold, active = 0  # noqa: E501
                ]

            # SoC limits and updates
            for t in range(self.T):
                constraints += [
                    SoC_PCM_h[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_h[t + 1] <= self.SoC_PCM_max,   # noqa: E501
                    SoC_PCM_c[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_c[t + 1] <= self.SoC_PCM_max   # noqa: E501
                ]

                # Update SoC for PCM and TCM (with the dynamic proportional loss for PCM)  # noqa: E501
                constraints += [
                    SoC_PCM_c[t + 1] == SoC_PCM_c[t] + 100 * ((PCM_char_c[t] * eer[t] - PCM_disc_c[t] * eer[t]) / self.Cm_PCM_c) * self.eta_PCM - pcm_c_loss[t],  # noqa: E501
                    SoC_PCM_h[t + 1] == SoC_PCM_h[t] + 100 * ((PCM_char_h[t] * cop[t] - PCM_disc_h[t] * cop[t]) / self.Cm_PCM_h) * self.eta_PCM - pcm_h_loss[t]  # noqa: E501
                ]

            surplus = self.df.loc[time_series, self.obj].values  # in MWh
            d_h = self.df.loc[time_series, 'D_H'].values
            d_c = self.df.loc[time_series, 'D_C'].values

            cumulative_future_demand_c = [sum(self.df.loc[time_series[t]:time_series[t]+timedelta(hours=self.T), 'D_C']) for t in range(self.T)]  # noqa: E501
            cumulative_future_demand_h = [sum(self.df.loc[time_series[t]:time_series[t]+timedelta(hours=self.T), 'D_H']) for t in range(self.T)]  # noqa: E501

            # Charge and discharge constraints
            for t in range(self.T):

                constraints += [
                    allocated_surplus_h[t] + allocated_surplus_c[t] <= surplus[t],  # noqa: E501
                    allocated_surplus_h[t] <= cp.maximum(0, cumulative_future_demand_h[t]) * surplus[t] * u_h[t],  # noqa: E501, this is to determine whether allocated_surplus eqquals to 0 or not
                    allocated_surplus_c[t] <= cp.maximum(0, cumulative_future_demand_c[t]) * surplus[t] * u_c[t],  # noqa: E501, this is to determine whether allocated_surplus eqquals to 0 or not
                    PCM_char_h[t] <= allocated_surplus_h[t],  # charge constraint  # noqa: E501
                    PCM_char_c[t] <= allocated_surplus_c[t],  # noqa: E501
                    PCM_disc_c[t] <= d_c[t] * (1 - u_c[t]),   # discharge constraint  # noqa: E501
                    PCM_disc_h[t] <= d_h[t] * (1 - u_h[t])    # discharge constraint  # noqa: E501
                ]

            # Objective function: Minimize surplus energy used for charging
            objective = cp.Minimize(np.sum(d_h + d_c) - cp.sum(PCM_disc_c + PCM_disc_h) + np.sum(surplus) - cp.sum(PCM_char_h + PCM_char_c))  # noqa: E501  # np.sum(surplus) - cp.sum(PCM_char_h + PCM_char_c)

            # Formulate the problem
            problem = cp.Problem(objective, constraints)

            # Solve the problem
            problem.solve(solver='CPLEX', verbose=False)     # MILP problem  # noqa: E501

            # Extract results
            results = {
                'x_PCM_h': PCM_disc_h.value,
                'y_PCM_h': PCM_char_h.value,
                'x_PCM_c': PCM_disc_c.value,
                'y_PCM_c': PCM_char_c.value,
                'SoC_PCM_h': SoC_PCM_h.value[:-1],
                'SoC_PCM_c': SoC_PCM_c.value[:-1],
                'surplus': surplus,
                'D_H': d_h,
                'D_C': d_c
            }

            df_results_pcm = pd.DataFrame(results)

            # Update initial conditions for the next iteration
            self.SoC_PCM_h_init.append(SoC_PCM_h.value[-1])
            self.SoC_PCM_c_init.append(SoC_PCM_c.value[-1])

            print(f'{t_start} PCM optimization finished')

            SoC_TCM_h_init = self.SoC_TCM_h_init[-1]
            SoC_TCM_c_init = self.SoC_TCM_c_init[-1]

            # CVXPY Variables
            TCM_disc_c = cp.Variable(self.T, nonneg=True)     # Cooling TCM discharge in electric [MWh] # noqa: E501
            TCM_disc_h = cp.Variable(self.T, nonneg=True)     # Heating TCM discharge in electric [MWh]  # noqa: E501
            TCM_char_c = cp.Variable(self.T, nonneg=True)     # Heating TCM charge in electric [MWh] # noqa: E501
            TCM_char_h = cp.Variable(self.T, nonneg=True)     # Heating TCM charge in electric [MWh] # noqa: E501
            SoC_TCM_h = cp.Variable(self.T + 1, pos=True)     # SoC of Heating TCM   # noqa: E501
            SoC_TCM_c = cp.Variable(self.T + 1, pos=True)     # SoC of Cooling TCM   # noqa: E501
            u_h = cp.Variable(self.T, boolean=True)             # Binary variable, 1 for charge and 0 for discharge  # noqa: E501
            u_c = cp.Variable(self.T, boolean=True)             # Binary variable, 1 for charge and 0 for discharge  # noqa: E501
            allocated_surplus_h = cp.Variable(self.T, nonneg=True)    # the surplus allocated for heating  # noqa: E501
            allocated_surplus_c = cp.Variable(self.T, nonneg=True)    # the surplus allocated for cooling  # noqa: E501

            # Initial SoC conditions
            constraints = [
                SoC_TCM_h[0] == SoC_TCM_h_init,
                SoC_TCM_c[0] == SoC_TCM_c_init
            ]

            # SoC limits and updates
            for t in range(self.T):
                constraints += [
                    SoC_TCM_h[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_h[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_c[t + 1] <= self.SoC_TCM_max
                ]

                # Update SoC for TCM and TCM (with the dynamic proportional loss for TCM)  # noqa: E501
                constraints += [
                    SoC_TCM_h[t + 1] == SoC_TCM_h[t] + 100 * (((TCM_char_h[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_h[t] * self.eta_TCM_h_dis) * self.alpha) / self.Cm_TCM_h),  # noqa: E501
                    SoC_TCM_c[t + 1] == SoC_TCM_c[t] + 100 * (((TCM_char_c[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_c[t] * self.eta_TCM_c_dis) * self.alpha) / self.Cm_TCM_c)   # noqa: E501
                ]

            surplus = self.df.loc[time_series, self.obj].values - df_results_pcm['y_PCM_h'].values - df_results_pcm['y_PCM_c'].values   # noqa: E501
            d_h = self.df.loc[time_series, 'D_H'].values - df_results_pcm['x_PCM_h'].values      # noqa: E501
            d_c = self.df.loc[time_series, 'D_C'].values - df_results_pcm['x_PCM_c'].values      # noqa: E501

            cumulative_future_demand_h = [sum(d_h[t:self.T]) for t in range(self.T)]  # noqa: E501
            cumulative_future_demand_c = [sum(d_c[t:self.T]) for t in range(self.T)]  # noqa: E501

            # Charge and discharge constraints
            for t in range(self.T):
                constraints += [
                    allocated_surplus_h[t] + allocated_surplus_c[t] <= surplus[t],  # noqa: E501
                    allocated_surplus_h[t] <= cp.maximum(0, cumulative_future_demand_h[t]) * surplus[t] * u_h[t],  # noqa: E501
                    allocated_surplus_c[t] <= cp.maximum(0, cumulative_future_demand_c[t]) * surplus[t] * u_c[t],  # noqa: E501
                    TCM_char_h[t] <= allocated_surplus_h[t],  # charge constraint  # noqa: E501
                    TCM_char_c[t] <= allocated_surplus_c[t],  # noqa: E501
                    TCM_disc_c[t] <= d_c[t] * (1 - u_c[t]),   # discharge constraint  # noqa: E501
                    TCM_disc_h[t] <= d_h[t] * (1 - u_h[t])    # discharge constraint  # noqa: E501
                ]

            # Objective function: Minimize surplus energy used for charging
            objective = cp.Minimize(np.sum(d_h + d_c) - cp.sum(TCM_disc_c + TCM_disc_h) + np.sum(surplus) - cp.sum(TCM_char_h + TCM_char_c))  # noqa: E501  # np.sum(surplus) - cp.sum(TCM_char_h + TCM_char_c)

            # Formulate the problem
            problem = cp.Problem(objective, constraints)

            # Solve the problem
            problem.solve(solver='CPLEX', verbose=False)     # MILP problem  # noqa: E501

            # Extract results
            results = {
                'x_TCM_h': TCM_disc_h.value,
                'y_TCM_h': TCM_char_h.value,
                'x_TCM_c': TCM_disc_c.value,
                'y_TCM_c': TCM_char_c.value,
                'SoC_TCM_h': SoC_TCM_h.value[:-1],
                'SoC_TCM_c': SoC_TCM_c.value[:-1]
            }

            df_results_tcm = pd.DataFrame(results)
            new_rows = pd.concat([df_results_pcm, df_results_tcm], axis=1)   # noqa: E501
            df_results = pd.concat([df_results, new_rows], ignore_index=True, axis=0)   # noqa: E501

            # Update initial conditions for the next iteration
            self.SoC_TCM_h_init.append(SoC_TCM_h.value[-1])
            self.SoC_TCM_c_init.append(SoC_TCM_c.value[-1])

            print(f'{t_start} TCM optimization finished')

            t_start += timedelta(hours=self.T)

        return df_results
