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
        self.f_loss_PCM = 100.0 / 24.0                 # Maximum loss coefficient for PCM storage (100 MWh over 24 hours)  # noqa: E501
        self.D_H = D_H                                 # demand for heating (in electric)  # noqa: E501
        self.D_C = D_C                                 # demand for cooling (in electric)  # noqa: E501
        self.df = df_simplified_calc                   # input dataframe
        self.T = horizon                               # prediction horizon
        self.dt = 1                                    # time step, 1h
        self.alpha = 2.5                               # Conversion factor for TCM [-] (COP of high temperature heat pump for charging TCM) # noqa: E501
        self.obj = optimization_obj                    # The objective of optimization (surplus_all or surplus_RES)  # noqa: E501
        self.df['D_H'] = self.D_H
        self.df['D_C'] = self.D_C
        self.COP = COP
        self.EER = EER
        self.OverallSurplus_GridStorage = self.df['surplus_all'].sum()
        self.RESSurplus_GridStorage = self.df['surplus_RES'].sum()

    def opt(self, t_start, t_end):
        df_results = pd.DataFrame()

        SoC_PCM_h_init = self.SoC_PCM_min
        SoC_PCM_c_init = self.SoC_PCM_min
        SoC_TCM_h_init = self.SoC_TCM_min
        SoC_TCM_c_init = self.SoC_TCM_min

        while t_start < t_end:
            time_series = pd.date_range(start=t_start, periods=self.T, freq='1h')  # noqa: E501

            # CVXPY Variables
            TCM_disc_h = cp.Variable(self.T, nonneg=True)     # TCM discharge in electric for heating # noqa: E501
            TCM_disc_c = cp.Variable(self.T, nonneg=True)     # TCM discharge in electric for cooling # noqa: E501
            PCM_disc_c = cp.Variable(self.T, nonneg=True)     # Cooling PCM discharge in electric # noqa: E501
            PCM_disc_h = cp.Variable(self.T, nonneg=True)     # Heating PCM discharge in electric  # noqa: E501
            TCM_char_h = cp.Variable(self.T, nonneg=True)     # TCM charge in electric # noqa: E501
            TCM_char_c = cp.Variable(self.T, nonneg=True)     # TCM charge in electric # noqa: E501
            PCM_char_c = cp.Variable(self.T, nonneg=True)     # Heating PCM charge in electric # noqa: E501
            PCM_char_h = cp.Variable(self.T, nonneg=True)     # Heating PCM charge in electric # noqa: E501
            SoC_PCM_h = cp.Variable(self.T + 1, pos=True)     # SoC of Heating PCM  # noqa: E501
            SoC_PCM_c = cp.Variable(self.T + 1, pos=True)     # SoC of Cooling PCM  # noqa: E501
            SoC_TCM_h = cp.Variable(self.T + 1, pos=True)     # SoC of Heating TCM   # noqa: E501
            SoC_TCM_c = cp.Variable(self.T + 1, pos=True)     # SoC of Cooling TCM   # noqa: E501
            u = cp.Variable(self.T, boolean=True)             # Binary variable for charge/discharge  # noqa: E501

            # Initial SoC conditions
            constraints = [
                SoC_PCM_h[0] == SoC_PCM_h_init,
                SoC_PCM_c[0] == SoC_PCM_c_init,
                SoC_TCM_h[0] == SoC_TCM_h_init,
                SoC_TCM_c[0] == SoC_TCM_c_init
            ]

            loss_h = 0
            loss_c = 0

            # SoC limits and updates
            for t in range(self.T):
                constraints += [
                    SoC_PCM_h[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_h[t + 1] <= self.SoC_PCM_max,
                    SoC_PCM_c[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_c[t + 1] <= self.SoC_PCM_max,
                    SoC_TCM_h[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_h[t + 1] <= self.SoC_TCM_max,
                    SoC_TCM_c[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_c[t + 1] <= self.SoC_TCM_max
                ]

                # Calculate dynamic loss factor (scales between 0 and 1 based on SoC level)  # noqa: E501
                loss_factor_h = (SoC_PCM_h[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)  # noqa: E501
                loss_factor_c = (SoC_PCM_c[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)  # noqa: E501

                # Calculate PCM loss proportional to the state of charge (approaching max loss at SoC max)  # noqa: E501
                pcm_h_loss = loss_factor_h * self.f_loss_PCM
                pcm_c_loss = loss_factor_c * self.f_loss_PCM

                loss_h += pcm_h_loss
                loss_c += pcm_c_loss

                '''
                # Calculate dynamic loss factor (scales between 0 and 1 based on SoC level)  # noqa: E501
                loss_factor_h = (SoC_TCM_h[t] - self.SoC_TCM_min) / (self.SoC_TCM_max - self.SoC_TCM_min)  # noqa: E501
                loss_factor_c = (SoC_TCM_c[t] - self.SoC_TCM_min) / (self.SoC_TCM_max - self.SoC_TCM_min)  # noqa: E501

                # Calculate TCM loss proportional to the state of charge (approaching max loss at SoC max)  # noqa: E501
                TCM_h_loss = loss_factor_h * self.f_loss_PCM
                TCM_c_loss = loss_factor_c * self.f_loss_PCM

                loss_h_t += TCM_h_loss
                loss_c_t += TCM_c_loss
                '''

                # Update SoC for PCM and TCM (with the dynamic proportional loss for PCM)  # noqa: E501
                constraints += [
                    SoC_PCM_c[t + 1] == SoC_PCM_c[t] + 100 * ((PCM_char_c[t] * self.EER[time_series[t]] - PCM_disc_c[t] * self.EER[time_series[t]]) / self.Cm_PCM_c) * self.eta_PCM - pcm_c_loss,  # noqa: E501
                    SoC_PCM_h[t + 1] == SoC_PCM_h[t] + 100 * ((PCM_char_h[t] * self.COP[time_series[t]] - PCM_disc_h[t] * self.COP[time_series[t]]) / self.Cm_PCM_h) * self.eta_PCM - pcm_h_loss,  # noqa: E501
                    SoC_TCM_h[t + 1] == SoC_TCM_h[t] + 100 * (((TCM_char_h[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_h[t] * self.eta_TCM_h_dis) * self.alpha) / self.Cm_TCM_h),  # noqa: E501
                    SoC_TCM_c[t + 1] == SoC_TCM_c[t] + 100 * (((TCM_char_c[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_c[t] * self.eta_TCM_c_dis) * self.alpha) / self.Cm_TCM_c)   # noqa: E501
                ]

            surplus = self.df.loc[time_series, self.obj].values
            d_h = self.df.loc[time_series, 'D_H'].values
            d_c = self.df.loc[time_series, 'D_C'].values

            # Charge and discharge constraints
            for t in range(self.T):
                constraints += [
                    TCM_char_h[t] + TCM_char_c[t] + PCM_char_h[t] + PCM_char_c[t] <= surplus[t] * u[t],  # charge constraint  # noqa: E501
                    TCM_disc_h[t] + PCM_disc_h[t] <= d_h[t] * (1 - u[t]),  # discharge constraint  # noqa: E501
                    TCM_disc_c[t] + PCM_disc_c[t] <= d_c[t] * (1 - u[t])
                ]

            # penalty = (cp.sum(PCM_disc_h) - loss_h) + (cp.sum(PCM_disc_c) - loss_c)  # noqa: E501

            # Objective function: Minimize surplus energy used for charging
            objective = cp.Minimize(np.sum(surplus) - cp.sum(TCM_char_h + TCM_char_c + PCM_char_h + PCM_char_c) - cp.sum(TCM_disc_h + TCM_disc_c + PCM_disc_h + PCM_disc_c))  # noqa: E501

            # Formulate the problem
            problem = cp.Problem(objective, constraints)

            # Solve the problem by using open source solver
            # If you have error like 'Solver is not installed', try to run 'pip install COPT'  # noqa: E501
            problem.solve(solver='COPT', verbose=False)     # MILP problem  # noqa: E501

            # Extract results
            results = {
                'x_TCM_h': TCM_disc_h.value,
                'x_TCM_c': TCM_disc_c.value,
                'y_TCM_h': TCM_char_h.value,
                'y_TCM_c': TCM_char_c.value,
                'x_PCM_h': PCM_disc_h.value,
                'y_PCM_h': PCM_char_h.value,
                'x_PCM_c': PCM_disc_c.value,
                'y_PCM_c': PCM_char_c.value,
                'SoC_TCM_h': SoC_TCM_h.value[:-1],
                'SoC_TCM_c': SoC_TCM_c.value[:-1],
                'SoC_PCM_h': SoC_PCM_h.value[:-1],
                'SoC_PCM_c': SoC_PCM_c.value[:-1],
                'surplus': surplus,
                'D_H': d_h,
                'D_C': d_c
            }

            new_row_df = pd.DataFrame(results)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

            # Update initial conditions for the next iteration
            SoC_PCM_h_init = SoC_PCM_h.value[-1]
            SoC_PCM_c_init = SoC_PCM_c.value[-1]
            SoC_TCM_h_init = SoC_TCM_h.value[-1]
            SoC_TCM_c_init = SoC_TCM_c.value[-1]

            print(f'{t_start} optimization finished')

            t_start += timedelta(hours=self.T)

        return df_results
