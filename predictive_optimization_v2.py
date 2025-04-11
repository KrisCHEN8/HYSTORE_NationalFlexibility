import pandas as pd
import cvxpy as cp
from datetime import timedelta
import numpy as np


class PredictiveOptimizerCVXPY:
    def __init__(self,
                 D_H,                   # Heating demand in electric [GWh]
                 D_C,                   # Cooling demand in electric [GWh]
                 df_simplified_calc,    # Input dataframe containing surplus_all and surplus_RES
                 df_emission,           # Input dataframe containing CO₂ emission data
                 horizon,               # Optimization horizon (in hours)
                 COP,                   # Hourly COP values
                 EER,                   # Hourly EER values
                 Cm_dict,               # Dictionary of storage capacities for TCM and PCM
                 optimization_obj,      # Which surplus column to use: surplus_all or surplus_RES
                 ):
        # Storage capacities for TCM [MWh_thermal]
        self.Cm_TCM_h = Cm_dict['Cm_h_TCM']
        self.Cm_TCM_c = Cm_dict['Cm_c_TCM']
        # Storage capacities for PCM [MWh_thermal]
        self.Cm_PCM_h = Cm_dict['Cm_h_PCM']
        self.Cm_PCM_c = Cm_dict['Cm_c_PCM']

        # Storage efficiency parameters
        self.eta_TCM_c_dis = 0.5  # Efficiency for TCM cooling discharge
        self.eta_TCM_h_dis = 1.1  # Efficiency for TCM heating discharge
        self.eta_TCM_ch = 1.0     # Efficiency for TCM charging (both heating and cooling)
        self.eta_PCM = 0.8        # Efficiency for PCM storage

        # State-of-charge (SoC) limits (in %)
        self.SoC_TCM_max = 80.0
        self.SoC_TCM_min = 15.0
        self.SoC_PCM_max = 75.0
        self.SoC_PCM_min = 20.0

        # Loss parameters for PCM
        self.f_loss_PCM = (self.SoC_PCM_max - self.SoC_PCM_min) / 24.0
        # (Constant used in the exponential constraint for modeling PCM losses, here negative)
        self.k = -7

        self.D_H = D_H  # Heating demand (electric)
        self.D_C = D_C  # Cooling demand (electric)
        self.df = df_simplified_calc.copy()
        self.emission = df_emission.copy()
        self.T = horizon  # prediction horizon
        self.dt = 1       # time step: 1h
        self.alpha = 2.5  # conversion factor for TCM charging
        # Optimization objective indicator (to select surplus column from df)
        self.obj = optimization_obj

        # Include demands in the dataframe
        self.df['D_H'] = self.D_H
        self.df['D_C'] = self.D_C

        # Hourly COP and EER arrays (indexed by datetime)
        self.COP = COP
        self.EER = EER

        # Initialize SoC lists with starting conditions (use the minimum allowed)
        self.SoC_PCM_h_init = [self.SoC_PCM_min]
        self.SoC_PCM_c_init = [self.SoC_PCM_min]
        self.SoC_TCM_h_init = [self.SoC_TCM_min]
        self.SoC_TCM_c_init = [self.SoC_TCM_min]

        # Save the original initial conditions for use in Pareto front experiments.
        self.initial_conditions = {
            'SoC_PCM_h': self.SoC_PCM_h_init.copy(),
            'SoC_PCM_c': self.SoC_PCM_c_init.copy(),
            'SoC_TCM_h': self.SoC_TCM_h_init.copy(),
            'SoC_TCM_c': self.SoC_TCM_c_init.copy(),
        }

    def reset_initial_conditions(self):
        # Use this to reinitialize the state variables between Pareto runs.
        self.SoC_PCM_h_init = self.initial_conditions['SoC_PCM_h'].copy()
        self.SoC_PCM_c_init = self.initial_conditions['SoC_PCM_c'].copy()
        self.SoC_TCM_h_init = self.initial_conditions['SoC_TCM_h'].copy()
        self.SoC_TCM_c_init = self.initial_conditions['SoC_TCM_c'].copy()

    def opt(self, t_start, t_end, lambda_value):
        """
        Runs the receding horizon optimization using a weighted-sum Pareto formulation.

        Parameters:
          t_start: Starting timestamp (datetime)
          t_end: Ending timestamp (datetime)
          lambda_value: Pareto weight for the surplus residual cost.
                        (0 <= lambda_value <= 1). When lambda_value=1, the optimizer
                        focuses solely on reducing unused surplus; when 0, solely on reducing carbon cost.

        Returns:
          A tuple (df_results, total_surplus_cost, total_carbon_cost) where:
            - df_results: DataFrame with selected decision variables from each time interval,
            - total_surplus_cost: Aggregated surplus residual cost,
            - total_carbon_cost: Aggregated carbon cost.
        """
        df_results = pd.DataFrame()
        total_surplus_cost = 0.0
        total_carbon_cost = 0.0

        # Get the current state-of-charge initial values for both cooling and heating
        SoC_PCM_h_init = self.SoC_PCM_h_init[-1]
        SoC_PCM_c_init = self.SoC_PCM_c_init[-1]
        SoC_TCM_h_init = self.SoC_TCM_h_init[-1]
        SoC_TCM_c_init = self.SoC_TCM_c_init[-1]

        while t_start < t_end:
            time_series = pd.date_range(start=t_start, periods=self.T, freq='1h')

            # ------------------ Cooling Optimization ----------------------
            # Create CVXPY variables for cooling
            PCM_disc_c = cp.Variable(self.T, nonneg=True)   # PCM discharge [MWh]
            PCM_char_c = cp.Variable(self.T, nonneg=True)     # PCM charge [MWh]
            SoC_PCM_c = cp.Variable(self.T + 1)               # PCM state-of-charge
            epsilon_c = cp.Variable(self.T)                   # PCM loss slack variable

            TCM_disc_c = cp.Variable(self.T, nonneg=True)     # TCM discharge [MWh]
            TCM_char_c = cp.Variable(self.T, nonneg=True)       # TCM charge [MWh]
            SoC_TCM_c = cp.Variable(self.T + 1)               # TCM state-of-charge

            allocated_surplus_pcm_c = cp.Variable(self.T, nonneg=True)
            allocated_surplus_tcm_c = cp.Variable(self.T, nonneg=True)
            u_c_pcm = cp.Variable(self.T, boolean=True)
            u_c_tcm = cp.Variable(self.T, boolean=True)

            # Initial conditions for cooling optimization
            constraints = [
                SoC_PCM_c[0] == SoC_PCM_c_init,
                SoC_TCM_c[0] == SoC_TCM_c_init
            ]

            eer = self.EER[time_series].values
            surplus = self.df.loc[time_series, self.obj].values
            d_c = self.df.loc[time_series, 'D_C'].values
            co2 = self.emission.loc[time_series, 'Mix elettrico.1'].values

            cumulative_future_demand_c = [
                    sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_C'])   # noqa: E501
                    for t in range(self.T)
             ]

            # For each time step in the prediction horizon
            for t in range(self.T):
                constraints += [
                    # Enforce SoC limits for PCM and TCM
                    SoC_PCM_c[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_c[t + 1] <= self.SoC_PCM_max,
                    SoC_TCM_c[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_c[t + 1] <= self.SoC_TCM_max,

                    # PCM loss modelling using exponential (using cp.exp for CVXPY)
                    epsilon_c[t] >= cp.exp(self.k * (SoC_PCM_c[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)),

                    # Update rule for PCM SoC (charging/discharging, with conversion factor and losses)
                    SoC_PCM_c[t + 1] == SoC_PCM_c[t] +
                        100 * ((PCM_char_c[t] * eer[t] - PCM_disc_c[t] * eer[t]) / self.Cm_PCM_c) * self.eta_PCM
                        - self.f_loss_PCM * (1 - epsilon_c[t]),

                    # Update rule for TCM SoC
                    SoC_TCM_c[t + 1] == SoC_TCM_c[t] +
                        100 * (((TCM_char_c[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_c[t] * self.eta_TCM_c_dis) * eer[t]) / self.Cm_TCM_c),

                    # Surplus allocation constraints
                    allocated_surplus_pcm_c[t] + allocated_surplus_tcm_c[t] <= surplus[t],
                    allocated_surplus_pcm_c[t] <= cumulative_future_demand_c * u_c_pcm[t],
                    allocated_surplus_tcm_c[t] <= cumulative_future_demand_c * u_c_tcm[t],
                    PCM_char_c[t] <= allocated_surplus_pcm_c[t],
                    TCM_char_c[t] <= allocated_surplus_tcm_c[t],
                    PCM_disc_c[t] + TCM_disc_c[t] <= d_c[t],
                    PCM_disc_c[t] <= d_c[t] * (1 - u_c_pcm[t]),
                    TCM_disc_c[t] <= d_c[t] * (1 - u_c_tcm[t])
                ]

            # Define the two cost terms for cooling:
            # (1) Surplus residual cost: unused surplus is surplus minus the charging amounts.
            f_surplus_c = cp.sum(surplus - (PCM_char_c + TCM_char_c))
            # (2) Carbon cost: unmet cooling demand (i.e. demand minus discharge) times emission factor.
            f_carbon_c = cp.sum(cp.multiply(PCM_disc_c + TCM_disc_c, co2))

            objective_c = cp.Minimize(f_surplus_c - lambda_value * f_carbon_c + 1e9 * cp.sum(epsilon_c))
            prob_c = cp.Problem(objective_c, constraints)
            prob_c.solve(solver='MOSEK', verbose=False)

            # Record cooling results
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

            # Compute (for Pareto evaluation) the cooling cost contributions:
            surplus_cost_c = np.sum(surplus - (PCM_char_c.value + TCM_char_c.value))
            carbon_cost_c = np.sum((PCM_disc_c.value + TCM_disc_c.value) * co2)

            # Update the cooling storage initial conditions for later use
            self.SoC_PCM_c_init.append(SoC_PCM_c.value[-1])
            self.SoC_TCM_c_init.append(SoC_TCM_c.value[-1])

            # ------------------ Heating Optimization ----------------------
            # Create CVXPY variables for heating
            PCM_disc_h = cp.Variable(self.T, nonneg=True)
            PCM_char_h = cp.Variable(self.T, nonneg=True)
            SoC_PCM_h = cp.Variable(self.T + 1)
            epsilon_h = cp.Variable(self.T)

            TCM_disc_h = cp.Variable(self.T, nonneg=True)
            TCM_char_h = cp.Variable(self.T, nonneg=True)
            SoC_TCM_h = cp.Variable(self.T + 1)

            allocated_surplus_pcm_h = cp.Variable(self.T, nonneg=True)
            allocated_surplus_tcm_h = cp.Variable(self.T, nonneg=True)
            u_h_pcm = cp.Variable(self.T, boolean=True)
            u_h_tcm = cp.Variable(self.T, boolean=True)

            # Initial conditions for heating optimization
            constraints_h = [
                SoC_PCM_h[0] == SoC_PCM_h_init,
                SoC_TCM_h[0] == SoC_TCM_h_init
            ]

            cop = self.COP[time_series].values
            d_h = self.df.loc[time_series, 'D_H'].values
            co2 = self.emission.loc[time_series, 'Mix elettrico.1'].values
            # For heating, update the available surplus by subtracting the energy used in cooling.
            surplus_h = surplus - (df_results_cooling['y_PCM_c'].values + df_results_cooling['y_TCM_c'].values)

            cumulative_future_demand_h = [
                    sum(self.df.loc[time_series[t]:time_series[t] + timedelta(hours=self.T), 'D_H'])   # noqa: E501
                    for t in range(self.T)
             ]

            for t in range(self.T):
                constraints_h += [
                    SoC_PCM_h[t + 1] >= self.SoC_PCM_min,
                    SoC_PCM_h[t + 1] <= self.SoC_PCM_max,
                    SoC_TCM_h[t + 1] >= self.SoC_TCM_min,
                    SoC_TCM_h[t + 1] <= self.SoC_TCM_max,

                    epsilon_h[t] >= cp.exp(self.k * (SoC_PCM_h[t] - self.SoC_PCM_min) / (self.SoC_PCM_max - self.SoC_PCM_min)),

                    SoC_PCM_h[t + 1] == SoC_PCM_h[t] +
                        100 * ((PCM_char_h[t] * cop[t] - PCM_disc_h[t] * cop[t]) / self.Cm_PCM_h) * self.eta_PCM
                        - self.f_loss_PCM * (1 - epsilon_h[t]),

                    SoC_TCM_h[t + 1] == SoC_TCM_h[t] +
                        100 * (((TCM_char_h[t] * self.eta_TCM_ch) * self.alpha - (TCM_disc_h[t] * self.eta_TCM_h_dis) * cop[t]) / self.Cm_TCM_h),

                    allocated_surplus_pcm_h[t] + allocated_surplus_tcm_h[t] <= surplus_h[t],
                    allocated_surplus_pcm_h[t] <= cumulative_future_demand_h * u_h_pcm[t],
                    allocated_surplus_tcm_h[t] <= cumulative_future_demand_h * u_h_tcm[t],
                    PCM_char_h[t] <= allocated_surplus_pcm_h[t],
                    TCM_char_h[t] <= allocated_surplus_tcm_h[t],
                    PCM_disc_h[t] + TCM_disc_h[t] <= d_h[t],
                    PCM_disc_h[t] <= d_h[t] * (1 - u_h_pcm[t]),
                    TCM_disc_h[t] <= d_h[t] * (1 - u_h_tcm[t])
                ]

            # Define the two cost terms for heating
            f_surplus_h = cp.sum(surplus_h - (PCM_char_h + TCM_char_h))
            f_carbon_h = cp.sum(cp.multiply(PCM_disc_h + TCM_disc_h, co2))

            objective_h = cp.Minimize(f_surplus_h + lambda_value * f_carbon_h + 1e9 * cp.sum(epsilon_h))
            prob_h = cp.Problem(objective_h, constraints_h)
            prob_h.solve(solver='MOSEK', verbose=False)

            # Record heating results
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

            # Compute the heating cost contributions
            surplus_cost_h = np.sum(surplus_h - (PCM_char_h.value + TCM_char_h.value))
            carbon_cost_h = np.sum((PCM_disc_h.value + TCM_disc_h.value) * co2)

            # Update the heating storage initial conditions for the next iteration
            self.SoC_PCM_h_init.append(SoC_PCM_h.value[-1])
            self.SoC_TCM_h_init.append(SoC_TCM_h.value[-1])

            # Combine first-step results (you can later analyze the full time series if needed)
            first_row_cooling = df_results_cooling  #.iloc[[0], :].reset_index(drop=True)
            first_row_heating = df_results_heating  #.iloc[[0], :].reset_index(drop=True)
            combined_first_rows = pd.concat([first_row_cooling, first_row_heating], axis=1)
            df_results = pd.concat([df_results, combined_first_rows], ignore_index=True)

            # Update the aggregate cost measures
            total_surplus_cost += (surplus_cost_c + surplus_cost_h)
            total_carbon_cost += (carbon_cost_c + carbon_cost_h)

            print(f'{t_start} optimization finished with lambda = {lambda_value}')
            t_start += timedelta(hours=self.T)

        return df_results, total_surplus_cost, total_carbon_cost

    def pareto_front(self, t_start, t_end, lambda_values):
        """
        Runs the optimization for a set of Pareto weights (lambda values), and collects the
        aggregate cost components for each run.

        Parameters:
          t_start: Start datetime
          t_end: End datetime
          lambda_values: Iterable of lambda values (each in [0,1]) to sweep

        Returns:
          DataFrame with columns: 'lambda', 'TotalSurplusCost', 'TotalCarbonCost'
        """
        pareto_results = []

        for lam in lambda_values:
            # Reset the SoC initial conditions for each Pareto run
            self.reset_initial_conditions()
            # Run the receding horizon optimization for the given lambda
            _, tot_surplus, tot_carbon = self.opt(t_start, t_end, lam)
            pareto_results.append({
                'lambda': lam,
                'TotalSurplusCost': tot_surplus,
                'TotalCarbonCost': tot_carbon
            })
            print(f'Pareto run with lambda={lam} finished.')

        return pd.DataFrame(pareto_results)
