import os
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient


dict_columns = {
    'AT': [0, 2, 4, 6, 8, 10, 12, 14, 17, 18, 20, 22, 24],
    'ES': [1,2,3,4,5,6,7,15,0,8,11,13,14,16,17,18,19,20,9,12],
    'SE_4': [0,1,2,3,4],
    'SE_3': [0,1,2,3,4,5,6],
    'SE_2': [0,1,2,3,4],
    'SE_1': [0,1,2,3],
    'IT_CSUD': [1,2,3,8,0,6,9,10,4,7],
    'IT_SICI': [1,2,3,7,0,6,8,9,4],
    'IT_SUD': [1,2,3,5,8,0,6,9,10,7],
    'IT_SARD': [1,2,3,8,0,6,9,11,4,7],
    'IT_NORD': [1,2,3,4,9,0,7,10,11,12,5,8],
    'IT_CNOR': [1,2,6,0,3,4,7,8,5]
}


class EntsoeData:
    def __init__(self, country, start, end, range_time_generation="1h"):
        self.client = EntsoePandasClient(api_key='a817ac5b-7d6a-4192-8d24-ff6a990f7d31')
        self.country = country
        self.start = start
        self.end = end
        self.range_time_generation = range_time_generation

    def get_generation_data(self, zone_name=None) -> pd.DataFrame:
        if zone_name is None:
            self.country_code = self.country
        else:
            self.country_code = self.country + "_" + zone_name

        df_generation = self.client.query_generation(self.country_code, start=self.start, end=self.end, psr_type=None)
        date_index = pd.date_range(start=self.start, periods=len(df_generation), freq=self.range_time_generation)
        df_generation.index = date_index
        # print(df_generation.columns)

        if isinstance(df_generation.columns, pd.MultiIndex):
            df_generation.columns = [
               (
                    "Hydro Pumped Storage charge" if col[0] == "Hydro Pumped Storage" and col[1] == "Actual Aggregated"
                     else "Hydro Pumped Storage discharge" if col[0] == "Hydro Pumped Storage" and col[1] == "Actual Consumption"
                    else col[0]
                )
                for col in df_generation.columns
            ]

        else:
            pass

        useful_cols = dict_columns[self.country_code]
        df_generation = df_generation.iloc[:, useful_cols]
        df_generation.fillna(0, inplace=True)

        print(df_generation.columns)

        print(f'{self.country_code} obtained.')

        return df_generation

    def aggregate(self, generation_list: list[pd.DataFrame]) -> pd.DataFrame:
        try:
            # Check if the list is empty.
            if not generation_list:
                raise ValueError("The dataframe list is empty.")

            # Create a union of all columns from all dataframes.
            all_columns = set()
            for df in generation_list:
                all_columns.update(df.columns)
            # Optionally, sort the columns for a consistent order.
            all_columns = sorted(all_columns)

            print(all_columns)

            # Reindex each DataFrame so that it contains all columns, filling missing columns with 0.
            standardized_list = [
                df.reindex(columns=all_columns, fill_value=0) for df in generation_list
            ]

            # Create an aggregated DataFrame by adding the dataframes element-wise.
            df_aggregated = standardized_list[0].copy()
            for df in standardized_list[1:]:
                df_aggregated = df_aggregated.add(df, fill_value=0)

            return df_aggregated

        except Exception as e:
            raise e


def calculate_emission_factor(df: pd.DataFrame) -> pd.DataFrame:
    other_emission = 870 * 1e3 * 1e-6   # ton CO2/MWh  coal
    fossilgas_emission = 368.3 * 1e3 * 1e-6  # ton CO2/MWh
    fossiloil_emission = 548.9 * 1e3 * 1e-6  # ton CO2/MWh

    df_generation = df.copy()

    # If the 'Other' column is present, calculate co2 emissions
    if 'Other' in df_generation.columns:
        df_generation['co2_other'] = df_generation['Other'] * other_emission

    # If the 'Fossil Coal-derived gas' column is present, calculate co2 emissions.
    # Note: In your example, the same `other_emission` is applied.
    if 'Fossil Coal-derived gas' in df_generation.columns:
        df_generation['co2_fossilcoal'] = df_generation['Fossil Coal-derived gas'] * other_emission

    # If the 'Fossil Gas' column is present, calculate co2 emissions.
    if 'Fossil Gas' in df_generation.columns:
        df_generation['co2_fossilgas'] = df_generation['Fossil Gas'] * fossilgas_emission

    # If the 'Fossil Oil' column is present, calculate co2 emissions.
    if 'Fossil Oil' in df_generation.columns:
        df_generation['co2_fossiloil'] = df_generation['Fossil Oil'] * fossiloil_emission

    if 'Fossil hard coal' in df_generation.columns:
        df_generation['co2_fossilhcoal'] = df_generation['Fossil hard coal'] * other_emission

    # sum the emissions
    emission_cols = ['co2_other', 'co2_fossilcoal', 'co2_fossilgas', 'co2_fossiloil', 'co2_fossilhcoal']
    existing_cols = [col for col in emission_cols if col in df_generation.columns]
    df_generation['co2_total'] = df_generation[existing_cols].sum(axis=1)

    if 'Hydro Pumped Storage discharge' in df_generation.columns:
        # calculate the emission factor
        df_generation['emission_factor'] = (
            df_generation['co2_total'] /
            df.drop(columns=['Hydro Pumped Storage discharge']).sum(axis=1)
        )
    else:
        df_generation['emission_factor'] = df_generation['co2_total'] / df.sum(axis=1)

    return df_generation


if __name__ == "__main__":
    start = pd.Timestamp('20220101', tz='Europe/Rome')
    end = pd.Timestamp('20230101', tz='Europe/Rome')

    entsoe_data = EntsoeData('AT', start, end, range_time_generation="1h")
    df_generation = entsoe_data.get_generation_data(austria=True)
    print(df_generation.columns)
