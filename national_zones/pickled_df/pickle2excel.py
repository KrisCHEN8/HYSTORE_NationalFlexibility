import pandas as pd

emissions_pkl = ['emission_AUT_2022.pkl',
                 'emission_ESP_2022.pkl',
                 'emission_IT_2022.pkl',
                 'emission_SE_2022.pkl',]

for df in emissions_pkl:
    df_path = r'C:\Users\yangzhec\OneDrive - KTH\Projects\HYSTORE\HYSTORE_NationalFlexibility\national_zones\pickled_df\\' + df
    emission = pd.read_pickle(df_path)
    # Save to Excel
    excel_path = r'C:\Users\yangzhec\OneDrive - KTH\Projects\HYSTORE\HYSTORE_NationalFlexibility\national_zones\pickled_df\\' + df.replace('.pkl', '.xlsx')
    emission.to_excel(excel_path, index=False)
    print(f"Saved {df} to {excel_path}")
