import os
import pandas as pd

# Get all .xlsx files in the current working directory
cwd = os.getcwd()
xlsx_files = [f for f in os.listdir(cwd) if f.endswith('.xlsx')]

for file in xlsx_files:
    try:
        # Load all sheets
        xls = pd.read_excel(file, sheet_name=None, engine='openpyxl')

        # Extract Sheet1 (case-insensitive check)
        sheet1_name = next((s for s in xls if s.lower() == 'sheet1'), None)
        if sheet1_name is None:
            print(f"Sheet1 not found in {file}")
            continue

        df1 = xls[sheet1_name]

        # Check if the required columns are present
        if 'epsilon_h' not in df1.columns or 'epsilon_c' not in df1.columns:
            print(f"Columns not found in {file}")
            continue

        # Extract the columns for Sheet2
        df2 = df1[['epsilon_h', 'epsilon_c']]

        # Remove the columns from Sheet1
        df1_modified = df1.drop(columns=['epsilon_h', 'epsilon_c'])

        # Write back to the same file:
        with pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Replace Sheet1 with the modified version
            df1_modified.to_excel(writer, sheet_name=sheet1_name, index=False)
            # Create or replace Sheet2 with the extracted columns
            df2.to_excel(writer, sheet_name='Sheet2', index=False)

        print(f"Processed {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
