import os
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient


#parameters
start = pd.Timestamp('20220101', tz='Europe/Rome')
end = pd.Timestamp('20230101', tz='Europe/Rome')
country_code = 'IT_SICI' 
range_time_load="15T"
range_time_generation="15T"


column_names_load = ["Actual load"]
column_names_generation=["Biomass","Fossil Coal-derived gas","Fossil Gas","Fossil Oil",
                         "Hydro Pumped Storage charge","Hydro Pumped Storage discharge","Hydro Run-of-river and poundage","Other","Solar","Wind Onshore"]

#initialise client
client = EntsoePandasClient(api_key='5b6c09de-3782-4898-a937-dcb08794deba')

#make queries to generate the dataframes
df_load = client.query_load(country_code, start=start, end=end)
df_generation=client.query_generation(country_code, start=start, end=end, psr_type=None)
#df_generation=pd.read_csv('df_generation.txt',skiprows=2,header=None)
#df_generation.to_csv('df_generation_pandas.txt')
current_columns_generation = df_generation.columns.tolist()
#df_generation.to_csv('df_generation.txt')
current_columns_load = df_load.columns.tolist()

# Replace the column names
df_generation.columns = column_names_generation
df_load.columns = column_names_load
# 
df_generation_initial=df_generation.copy()
#Add index values for future use in plots
df_generation['Time'] = df_generation.index.values

print(df_generation)

#df_generation['Time'] = df_generation.index.values

#create the simplified dataframe with resources joined by type
column_indices_fossil = [1,2,3,7]
column_indices_renewables=[0,6,8,9]
column_indices_storage=[4]

sum_fossil = df_generation.iloc[:, column_indices_fossil].sum(axis=1)
sum_renewables = df_generation.iloc[:, column_indices_renewables].sum(axis=1)
sum_storage=df_generation.iloc[:, column_indices_storage].sum(axis=1)

#sum the fossil sources
summed_fossil = pd.DataFrame({'sum_fossil': sum_fossil})
df_generation['Fossil']=summed_fossil['sum_fossil']

#sum renewables
summed_renewables = pd.DataFrame({'sum_renewables': sum_renewables})
df_generation['Renewables']=summed_renewables['sum_renewables']

#sum storage
summed_storage = pd.DataFrame({'sum_storage': sum_storage})
df_generation['Storage']=summed_storage['sum_storage']

#create the simplified dataframe joining resources by type

#join the dataframes (complete version)
#df_load.drop(df_load.columns[0], axis=1, inplace=True)
frames=[df_generation,df_load]
df_complete=pd.concat(frames, axis=1)
df_complete.to_csv('df_complete.txt')
#extract the columns for simplified datafame
columns_to_extract= ['Time','Fossil','Renewables','Storage','Hydro Pumped Storage charge','Actual load']

#join  the dataframes (simplified)
df_simplified=df_complete[columns_to_extract].copy()

#calculate the surplus of renewables
df_simplified_calc = df_simplified.copy()
df_simplified_calc['surplus_all'] = df_simplified_calc['Fossil'] + df_simplified_calc['Renewables'] - df_simplified_calc['Actual load']
df_simplified_calc['surplus_RES'] = df_simplified_calc['Renewables'] - df_simplified_calc['Actual load']
# 
# df_simplified['surplus_all']= df_simplified['Fossil']+df_simplified['Renewables']-df_simplified['Actual load']
# df_simplified['surplus_RES']= df_simplified['Renewables']-df_simplified['Actual load']

#overall sums
overall_surplus=df_simplified_calc.loc[df_simplified_calc['surplus_all'] >0, 'surplus_all'].sum()
RES_surplus=df_simplified_calc.loc[df_simplified_calc['surplus_RES'] >0, 'surplus_RES'].sum()
hours_surplus_all = len(df_simplified_calc[df_simplified_calc['surplus_all'] >0])
hours_surplus_RES = len(df_simplified_calc[df_simplified_calc['surplus_RES'] >0])

#################charts sections
#
import matplotlib.pyplot as plt

fig, (ax1,ax2,ax3) = plt.subplots(3)

##ax1: overall stacked area. The *.T in the ax.stackplot command is needed because the area is created by transposing the dataframe
x_stacked=df_simplified_calc['Time']
columns_to_plot_ax1=['Fossil','Renewables','Storage','Actual load']
y_stacked=df_simplified_calc[columns_to_plot_ax1]
ax1.stackplot(x_stacked, y_stacked.T, labels=columns_to_plot_ax1)
# Add labels and title
ax1.set_xlabel("Time")
ax1.set_ylabel("Energy /Wh")
ax1.set_title("Energy Generation, Storage and Consumption Over Time")

# Add legend
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),ncol=len(columns_to_plot_ax1))

## Autoscale the axes
ax1.autoscale(enable=True, axis="both", tight=True)

##ax2: surplus
columns_to_plot_ax2=['surplus_all','surplus_RES']
y_stacked_ax2=df_simplified_calc[columns_to_plot_ax2]
ax2.stackplot(x_stacked, y_stacked_ax2.T, labels=columns_to_plot_ax2)
# Add labels and title
ax2.set_xlabel("Time")
ax2.set_ylabel("Energy /Wh")
ax2.set_title("Surplus Over Time")

# Add legend
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),ncol=len(columns_to_plot_ax2))

## Autoscale the axes
ax2.autoscale(enable=True, axis="both", tight=True)

##ax3: surplus vs storage
columns_to_plot_ax3=['surplus_all','surplus_RES','Storage','Hydro Pumped Storage charge']
y_stacked_ax3=df_simplified_calc[columns_to_plot_ax3]
ax3.stackplot(x_stacked, y_stacked_ax3.T, labels=columns_to_plot_ax3)
# Add labels and title
ax3.set_xlabel("Time")
ax3.set_ylabel("Energy /Wh")
ax3.set_title("Surplus and Storage Over Time")

# Add legend
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),ncol=len(columns_to_plot_ax2))

## Autoscale the axes
ax3.autoscale(enable=True, axis="both", tight=True)

# Adjust vertical spacing between subplots
plt.subplots_adjust(hspace=0.8)  # You can adjust the value as needed
plt.savefig("surplus.png", dpi=300, bbox_inches='tight')
# Show the plot
plt.show()

##create a heatmap with seaborn
import seaborn as sns
# Resample the dataframe to hourly mean
df_simplified_hourly = df_simplified_calc.resample('h').mean()
df_simplified_hourly['month'] = df_simplified_hourly.Time.dt.month
df_simplified_hourly['day'] = df_simplified_hourly.Time.dt.day
df_simplified_hourly['hour'] = df_simplified_hourly.Time.dt.hour
df_simplified_hourly['week'] = df_simplified_hourly.index.isocalendar().week
# Extract hour of the week (0 to 167) from the datetime index
df_simplified_hourly['hour_of_week'] = (df_simplified_hourly.index.dayofweek * 24) + df_simplified_hourly.index.hour


all_week_year_df = pd.pivot_table(df_simplified_hourly, values="surplus_RES",
                                   index=["hour_of_week"],
                                   columns=["week"],
                                   fill_value=0,
                                   margins=True)
ax = sns.heatmap(all_week_year_df, cmap='RdYlGn',
                 robust=True,
                 fmt='.2f',
                 annot=False,
                 linewidths=.5,
                 annot_kws={'size':11},
                 cbar_kws={'shrink':.8,
                           'label':'Surplus (MWh)'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=14)

# Reverse the y-axis
ax.invert_yaxis()

# Increase font size for the color bar (legend) and axes labels
ax.figure.axes[-1].yaxis.label.set_size(14)  # Font size for color bar label
ax.tick_params(axis='both', labelsize=14)  # Font size for x and y axis labels
#plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')

plt.show()
# 
##create stacked plot with monthly values
df_generation_hourly = df_generation_initial.resample('h').mean()

# Resample simplified hourly dataframe to monthly sums
monthly_sums = df_generation_hourly.groupby(pd.Grouper(freq="ME")).sum()

# Create a stacked area plot
plt.figure(figsize=(10, 6))
plt.stackplot(monthly_sums.index, monthly_sums.T, labels=monthly_sums.columns, cmap='tab20c')
plt.xlabel("Time")
plt.ylabel("Energy Production/ MWh")
# Set x-axis ticks to show month names
plt.xticks(ticks=monthly_sums.index, labels=monthly_sums.index.strftime("%b %Y"), rotation=45, ha="right")
plt.legend(loc='upper right')
plt.tight_layout()

plt.title("Monthly Energy Production by Source")

plt.show()

df_simplified_calc.to_pickle('df_IT_SICI.pkl')