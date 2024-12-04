# HYSTORE_NationalFlexibility
National flexibility assessment leveraging TCM (Thermalchemical material) and PCM (Phase change material). The roadmap for urban sustainability involves the transition to reliable and decarbonised energy networks. In this regard, business concepts based on sector coupling through the use of TES systems can play a key role. This research is placed in this context, with the aim of evaluating the flexibility potential of novel thermal energy storage systems in order to provide load shifting and peak shaving services to the electricity grid. The idea involves the modeling of the TES upscaling scenarios on the national territory and the simulation of energy demand starting from real data on the electricity grid from European TSOs. 


## Project Structure

Here is an overview of the project structure:

- **COP&EER/**: Contains files related to the Coefficient of Performance (COP) and Energy Efficiency Ratio (EER).
- **national_zones/**: Contains pickled data files and scripts for different national zones, including Spain, Italy, Austria, and Sweden.
- **res_v1/**: Annual simulation results for capacity of version 1.
- **res_v2/**: Annual simulation results for capacity of version 2.
- **Dati_capacities_scenarios.xlsx**: Excel file containing data scenarios on capacities.
- **Austria_optimization.py**: Optimization simulation script specific to Austria.
- **Italy_optimization.py**: Optimization simulation script specific to Italy.
- **Spain_optimization.py**: Optimization simulation script specific to Spain.
- **Sweden_optimization.py**: Optimization simulation script specific to Sweden.
- **data_agg.ipynb**: Jupyter notebook for data aggregation and preprocessing.
- **predictive_optimization.py**: Script for optimization formulation.
- **thermal_demand_per_country_electricity.csv**: CSV file with data on thermal demand per country.
