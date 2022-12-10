import numpy as np
import pandas as pd

'''
------------------------------------------------------
Loading and preprocessing code
------------------------------------------------------
'''

def create_SIR_data(country_name, file_path, lookup_table_path, start_date, end_date):
	# Load the lookup table
	lookup_table = pd.read_csv(lookup_table_path)
	
	# Load the data from JHU	
	csv_confirmed_path = file_path + 'time_series_covid19_confirmed_global.csv'
	csv_deaths_path = file_path + 'time_series_covid19_deaths_global.csv'
	csv_recovered_path = file_path + 'time_series_covid19_recovered_global.csv'
	lookup_table = pd.read_csv(lookup_table_path)

	confirmed_dataframe = pd.read_csv(csv_confirmed_path)
	deaths_dataframe = pd.read_csv(csv_deaths_path)
	recovered_dataframe = pd.read_csv(csv_recovered_path)

	dates = np.array(confirmed_dataframe.keys()[4:])

	# Get the population of the desired country
	lookup_flag_confirmed = lookup_table['Combined_Key'] == country_name
	Population = lookup_table.loc[lookup_flag_confirmed,]['Population'].values[0]

	# Get the relevant data for the SIR model
	country_flag_confirmed = confirmed_dataframe['Country/Region'] == country_name
	country_confirmed_dataframe = confirmed_dataframe.loc[country_flag_confirmed,]

	country_flag_deaths = deaths_dataframe['Country/Region'] == country_name
	country_deaths_dataframe = deaths_dataframe.loc[country_flag_deaths,:]

	country_flag_recovered = recovered_dataframe['Country/Region'] == country_name
	country_recovered_dataframe = recovered_dataframe.loc[country_flag_recovered,:]

	# Extract the numbers
	confirmed = country_confirmed_dataframe.sum().to_numpy()[4:]
	deaths = country_deaths_dataframe.sum().to_numpy()[4:]
	recovered = country_recovered_dataframe.sum().to_numpy()[4:]
	dates_names = country_confirmed_dataframe.columns[4:]

	removed = deaths + recovered

	# Get only the days in the desired interval
	start_indx = np.where(dates_names == start_date)[0][0]
	end_indx = np.where(dates_names == end_date)[0][0]

	confirmed = confirmed[start_indx:end_indx]
	deaths = deaths[start_indx:end_indx]
	recovered = recovered[start_indx:end_indx]
	removed = removed[start_indx:end_indx]
	dates_names = dates_names[start_indx:end_indx]

	# Now make the data reflect the number of active cases
	number_days = len(confirmed)

	S = []; I = []; R =[]

	S.append(Population - confirmed[0] - removed[0])
	I.append(confirmed[0])
	R.append(removed[0])

	for i in range(1, number_days):
	    new_I = confirmed[i] - confirmed[i-1]
	    new_R = removed[i] - removed[i-1]

	    S.append(S[i-1] - new_I)
	    I.append(I[i-1] + new_I - new_R)
	    R.append(R[i-1] + new_R)

	S = np.array(S)
	I = np.array(I)
	R = np.array(R)

	return S, I, R