from functools import partial
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle

from DataProcessing import create_SIR_data
from Subpopulations import create_gaussian_dictionary, \
	find_theta_dictionary, find_theta_sa
from Subpopulations import mixture_exponentials, mixture_SIR
from Metrics import MAPE

from InfectiousModels import f_SIR, df_SIR, g_SIR, dg_SIR, learn_SIR, \
	compute_A_SIR, compute_A_2_Subpop, Q_mat_SIR, R_mat_SIR, SIR
from KalmanFilter import ExtendedKalmanFilter
from Metrics import MAPE

import warnings
warnings.filterwarnings("ignore")

def run_prediction_experiment(countries):
	results_dict = dict()
	
	for country_name in countries:
		try:
			results_dict[country_name] = dict()

			print(country_name)
			S, I, R = create_SIR_data(country_name, file_path, lookup_table_path, start_date, end_date)

			indexes_weekly = np.arange(0,S.shape[0],7)

			S = S[indexes_weekly]
			I = I[indexes_weekly]
			R = R[indexes_weekly]

			data = I[1:]

			# Set the data for the SIR model
			# -----------------------------------
			# Use the first observation as an estimation of the first state.
			mu_init_SIR = np.array([[S[0]], [I[0]], [R[0]]])

			s = np.square(mu_init_SIR[1,0] * 0.1/3)
			sigma_init_SIR = s*np.eye(mu_init_SIR.shape[0])

			# Use the rest of the data as the observations
			num_observations = len(S)
			Observations = list()

			for i in range(1, num_observations):
				instance = np.array([[S[i]],[I[i]],[R[i]]])
				Observations.append(instance)

			Observations = np.array(Observations)
			params = dict()
			params['epochs'] = 20
			params['print_ll'] = False
			# ------------------------------------
			min_observ = 5
			max_k = 4
			gt = list()
			predictions = list()
			slow = list()

			for i in range(4):
				results_dict[country_name]['Week ' + str(i+1)] = dict()

				results_dict[country_name]['Week ' + str(i+1)]['GT'] = list()

				results_dict[country_name]['Week ' + str(i+1)]['Gaussian_dict'] = list()
				results_dict[country_name]['Week ' + str(i+1)]['SIR_dict'] = list()
				results_dict[country_name]['Week ' + str(i+1)]['Gaussian_mix'] = list()
				results_dict[country_name]['Week ' + str(i+1)]['SIR_mix'] = list()

				results_dict[country_name]['Week ' + str(i+1)]['SLOW'] = list()
				results_dict[country_name]['Week ' + str(i+1)]['SIR'] = list()

			for i in range(min_observ, data.shape[0]-max_k):
				# ----------- SIR --------------------
				sir = SIR()
				sir.train(S[1:i], I[1:i], R[1:i])

				# ----------------------------------------

				# -------- Dictionary --------------------
				# Get the current observations
				c_data = np.array(data[0:i])
				bias = np.min(c_data)
				norm_I = c_data - bias

				# Using the dicionary, find the parameters theta that best explain
				# the data
				c_D_Gaussian = D_Gaussian[:,0:i]
				c_D_SIR = D_SIR[:,0:i]

				# Fit with Gaussians and SIR's with dictionaries
				theta_Gaussian = find_theta_dictionary(c_D_Gaussian, norm_I, lambda_reg = 1E-3)
				theta_SIR = find_theta_dictionary(c_D_SIR, norm_I, lambda_reg = 1E-3)

				# ----------- Mixture of models -----------
				# ---------------------------------
				# Set the parameters for the mixture of signals approach
				# ---------------------------------
				num_mixtures = 3

				# For the Gaussian model
				bounds_mu = (0,i+4)
				bounds_sigma = (1,6)
				bounds_coef = (0,300000)

				bound_list_Gaussian = [bounds_mu, bounds_sigma, bounds_coef]

				bounds_Gaussian = list()

				for element in bound_list_Gaussian:
					for j in range(num_mixtures):
						bounds_Gaussian.append(element)
						
				# For the SIR model
				bound_S = (0,1E8)
				bound_beta = (0,1)
				bound_gamma = (0,1)
				bound_coef = (0,1000)
				bound_k = (0,i+4)
				bound_list_SIR = [bound_S, bound_beta, bound_gamma, bound_coef, bound_k]

				bounds_SIR = list()

				for element in bound_list_SIR:
					for j in range(num_mixtures):
						bounds_SIR.append(element)


				# Fit with mixtuers of models
				params_gaussian = find_theta_sa(bounds_Gaussian, norm_I, mixture_exponentials)
				params_SIR = find_theta_sa(bounds_SIR, norm_I, mixture_SIR)

				# -----------------------------------------
				# Make the predictions in advance
				S_last = S[i-1]
				I_last = I[i-1]
				R_last = R[i-1]

				for k in range(1, max_k + 1):
					S_last, I_last, R_last = sir.predict_next(S_last, I_last, R_last)

					x_hat_Gaussian_Dict = np.matmul(D_Gaussian[:, i-1 + k].T, theta_Gaussian) + bias
					x_hat_SIR_Dict = np.matmul(D_SIR[:, i-1 + k].T, theta_SIR) + bias

					c_T = len(c_data)
					y_hat_Gaussian_Mix = mixture_exponentials(params_gaussian, c_T + k) + bias
					y_hat_SIR_Mix = mixture_SIR(params_SIR, c_T + k) + bias

					results_dict[country_name]['Week ' + str(k)]['GT'].append(data[i-1 + k])

					results_dict[country_name]['Week ' + str(k)]['Gaussian_dict'].append(x_hat_Gaussian_Dict)
					results_dict[country_name]['Week ' + str(k)]['SIR_dict'].append(x_hat_SIR_Dict)
					results_dict[country_name]['Week ' + str(k)]['Gaussian_mix'].append(y_hat_Gaussian_Mix[-1])
					results_dict[country_name]['Week ' + str(k)]['SIR_mix'].append(y_hat_SIR_Mix[-1])

					results_dict[country_name]['Week ' + str(k)]['SLOW'].append(c_data[-1])

					results_dict[country_name]['Week ' + str(k)]['SIR'].append(I_last)
		except:
			print('Error processing ', country_name)
	return results_dict

# Set the parameters of this experiment
start_date = '7/30/20'
end_date = '7/30/21'

file_path = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
lookup_table_path = '../COVID-19/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'

# Get the name of all available countries in the dataset
dataframe = pd.read_csv(lookup_table_path)

country_names = np.unique(dataframe['Country_Region'].values)
num_countries = len(country_names)

country_list = list()

for i in range(0, num_countries, 25):
	country_list.append(country_names[i:i+25])

# -------------------------------------------------
# Set the parameters for the Dictionary approaches
# -------------------------------------------------
# Create a dictionaty with Gaussians
T = 52
possible_mu = np.arange(0,52,2)
possible_sigma = np.arange(1,30,2)
possible_skewness = np.array([0])

D_Gaussian = create_gaussian_dictionary(
	T, possible_mu, possible_sigma, possible_skewness
)

# Load the dictionary with SIR
D_SIR = pickle.load(open('./D_SIR.pkl', 'rb'))

pool = mp.Pool(8)
results = pool.map(run_prediction_experiment, [country_partial for country_partial in country_list])
pool.close()

pickle.dump(results, open('./results_bounded_mix.pkl', 'wb'))