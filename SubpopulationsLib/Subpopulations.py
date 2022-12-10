import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing
from scipy.stats import skewnorm
from functools import partial

from InfectiousModels import f_SIR_Kronecker

def create_gaussian_dictionary(T=52, mu_list=None, sigma_list=None, skewness_list=None):
	'''
	Description:
		This function creates a dictionary with normalized Gaussian pdf's. It will create
		a Gaussian for every possible combination of the elements of mu_list, sigma_list,
		and skewness_list. Every Gaussian will have a length T.

	Input:
		T - Length of every time-series in the dictionary
		mu_list - List that includes the mean of the Gaussian distributions.
			The default value is [0, 2, 4, ..., T]
		sigma_list - List that includes the standard deviation of the Gaussian distributions
			The default value is [1, 3, 5, ..., T/2]
		skewness_list - list that includes the skewness of the Gaussian distributions
			The default value is [0]
	Output:
		D -  a numpy array with len(mu_list) * len(sigma_list) * len(skewness_list) time series,
			each of length T. It contains one time-series per every possible combination of the
			parameters.
	'''

	# When some parameter is not given, use the default values.
	if mu_list is None:
		possible_mu = np.arange(0, T, 2)
	else:
		possible_mu = np.array(mu_list)

	if sigma_list is None:
		possible_sigma = np.arange(1, int(T/2), 2)
	else:
		possible_sigma = np.array(sigma_list)

	if skewness_list is None:
		possible_skewness = np.array([0])
	else:
		possible_skewness = np.array(skewness_list)

	# Create a sequence x = [0, 1, 2, ..., T]
	x = np.arange(0, T, 1)

	# Create an empty list that will be converted into a dictionary
	D = list()

	# Create a time-series for every possible combination of the parameters
	for mu in possible_mu:
		for sigma in possible_sigma:
			for skewness in possible_skewness:
				# Create the pdf
				pdf = skewnorm.pdf(x, skewness, mu, sigma)
				Z = np.max(pdf)
				# Append the time-series to the dictionary
				D.append(pdf/Z)


	# Transform the dictionary into a numpy array
	D = np.array(D)

	return D

def mse_dual_annealing(params, T, y, mixture_fun):
	'''
	Description:
		This function is used along with the optimization algorithm dual_annealing
		to find the parameters that optimize the MSE between a time-series, y,
		and a mixture of signals of time T.
	'''
	y_hat = mixture_fun(params, T)
	error = y - y_hat
	MSE = np.dot(error, error)
	
	return MSE

def regularized_mse(theta, D, x, lambda_reg=0, W=None):
	'''
	Description:
		This function computes the regularized mean squared error:
		reg_mse = || x - D^T theta||_2^2 + lambda_reg * || theta ||_2^2

	Inputs:
		D - A numpy array of size (num_timeseries, T), containing a set of basis functions.
			Each time-series has length T.
		theta - Vector of length (num_timeseries) that contains the coefficient associated
			to each time-series in D.
		x - Original time-series to be reconstructed with the dictionary
		lambda-reg - Regularization parameter
		W - A vector of length T that has a weight associated to each time-point in x.
			The default value is 1 per each point.

	Output:
		reg_mse - regularized mean squared error between the original image, x, and its
			reconstruction x_hat.
	'''

	# Reconstruct x_hat with the dictionary and the coefficients.
	x_hat = np.dot(D.T, theta)

	# Compute the signed error of each element in the time-series with its reconstruction
	error = x - x_hat

	# Apply the weight to the errors
	if W is None:
		error_weights = np.eye(len(x))
	else:
		error_weights = np.diag(W)

	weighted_error = np.dot(error, error_weights)

	# Compute the mean squared error
	mse = np.dot(weighted_error, error) / len(error)

	# Compute the regularization term
	reg = lambda_reg * np.dot(theta, theta)

	reg_mse = mse + reg
	
	return reg_mse

def grad_regularized_mse(theta, Dx, DDT, lambda_reg):
	'''
	Description:
		This dunction computes the gradient of the mse + reg

	Inputs:
		theta - Vector of length (num_timeseries) that contains the coefficient associated
			to each time-series in D.
		Dx - Matrix multiplication of the dictionary D, and the time-series x
		DDT - Matrix multipliaction of the dictionaty, D, and its transpose
		lambda-reg - Regularization parameter

	Outputs:
		grad - Gradient of the cost function wrt theta
	'''

	grad = -2*Dx + 2*np.dot(DDT, theta) + 2*lambda_reg*theta

	return grad

def find_theta_dictionary(D, x, lambda_reg=0, W=None):
	'''
	Description:
		This function finds the parameters, theta, that minimize the weighted,
		regularized, mean squared error.

	Inputs:
		D - Dictionary. A numpy array of size (num_timeseries, length_time_series)
		x - signal to be modeled
		lamda_reg - regularization parameter
		W - Weights applied to each timepoint in x. Vector of length (length_timeseries)

	Output:
		theta - numpy array of length (num_timeseries) that minimize the 
		weighted, regularized mean squared error.
	'''
	num_timeseries, num_timepoints = D.shape

	# Compute some auxiliary variables
	if W is None:
		weights = np.eye(num_timepoints)
	else:
		weights = np.diag(W)

	DW = np.dot(D, weights)
	DWDT = np.matmul(DW, D.T)
	DWx = np.matmul(DW, x)

	par_cost = partial(regularized_mse, D=D, x=x, lambda_reg=lambda_reg, W=W)
	par_grad = partial(grad_regularized_mse, Dx=DWx, DDT=DWDT, lambda_reg=lambda_reg)
	theta_init = np.zeros(num_timeseries)

	# Set the constraints
	lb = np.zeros(num_timeseries)
	ub = np.inf*np.ones(num_timeseries)
	bounds = Bounds(lb, ub)

	# Optimize the cost function with the given constraints
	res = minimize(par_cost, x0=theta_init, jac=par_grad, bounds=bounds)
	theta = res.x

	return theta

def find_theta_sa(bounds, y, mixture_fun, maxiter=1000):
	'''
	Description:
		This function finds the parameters that minimize the MSE between the ground
		truth time-series and the predictions of a mixture model.

	'''
	T = len(y)
	optimization_fun = partial(mse_dual_annealing, T=T, y=y, 
		mixture_fun=mixture_fun)
	opt = dual_annealing(optimization_fun, bounds, maxiter=maxiter)
	params = opt.x

	return params

def mixture_exponentials(params, T):
	'''
	Description:
		This function creates a mixture of Gaussian exponentials evaluated over
		a time-series of length T.
	Inputs:
		params - parameters of the exponentials 
				[(mu_1, sigma_1, coef_1), 
				 (mu_2, sigma_2, coef_2), 
				 ...]
		T - length of the time series
	'''
	# Create the time series
	x = np.arange(0, T, 1)

	# Detect the number of exponentials
	k = int(len(params)/3)
	
	# Compute the sum of exponentials
	output = 0.0
	for i in range(k):
		exp_term = -0.5 * np.square(( x - params[i])/ params[i+k])
		output += params[i+2*k]*np.exp(exp_term)
		
	return output

def mixture_SIR(params, T):
	'''
	Description:
		This function creates a mixture of SIR models evaluated over 
		a time-series of length T.
	Inputs:
		params - parameters of the SIR 
				[(S_1, beta_1, gamma_1, coef_1, k_1), 
				 (S_2, beta_2, gamma_2, coef_2, k_2), 
				 ...]
		T - length of the time series
	'''

	# Detect the number of SIR models
	n = int(len(params)/5)
	
	# Compute the sum of exponentials
	output = 0.0
	for i in range(n):
		S = params[i]
		beta = params[i+n]
		gamma = params[i+2*n]
		coef = params[i+3*n]
		k = int(params[i+4*n])
		
		X = np.array([S, 0, 0])
		theta = np.array([beta, gamma, coef, k])
		
		data = f_SIR_Kronecker(X, theta, T)
		output += data
		
	return output[:,1]

