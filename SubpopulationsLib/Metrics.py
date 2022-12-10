import numpy as np

def MAPE(y_real, y_predicted):
	'''
	Description:
		MAPE: mean absolute percentage error between the predictions and 
		ground truth

	Inputs:
		y_real - A numpy array containing the ground truth
		y_predicted - A numpuy array containing the predictions

	Output:
		MAPE
		standard deviation
	'''
	error = y_real - y_predicted
	absolute_error = np.abs(error)

	percentage_error = 100*absolute_error/y_real

	MAPE = np.mean(percentage_error)
	average_deviation = np.std(percentage_error)

	return MAPE, average_deviation