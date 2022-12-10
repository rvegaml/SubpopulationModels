import numpy as np

# Set the error matrices for the process
def Q_mat_SIR(x):
    s = np.square(x[1,0]*0.1/3)
    Q = np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, s]
    ])
    return Q

def R_mat_SIR(x):
    s = np.square(x[1,0]*0.1/3)
    R = np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, s]
    ])
    
    return R
    
def f_SIR(X, theta):
	'''
	This functions computes the next state, given the current state (S,I,R).
	X is a numpy array of dimensions 3 x 1
	'''
	S = X[0,0]; I=X[1,0]; R=X[2,0];
	P = S + I + R

	beta = theta[0,0]
	gamma = theta[1,0]

	# Compute the changes
	change_S = beta*I*S / P
	change_R = gamma * I
	
	S_next = S - change_S
	R_next = R + change_R    
	I_next = I + change_S - change_R
	
	return np.array([[S_next], [I_next], [R_next]])

def df_SIR(X, theta):
	S = X[0,0]; I=X[1,0]; R=X[2,0];
	P = S + I + R

	beta = theta[0,0]
	gamma = theta[1,0]

	dSS = 1 - beta*I/P
	dSI = -beta*S/P
	dSR = 0

	dIS = beta*I/P
	dII = 1 + beta*S/P - gamma
	dIR = 0

	dRS = 0
	dRI = gamma
	dRR = 1

	A = np.array([
		[dSS, dSI, dSR],
		[dIS, dII, dIR],
		[dRS, dRI, dRR]
		])

	return A

def g_SIR(X, theta):
	H = np.eye(3)

	return np.matmul(H, X)

def dg_SIR(X, theta):
	H = np.eye(3)

	return H

def g_SIR_hidden(X, theta):
	H = np.array([
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0],
		])

	return np.matmul(H, X)

def dg_SIR_hidden(X, theta):
	H = np.array([
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0],
		])

	return H

def f_SIR_subpop(X, theta):
	'''
	This functions computes the next state, given the current state.
	This model returns the hidden state and the observations
	X is a numpy array of dimensions 5 x 1
	beta is an array of dimensions (2,)
	gamma is a scalar
	'''
	S_p = X[0,0]; S_m=X[1,0]
	I_p = X[2,0]; I_m=X[3,0]
	R = X[4,0]
	P = np.sum(X[:,0])

	beta_p = theta[0,0]; beta_m = theta[1,0]
	gamma = theta[2,0]

	# Compute the changes
	change_S_p = beta_p*I_p*S_p / P
	change_S_m = beta_m*I_m*S_m / P

	change_R_p = gamma * I_p
	change_R_m = gamma * I_m
	
	S_p_next = S_p - change_S_p
	S_m_next = S_m - change_S_m
	
	I_p_next = I_p + change_S_p - change_R_p
	I_m_next = I_m + change_S_m - change_R_m
	
	R_next = R + change_R_p + change_R_m
	
	X_next = np.array([[S_p_next], [S_m_next], [I_p_next], [I_m_next], [R_next]])
	
	return X_next

def df_SIR_subpop(X, theta):
	S_p = X[0,0]; S_m=X[1,0]; 
	I_p = X[2,0]; I_m=X[3,0];
	R_a=X[4,0];
	P = np.sum(X[:,0])

	beta_p = theta[0,0]; beta_m = theta[1,0]
	gamma = theta[2,0]

	dSpSp = 1 - beta_p*I_p/P
	dSpSm = 0
	dSpIp = -S_p*beta_p/P
	dSpIm = 0
	dSpRa = 0

	dSmSp = 0
	dSmSm = 1 - beta_m*I_m/P
	dSmIp = 0
	dSmIm = -S_m*beta_m/P
	dSmRa = 0

	dIpSp = beta_p*I_p/P
	dIpSm = 0
	dIpIp = S_p*beta_p/P - gamma + 1
	dIpIm = 0
	dIpRa = 0

	dImSp = 0
	dImSm = I_m*beta_m/P
	dImIp = 0
	dImIm = S_m*beta_m/P - gamma + 1
	dImRa = 0

	dRaSp = 0
	dRaSm = 0
	dRaIp = gamma
	dRaIm = gamma
	dRaRa = 1

	A = np.array([
		[dSpSp, dSpSm, dSpIp, dSpIm, dSpRa],
		[dSmSp, dSmSm, dSmIp, dSmIm, dSmRa],
		[dIpSp, dIpSm, dIpIp, dIpIm, dIpRa],
		[dImSp, dImSm, dImIp, dImIm, dImRa],
		[dRaSp, dRaSm, dRaIp, dRaIm, dRaRa]
		])

	return A

def g_SIR_subpop(X, theta):

	H = np.array([
		[1, 1, 0, 0, 0],
		[0, 0, 1, 1, 0],
		[0, 0, 0, 0, 1]
	])

	return np.matmul(H, X)

def dg_SIR_subpop(X, theta):

	H = np.array([
		[1, 1, 0, 0, 0],
		[0, 0, 1, 1, 0],
		[0, 0, 0, 0, 1]
	])

	return H

# -------------------------------------------------------
# Learning the parameters of an SIR model given full data

def compute_A_SIR(last_X):
	# Get the parameters of the last state
	S=last_X[0,0]; I=last_X[1,0]; R=last_X[2,0]
	P = S + I + R

	A_x = np.array([
		[-S*I/P, 0],
		[S*I/P, -I],
		[0, I]
	])

	return A_x

def compute_A_2_Subpop(X):
	# Get the parameters of the last state
	S_1=X[0,0]; S_2=X[1,0]
	I_1=X[2,0]; I_2=X[3,0] 
	R_1=X[4,0]

	P = S_1 + S_2 + I_1 + I_2 + R_1

	A = np.array([
		[-S_1*I_1/P, 0, 0],
		[0, -S_2*I_2/P, 0],
		[S_1*I_1/P, 0, -I_1],
		[0, S_2*I_2/P, -I_2],
		[0, 0, I_1 + I_2]
		])

	return A

def learn_SIR(X_seq, O_seq, compute_A, epsilon=0.05):
	'''
	This function learns the parameters beta and gamma of an SIR model.
	It assumes that the observation model is known.

	Inputs:
		- X_seq is an array of size (num_obs+1, 3, 1)
		- O_seq is an array of size (num_obs  , 3, 1)
		- epsilon is a constant that determines the percentage of obsderved infected that 
			will be used for computing Q_t. If it is none, then Q_t = np.eye(3)
	'''

	num_obs = O_seq.shape[0]
	dim_X = X_seq.shape[1]

	term_1 = None
	term_2 = None

	for i in range(1, num_obs):
		last_X = X_seq[i-1]
		c_X = X_seq[i]
		c_O = O_seq[i]

		# Compute the matrix A_{t-1}
		A_t_1 = compute_A(last_X)

		# Get the noise matrix Q_t
		if epsilon is None:
			s = 1.0
		else:
			s = np.square(c_O[1,0]*epsilon/3)

		Q_t = s * np.eye(dim_X)
		
		if Q_t.shape[0] == 3:
			Q_t[0,0] = 100000000
			Q_t[2,2] = 100000000
		
		if Q_t.shape[0] == 5:
			Q_t[0,0] = 100000000
			Q_t[1,1] = 100000000
			Q_t[4,4] = 100000000

		# Compute the terms
		A_Q = np.matmul(A_t_1.T, np.linalg.inv(Q_t))
		if term_1 is None:
			term_1 = np.matmul(A_Q, A_t_1)
			term_2 = np.matmul(A_Q, c_X - last_X)
		else:
			term_1 += np.matmul(A_Q, A_t_1)
			term_2 += np.matmul(A_Q, c_X - last_X)


	theta = np.matmul(np.linalg.pinv(term_1), term_2)

	# Compute the log-likeliood
	ll = np.array([[0.0]])

	for i in range(1, num_obs):
		last_X = X_seq[i-1]
		c_X = X_seq[i]
		c_O = O_seq[i]

		# Compute the matrix A_{t-1}
		A_t_1 = compute_A(last_X)

		# Get the noise matrix Q_t
		if epsilon is None:
			s = 1.0
		else:
			s = np.square(c_O[1,0]*epsilon/3)

		Q_t = s * np.eye(dim_X)
		if Q_t.shape[0] == 3:
			Q_t[0,0] = 100000000
			Q_t[2,2] = 100000000
		
		if Q_t.shape[0] == 5:
			Q_t[0,0] = 100000000
			Q_t[1,1] = 100000000
			Q_t[4,4] = 100000000

		mu_term = c_X - np.matmul(A_t_1, theta) - last_X
		mu_T_Q = np.matmul(mu_term.T, np.linalg.inv(Q_t))

		ll -= np.matmul(mu_T_Q, mu_term)

	return theta, ll

def f_SIR_Kronecker(X, theta, T):
	'''
	This functions computes the next state, given the current state (S,I,R).
	X is a numpy array of length 3 with the initial values of S, I, R
	theta is an array of length 4 with the values [beta, gamma, coef, k]
	'''
	# Get the initial value of the model
	S=X[0]; I=X[1]; R=X[2];
	P = S + I + R
	data = [[S,I,R]]
	
	# Get the parameters
	beta = theta[0]
	gamma = theta[1]
	coef = theta[2]
	k = theta[3]

	for t in range(T):
		# Get the latest point
		S = data[-1][0]
		I = data[-1][1]
		R = data[-1][2]
		
		# Compute the changes
		if t == k:
			imported = coef
		else:
			imported = 0    
		change_S = beta*I*S / P + imported
		change_R = gamma * I

		S_next = S - change_S
		R_next = R + change_R    
		I_next = I + change_S - change_R

		data.append([S_next, I_next, R_next])
	
	data = np.array(data)
	
	return data[1:,:]

class SIR():
	def __init__(self, beta=0, gamma=0):
		self.beta = beta
		self.gamma = gamma

	def beta_gamma_solver(self, S, I, R):
		# Create the dataset for solving for Beta and Gamma
		S_next = np.array(S[1:])
		I_next = np.array(I[1:])
		R_next = np.array(R[1:])
		
		S_t = np.array(S[0:-1])
		I_t = np.array(I[0:-1])
		R_t = np.array(R[0:-1])
		
		P = S[0] + I[0] + R[0]
			
		delta_S = S_next - S_t
		delta_I = I_next - I_t
		delta_R = R_next - R_t
		
		# Create the matrices to solve the linear equations
		a=0; b=0; c=0; d=0; e=0
		
		num_elements = len(delta_S)
		
		for t in range(num_elements):
			a += 2 * (S_t[t]*I_t[t]/P)**2
			b -= S_t[t]*I_t[t]*I_t[t]/P
			c += 2 * I_t[t]*I_t[t]
			d -= S_t[t]*I_t[t]*(delta_S[t] - delta_I[t])/P
			e -= (delta_I[t] - delta_R[t]) * I_t[t]
			
		A_mat = np.array([
			[a, b],
			[b, c]
		])
		
		b_vec = np.array([[d],[e]])
		
		# Solve the system of linear equations
		solution = np.dot( np.linalg.inv(A_mat), b_vec )
		
		beta = solution[0,0]
		gamma = solution[1,0]
		
		return beta, gamma

	def train(self, S, I, R):
		'''
		Description:
			This function receives a sequence of values as an input
			and learns the parameters beta and gamma that minimize the
			error between a predicted and the observed time series

		Input:
			S - A numpy array of dimensions (num_timepoints) containing the
				Susceptible data
			I - Similar to S, but for the infected
			R - Similar to S, but for the removed
		'''
		self.beta, self.gamma = self.beta_gamma_solver(S, I, R)

	def predict_next(self, S, I, R):
		'''
		Description:
			Given values of S, I, and R compute the next state.
		Input:
			S - number of susceptibles
			I - number of infected
			R - number of removed
		'''
		P = S + I + R

		S_next = -self.beta*S*I/P + S
		I_next = self.beta*S*I/P - self.gamma*I + I
		R_next = self.gamma*I + R

		return S_next, I_next, R_next

	def reconstruct_sequence(self, S_list, I_list, R_list):
		num_timepoints = len(S_list)
		predicted_sequence = list()

		for S, I, R, in zip(S_list, I_list, R_list):
			S_next, I_next, R_next = self.predict_next(S,I,R)
			state = np.array([S,I,R])
			predicted_sequence.append(state)

		predicted_sequence = np.array(predicted_sequence)

		return predicted_sequence[0:-1,:]
