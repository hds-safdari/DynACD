"""
	Class definition of Dynamic Anomaly Detection, the algorithm to perform inference on dynamical networks. 
	The latent variables are related to community memberships and anomaly parameters parameters.
	The latent variables at T=0 are drived separately from those at T>0.  
	In this version of the algorithm, all the latent variables are static, exept w and ell, i.e., w(t), ell(t). 
	In this version, the compuations for the initial time step is separated from other time steps. 
	The dataset is divided to data0 and data_b1mAtm1At, which is equivalent to \hat{A}(t) in Dynamic_CRep. 
"""

from __future__ import print_function 
import time
import sys
import sktensor as skt
import numpy as np
import pandas as pd
from termcolor import colored 
import Dyn_ACD.time_glob as gl
from scipy.stats import poisson 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq, root, root_scalar
from scipy.sparse import coo_matrix
import scipy.sparse as sps
from scipy.sparse.linalg import inv
from scipy import sparse

# def timer_func(some_function):
# 	# This function shows the execution time of
# 	# the function object passed
# 	def wrapper(*args, **kwargs):
# 		t1 = time.time()
# 		result = some_function(*args, **kwargs)
# 		t2 = time.time()
# 		print(f'Function {some_function.__name__!r} executed in {(t2-t1):.4f}s')
# 		return result
#
# 	return wrapper

EPS = 1e-12

class Dyn_ACD_wtemp:  
	def __init__(self, N=100, L=1, K=3, undirected=False, initialization=0, ag=1.0,bg=0., rseed=10, inf=1e15, err_max=1e-12, err=0.01, 
				 N_real=1, tolerance=0.001, decision=10, max_iter=800, out_inference=False,
				 in_parameters = '../data/input/synthetic/theta_500_3_5.0_6_0.05_0.2_10',
				 fix_communities=False,fix_w=False,plot_loglik=False,beta0 = 0.1,phi0 = 0.1, ell0= 0.1, pibr0 = 0.1, mupr0= 0.1,
				 fix_pibr=False, fix_mupr=False,fix_phi = False, fix_ell= False,
				 out_folder='../data/output/', end_file='.dat', assortative=True, fix_beta=False,
				 constrained=False, verbose=False, flag_anomaly = False):

		self.N = N  # number of nodes
		self.L = L  # number of layers 
		self.K = K  # number of communities
		self.undirected = undirected  # flag to call the undirected network
		self.rseed = rseed  # random seed for the initialization
		self.inf = inf  # initial value of the pseudo log-likelihood
		self.err_max = err_max  # minimum value for the parameters
		self.err = err  # noise for the initialization
		self.N_real = N_real  # number of iterations with different random initialization
		self.tolerance = tolerance  # tolerance parameter for convergence
		self.decision  = decision  # convergence parameter
		self.max_iter  = max_iter  # maximum number of EM steps before aborting
		self.out_inference = out_inference  # flag for storing the inferred parameters
		self.out_folder = out_folder  # path for storing the output
		self.end_file   = end_file  # output file suffix
		self.assortative = assortative  # if True, the network is assortative
		self.fix_pibr = fix_pibr  # if True, the pibr parameter is fixed
		self.fix_mupr = fix_mupr  # if True, the mupr parameter is fixed
		self.fix_phi  = fix_phi  # if True, the phi parameter is fixed
		self.fix_ell  = fix_ell  # if True, the l parameter is fixed 
		self.fix_beta = fix_beta  # if True, the beta parameter is fixed
		self.fix_communities = fix_communities
		self.fix_w = fix_w
		self.constrained = constrained  # if True, use the configuration with constraints on the updates
		self.verbose     = verbose  # flag to print details

		self.ag = ag # shape of gamma prior
		self.bg = bg # rate of gamma prior 
		self.beta0 = beta0
		self.pibr0 = pibr0  # initial value for the mu in 
		self.mupr0 = mupr0  # initial value for the pi in Bernolie dist
		self.phi0  = phi0  # initial value for the reciprocity coefficient 
		self.ell0  = ell0
		self.plot_loglik   = plot_loglik 
		self.in_parameters = in_parameters
		self.flag_anomaly  = flag_anomaly  

		if initialization not in {0, 1, 2, 3, 4}:  # indicator for choosing how to initialize u, v and w
			raise ValueError(f'The initialization parameter has to be in {[0, 1, 2, 3, 4]}. It is used as an indicator to '
							 'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
							 'will be generated randomly, otherwise they will upload from file.')
		self.initialization = initialization
		self.pibr0 = 0.3
		

		if self.initialization == 0 or self.initialization == 2:
			rand_num = 0.1
			self.beta0 = rand_num 
			if self.flag_anomaly:
				self.phi0 = rand_num 
				self.mupr0 = rand_num 
				self.pibr0 = rand_num 
				self.ell0 = rand_num 
		
		elif self.initialization == 1:
			self.theta = np.load(self.in_parameters + '.npz',allow_pickle=True)  
			self.N, self.K = self.theta['u'].shape   
			theta_key = list(self.theta.files)
			if 'beta' in theta_key:
				self.beta0  = self.theta['beta']   
			rand_num = 0.9
			if self.flag_anomaly:
				self.phi0 = rand_num 
				self.mupr0 = rand_num 
				self.pibr0 = 0.4 
				self.ell0 = rand_num 
		elif self.initialization == 4 or self.initialization == 3:
			# rng = np.random.RandomState(self.rseed)
			self.theta = np.load(self.in_parameters + '.npz',allow_pickle=True)  
			self.N, self.K = self.theta['u'].shape   
			theta_key = list(self.theta.files)
			if 'beta' in theta_key:
				self.beta0  = self.theta['beta']  
			if 'phi' in theta_key:
				self.phi0   = self.theta['phi']
			if 'mu' in theta_key:
				self.mupr0  = self.theta['mu']
			if 'pi' in theta_key:
				self.pibr0  = self.theta['pi']
			if 'ell' in theta_key:
				self.ell0   = self.theta['ell']  
		
		if self.pibr0 is not None:
			if (self.pibr0 < 0) or (self.pibr0 > 1):
				raise ValueError('The reciprocity coefficient pibr0 has to be in [0, 1]!')

		if self.mupr0 is not None:
			if (self.mupr0 < 0) or (self.mupr0 > 1):
				raise ValueError('The reciprocity coefficient mupr0 has to be in [0, 1]!')
		
		self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
		self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership  

		# values of the parameters in the previous iteration
		self.u_old = np.zeros((self.N, self.K), dtype=float)  # out-going membership
		self.v_old = np.zeros((self.N, self.K), dtype=float)  # in-going membership 
 
		
		# values of the affinity tensor: in this case w is always ASSORTATIVE 
		if self.L > 0:
			if self.assortative:  # purely diagonal matrix
				self.w = np.zeros((self.L+1, self.K), dtype=float) 
				self.w_old = np.zeros((self.L+1, self.K), dtype=float)
				self.w_f = np.zeros((self.L+1, self.K), dtype=float)
			else:
				self.w = np.zeros((self.L+1, self.K, self.K), dtype=float)
				self.w_old = np.zeros((self.L+1, self.K, self.K), dtype=float)
				self.w_f = np.zeros((self.L+1, self.K, self.K), dtype=float) 
		else:
			if self.assortative:  # purely diagonal matrix
				self.w = np.zeros((1, self.K), dtype=float) 
				self.w_old = np.zeros((1, self.K), dtype=float)
				self.w_f = np.zeros((1, self.K), dtype=float)
			else:
				self.w = np.zeros((1, self.K, self.K), dtype=float)
				self.w_old = np.zeros((1, self.K, self.K), dtype=float)
				self.w_f = np.zeros((1, self.K, self.K), dtype=float)  
		

		if self.ag < 1 :
			self.ag = 1.
		if self.bg < 0:
			self.bg = 0. 
		
		
		if self.fix_beta:
			if self.beta0 is not None:
				self.beta = self.beta_old = self.beta_f = self.beta0
			else:
				self.beta = self.beta_old = self.beta_f = 1.
		

		if self.flag_anomaly == False: 
			self.phi =  self.phi_old = self.phi_f =self.phi0 =  1.
			self.ell =  self.ell_old = self.ell_f = self.ell0= 0.
			self.pibr =  self.pibr_old = self.pibr_f = self.pibr0= 0.
			self.mupr = self.mupr_old = self.mupr_f = self.mupr0 = 0.
			self.fix_phi = self.fix_ell = self.fix_pibr = self.fix_mupr = True 

		if self.fix_phi and self.flag_anomaly == True and self.phi0 is not None:
			self.phi = self.phi_old = self.phi_f = self.phi0
		
		if self.fix_ell and self.flag_anomaly == True and self.ell0 is not None: 
			self.ell = self.ell_old = self.ell_f = float(self.ell0)
		

		if self.fix_pibr and self.flag_anomaly == True and self.pibr0 is not None:
			self.pibr = self.pibr_old = self.pibr_f = self.pibr0
		
		if self.fix_mupr and self.flag_anomaly == True and self.mupr0 is not None:
			self.mupr = self.mupr_old = self.mupr_f = self.mupr0 

	# @gl.timeit('fit')
	def fit(self, data, T, nodes):
		"""
			Model directed networks by using a probabilistic generative model that assume community parameters and
			reciprocity coefficient. The inference is performed via EM algorithm.
			Parameters
			----------
			data : ndarray/sptensor
				   Graph adjacency tensor.
			data_T: None/sptensor
					Graph adjacency tensor (transpose).
			data_T_vals : None/ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			nodes : list
					List of nodes IDs.
			Returns
			-------
			u_f : ndarray
				  Out-going membership matrix.
			v_f : ndarray
				  In-coming membership matrix.
			w_f : ndarray
				  Affinity tensor.
			eta_f : float
					Reciprocity coefficient.
			maxL : float
				   Maximum pseudo log-likelihood.
			final_it : int
					   Total number of iterations.
		""" 
		self.N = data.shape[-1]  
		T = max(0, min(T, data.shape[0]-1)) 
		print('T:', T)
		self.T = T 
		self.L = T   
		data = data[:T+1,:,:]  
		'''
		Pre-process data
		'''
		original_shape = data.shape 
		new_shape = (original_shape[0] - 1, original_shape[1], original_shape[2])  
		data_b1mAtm1At = np.zeros(new_shape) 	# Aij(t)*(1-Aij(t-1))   
		# data0 = np.zeros(data.shape[0])
		data_bAtm11mAt = np.zeros(new_shape)	# A(t-1) (1-A(t))
		data_bAtm1At = np.zeros(new_shape)	# A(t-1) A(t) 

		data0 = data[0:1, :, :]     
		data0_T = np.einsum('aij->aji', data0) # to calculate Pois terms by Aji(t) s in Q_{ij}   

		if self.T > 0:
			for i in range(self.T):
				data_b1mAtm1At[i,:,:] = data[i+1,:,:] * (1 - data[i,:,:]) # (1-A(t-1)) A(t) 
				data_bAtm11mAt[i,:,:] = (1 - data[i+1,:,:]) * data[i,:,:] # A(t-1) (1-A(t))
				data_bAtm1At[i,:,:]   = data[i+1,:,:] * data[i,:,:]       # A(t-1) A(t)  
		
		self.b1mAtm1At = data_b1mAtm1At.sum()
		self.bAtm11mAt = data_bAtm11mAt.sum()
		self.bAtm1At   = data_bAtm1At.sum() # needed in the update of Q	 
			

		'''
		transposes needed in Q. 
		'''
		data_b1mAtm1AtT = np.einsum('aij->aji', data_b1mAtm1At) # needed in the update of Q 	 	 
		data_bAtm11mAtT = np.einsum('aij->aji', data_bAtm11mAt) # needed in the update of Q	 
		data_bAtm1AtT   = np.einsum('aij->aji', data_bAtm1At) # needed in the update of Q  

		self.b1mAtm1At_T = data_b1mAtm1AtT.sum()
		self.bAtm11mAt_T = data_bAtm11mAtT.sum()
		self.bAtm1At_T   = data_bAtm1AtT.sum()


		sum_b1mAtm1At  =  data_b1mAtm1At.sum(axis=0)	
		sum_bAtm11mAt  =  data_bAtm11mAt.sum(axis=0)
		sum_bAtm1At    =  data_bAtm1At.sum(axis=0)

		sum_b1mAtm1At_T  = data_b1mAtm1AtT.sum(axis=0)
		sum_bAtm11mAt_T  = data_bAtm11mAtT.sum(axis=0)
		sum_bAtm1At_T    = data_bAtm1AtT.sum(axis=0)

		'''
		values of A(t) (1-A(t-1)), A(t-1) (1-A(t)), and A(t-1) A(t); and their transposes  nedded in Q.  
		''' 

		self.data0vals0    = get_item_array_from_subs(data0, data0.nonzero())      
		self.data0_T_vals0 = get_item_array_from_subs(data0_T, data0.nonzero())  
		'''
		sptensor of A(t) (1-A(t-1)), A(t-1) (1-A(t)), and A(t-1) A(t) nedded in Q. 
		'''
		data0 = preprocess(data0)   
		data0_T = preprocess(data0_T)    
		data_b1mAtm1At = preprocess(data_b1mAtm1At)  # to calculate numerator containing Aij(t)*(1-Aij(t-1))  
		data_bAtm11mAt = preprocess(data_bAtm11mAt) 
		data_bAtm1At   = preprocess(data_bAtm1At)  

		data_b1mAtm1AtT = preprocess(data_b1mAtm1AtT)  # to calculate numerator containing Aij(t)*(1-Aij(t-1))   
		data_bAtm11mAtT = preprocess(data_bAtm11mAtT) 
		data_bAtm1AtT   = preprocess(data_bAtm1AtT)    

		# save the indexes of the nonzero entries of Aij(t)*(1-Aij(t-1)) 
		if isinstance(data_b1mAtm1At, skt.dtensor):
			subs_nzp = data_b1mAtm1At.nonzero()
			print('subs_nzp',len(subs_nzp), subs_nzp[0].shape)
		elif isinstance(data_b1mAtm1At, skt.sptensor):
			subs_nzp = data_b1mAtm1At.subs   
 

		# save the indexes of the nonzero entries of  Aij(0)
		if isinstance(data0, skt.dtensor): 
			subs_nz = data0.nonzero()
			print(len(subs_nz),subs_nz[0].shape)
		elif isinstance(data0, skt.sptensor): 
			subs_nz = data0.subs   

		
		'''
		INFERENCE
		'''
		maxL = -self.inf  # initialization of the maximum log-likelihood
		rng = np.random.RandomState(self.rseed)

		for r in range(self.N_real):   
			self._initialize(rng=rng)  
			
			# if T > 0:   
			# 	self.beta_hat = np.ones(T) * self.beta
			# 	self.beta_hat[0] = 1

			self._update_old_variables() 

			# convergence local variables
			coincide, it = 0, 0
			convergence = False

			if self.verbose:
				print(f'Updating realization {r} ...', end=' \n')

			maxL =  - self.inf
			loglik_values = []
			time_start = time.time()
			loglik = -self.inf

			while not convergence and it < self.max_iter:   

				delta_u, delta_v, delta_w, delta_beta, delta_phi, delta_ell, delta_pibr, delta_mupr = self._update_em(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,
																				sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
				it, loglik, coincide, convergence = self._check_for_convergence(data_b1mAtm1At,data0,subs_nzp, subs_nz, T, r, it, loglik, coincide, convergence,
																				data_T=data_b1mAtm1AtT)  

				loglik_values.append(loglik)

			if self.verbose:
				# print('done!')
				print(f'Nreal = {r} - Loglikelihood = {loglik} - iterations = {it} - '
					f'time = {np.round(time.time() - time_start, 2)} seconds')
 
			if maxL < loglik:
				self._update_optimal_parameters()
				self.maxL = loglik
				self.final_it = it
				conv = convergence
				best_loglik_values = list(loglik_values)

			self.rseed += 1

			# end cycle over realizations

		if self.plot_loglik:
			plot_L(best_loglik_values, int_ticks=True)

		if self.out_inference:
			self.output_results(nodes)
		 
		return self.u_f, self.v_f, self.w_f, self.beta_f, self.phi_f, self.ell_f, self.pibr_f, self.mupr_f, self.maxL 


	def _initialize(self, rng=None):
		"""
			Random initialization of the parameters u, v, w, beta, ell, pibr, phi, mu.
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed) 
		if (self.pibr0 is not None) and (not self.fix_pibr):
			self.pibr = self.pibr0
		else: 
			if self.fix_pibr == False: 
				if self.verbose:
					print('pi is initialized randomly.')
				self._randomize_pibr(rng) 
		

		
		if (self.mupr0 is not None) and (not self.fix_mupr):
			self.mupr = self.mupr0
		else: 
			if self.fix_mupr == False:   
				if self.verbose:
					print('mu is initialized randomly.')
				self._randomize_mupr(rng) 
		

		if self.T > 0:
			if self.beta0 is not None:
				self.beta = self.beta0
			else:
				self._randomize_beta(rng) 
			
			if self.flag_anomaly:
				if self.phi0 is not None:
					self.phi = self.phi0
				else:
					self._randomize_phi(rng)
			else:
				self.phi = 1.
			
			if self.flag_anomaly:
				if self.ell0 is not None:
					self.ell = self.ell0 
				else:
					self._randomize_ell(rng)
			else:
				self.ell = np.zeros(self.T) 
		else:
			self.ell = 0. 
			self.phi = 1.
			self.beta = 1. 

		

		if self.initialization == 0:
			if self.verbose:
				print('u, v and w are initialized randomly.') 
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True) 
			self._randomize_w(rng=rng)
			self._randomize_u_v(rng=rng)  

		elif self.initialization == 1:
			if self.verbose:
				print('w is initialized randomly; u, and v are initialized using the input files:')
				print(self.in_parameters + '.npz')
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True)
			self._initialize_u(theta['u'])
			self._initialize_v(theta['v']) 
			self._randomize_w(rng=rng)

		elif self.initialization == 2:
			if self.verbose:
				print('u, v and w are initialized using the input files:')
				print(self.in_parameters + '.npz')
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True) 
			self._initialize_u(theta['u'])
			self._initialize_v(theta['v'])
			self._initialize_w(theta['w'])  

		elif self.initialization == 3:
			if self.verbose:
				print('u, and v are initialized randomly; w is initialized using the input files:')
				print(self.in_parameters + '.npz')
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True)
			self._randomize_u_v(rng=rng) 
			self._initialize_w(theta['w'])  
		elif self.initialization == 4:
			if self.verbose:
				print('u, v and w are initialized randomly.') 
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True) 
			self._initialize_u(theta['u'])
			self._initialize_v(theta['v'])
			self._initialize_w_RD(theta['w']) 

	def _initialize_u(self, u): 
		if u.shape[0] != self.N:
			raise ValueError('u.shape is different that the initialized one.',self.N,u.shape[0]) 

		self.u = u.copy()  
		max_entry = np.max(u)
		self.u += max_entry * self.err  * np.random.random_sample(self.u.shape)  

	def _initialize_v(self, v):
		if v.shape[0] != self.N:
			raise ValueError('v.shape is different that the initialized one.',self.N,v.shape[0]) 
		self.v = v.copy() 
		max_entry = np.max(v)
		self.v += max_entry * self.err * np.random.random_sample(self.v.shape)
	
	def _initialize_w(self, w):
		"""
			Initialize affinity tensor w from file.

			Parameters
			----------
			infile_name : str
						  Path of the input file.
		"""    
		# w_0 = np.diagonal(w) 
		if self.T > 0:  
			self.w[:] = np.diagonal(w) 
			max_entry = np.max(self.w)
			self.w += max_entry * self.err * np.random.random_sample(self.w.shape)
		else:
			self.w[0] = np.diagonal(w)
			max_entry = np.max(w)
			self.w += max_entry * self.err * np.random.random_sample(self.w.shape)  
	
	def _initialize_w_RD(self, w):
		"""
			Initialize affinity tensor w from file.

			Parameters
			----------
			infile_name : str
						  Path of the input file.
		"""    
		# w_0 = np.diagonal(w) 
		if self.T > 0:  
			self.w[:] = w[0]
			max_entry = np.max(self.w)
			self.w += max_entry * self.err * np.random.random_sample(self.w.shape)
		else:
			self.w[0] = np.diagonal(w)
			max_entry = np.max(w)
			self.w += max_entry * self.err * np.random.random_sample(self.w.shape)  
		

	
	def _randomize_beta(self, rng=None):
		"""
			Generate a random number in (0, 1.).
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		# self.beta = rng.random_sample(1) 
		self.beta  = rng.rand() 

	def _randomize_phi(self, rng=None):
		"""
			Generate a random number in (0, 1.).
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		# self.phi = rng.random_sample(1) 
		self.phi  = rng.rand() 
	
	def _randomize_ell(self, rng=None):
		"""
			Generate a random number in (0, 1.).
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		# self.ell = rng.random_sample(1) 
		# self.ell  = rng.rand(1,self.T)
		# self.ell  = [rng.uniform(0, 1) for _ in range(self.T)]
		self.ell = rng.rand()
	
	def _randomize_pibr(self, rng=None):
		"""
			Generate a random number in (0, 1.).

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)  
		# self.pibr  = rng.random_sample(1)[0] 
		self.pibr  = rng.rand()  
	
	def _randomize_mupr(self, rng=None):
		"""
			Generate a random number in (0, 1.).

			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		# self.mupr = rng.random_sample(1)[0]
		self.mupr  = rng.rand() 

	def _randomize_w(self, rng):
		"""
			Assign a random number in (0, 1.) to each entry of the affinity tensor w.
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator. 
		"""  
		if rng is None:
			rng = np.random.RandomState(self.rseed)
		L = self.w.shape[0]
		for i in range(L):
			if self.assortative:
				random_samples = 1 * np.random.random_sample(self.K)
				self.w[i, :self.K] = random_samples
			else: 
				for k in range(self.K):  
					for q in range(k, self.K):
						if q == k: 
							self.w[i, k, q] = rng.random_sample(1)[0]
						else:
							self.w[i, k, q] = self.w[i, q, k] = self.err * 1 * rng.random_sample(1)[0]   
		

	def _randomize_u_v(self, rng=None):
		"""
			Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""
		# if self.T > 0:
		# 	if rng is None:
		# 		rng = np.random.RandomState(self.rseed)
		# 	self.u = rng.random_sample((self.N,self.K)) * 0.01
		# 	row_sums = self.u.sum(axis=1) 
		# 	self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

		# 	if not self.undirected:
		# 		self.v = rng.random_sample((self.N,self.K)) * 0.01
		# 		row_sums = self.v.sum(axis=1)
		# 		self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		# 	else:
		# 		self.v = self.u
		

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		self.u = rng.random_sample((self.N,self.K))
		row_sums = self.u.sum(axis=1) 
		self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

		if not self.undirected:
			self.v = rng.random_sample((self.N,self.K))
			row_sums = self.v.sum(axis=1)
			self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		else:
			self.v = self.u 

	def _update_old_variables(self):
		"""
			Update values of the parameters in the previous iteration.
		""" 
		self.u_old = np.copy(self.u)
		self.v_old = np.copy(self.v)
		self.w_old = np.copy(self.w) 
		if self.flag_anomaly:
			self.pibr_old = np.copy(self.pibr)
			self.mupr_old = np.copy(self.mupr)
		else:
			self.pibr_old = np.copy(self.pibr0)
			self.mupr_old = np.copy(self.mupr0)
		
		if self.T>0: 
			self.phi_old = np.copy(self.phi)    
			self.ell_old = np.copy(self.ell)  
			self.beta_old = np.copy(self.beta)  

	# @gl.timeit('cache')

	# @timer_func
	def _update_cache(self,data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T,update_Q=True):
		"""
			Update the cache used in the em_update.
			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
		"""
		if self.T > 0:
			self.lambda0_nz = self._lambda0_nz(subs_nzp, self.u , self.v , self.w[1:])  
		self.lambda0_nz_0 = self._lambda0_nz(subs_nz, self.u , self.v , (self.w[0])[np.newaxis])  

		if self.assortative == False:
			if self.T > 0:
				self.lambda0_nzT = self._lambda0_nz(subs_nzp, self.v, self.u, np.einsum('akq->aqk',self.w[1:]))
			self.lambda0_nzT_0 = self._lambda0_nz(subs_nz, self.v, self.u, np.einsum('akq->aqk', (self.w[0])[np.newaxis,:]))  
		else:
			if self.T > 0:
				self.lambda0_nzT = self._lambda0_nz(subs_nzp, self.v, self.u,self.w[1:]) 
			self.lambda0_nzT_0 = self._lambda0_nz(subs_nz, self.v, self.u, (self.w[0])[np.newaxis])
		 
		if self.T > 0:
			self.M_nz = self.lambda0_nz   
			self.M_nz[self.M_nz == 0] = 1 

		self.M_nz_0 = self.lambda0_nz_0   
		self.M_nz_0[self.M_nz_0 == 0] = 1 

		if self.flag_anomaly == True:
			if update_Q == True:
				self.Qij_dense,self.Qij_nz = self._QIJ(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)    
		if self.T > 0:
			if isinstance(data_b1mAtm1At, skt.dtensor):
				if self.flag_anomaly == True:
					self.data_M_nz_Q  = data_b1mAtm1At[subs_nzp] * (1-self.Qij_dense)[subs_nzp]/ self.M_nz
				else:
					self.data_M_nz_Q = data_b1mAtm1At[subs_nzp] / self.M_nz
			
			elif isinstance(data_b1mAtm1At, skt.sptensor):
				if self.flag_anomaly == True:
					self.data_M_nz_Q = data_b1mAtm1At.vals * (1-self.Qij_nz) / self.M_nz   
				else:
					self.data_M_nz_Q = data_b1mAtm1At.vals / self.M_nz

		
		if isinstance(data0, skt.dtensor):
			if self.flag_anomaly == True:
				self.data_M_nz_Q_0  = data0[subs_nz] * (1-self.Q_ij_nz_0)[subs_nz]/ self.M_nz_0   
			else:
				self.data_M_nz_Q_0 = data0[subs_nz] / self.M_nz_0 
		
		elif isinstance(data0, skt.sptensor):
			if self.flag_anomaly == True:
				self.data_M_nz_Q_0 = data0.vals  * (1-self.Q_ij_nz_0) / self.M_nz_0
			else:
				self.data_M_nz_Q_0 = data0.vals / self.M_nz_0

	# @timer_func
	def _QIJ(self, data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T,EPS=1e-12):
		"""
			Compute the mean Q for  non-zero entries, and all the entries.

			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.

			Returns
			-------
			Q_ij_dense_tot : ndarray
						Q_ij_dense_tot[subs_nzp] for only non-zero entries, and  Q_ij_dense_tot for all entries, 
		"""      
		if self.T >0:
			lambda0_ija    = self._lambda0_full(self.u, self.v, self.w[1:])
		lambda0_ija_0  = self._lambda0_full(self.u, self.v,  (self.w[0])[np.newaxis])
		# lambda0_ijaT_0 = self._lambda0_full(self.v0, self.u0, self.w0)
		# lambda0_ijaT   = self._lambda0_full(self.v, self.u, self.w)
		if self.T >0:
			lambda0_ijaT = np.einsum('aij->aji', lambda0_ija)
		lambda0_ijaT_0 = np.einsum('aij->aji',lambda0_ija_0)
		# assert np.allclose(np.einsum('aij->aji',lambda0_ija),lambda0_ijaT)
		"""
			Compute Q_ij_dense at zeros of (1-Aij(t-1)) * Aij(t) by dense Aij(t-1) * (1-Aij(t)) and Aij(t-1) * Aij(t) 
		"""
		if not hasattr(self, 'dense_data0'):
			self.dense_data0 = data0.toarray()
			self.dense_data0T = data0_T.toarray()

		if self.T > 0:
			# A(t) (1-A(t-1)), A(t-1) (1-A(t)), and A(t-1) A(t) and their tranposes to  dense array
			if not hasattr(self,'dense_data_b1mAtm1At'):
				self.dense_data_b1mAtm1At = data_b1mAtm1At.toarray() 
				self.dense_data_bAtm11mAt = data_bAtm11mAt.toarray()
				self.dense_data_bAtm1At   = data_bAtm1At.toarray()
				self.dense_data_b1mAtm1AtT = data_b1mAtm1AtT.toarray()
				self.dense_data_bAtm11mAtT = data_bAtm11mAtT.toarray()
				self.dense_data_bAtm1AtT   = data_bAtm1AtT.toarray()

		Q_ij_dense_tot   = np.zeros(self.dense_data0.shape)
		Q_ij_dense_d_tot = np.zeros(self.dense_data0.shape)
		
		if self.T == 0:
			flag_dense = True
			if flag_dense:
				"""
					Compute Q_ij_dense for t=0, dense matrix 
				# """
				Q_ij_dense_0   = (self.mupr * poisson.pmf(self.dense_data0, self.pibr) * poisson.pmf(self.dense_data0T, self.pibr))#* np.exp(-self.pibr*2)
				Q_ij_dense_0_d = (1-self.mupr) * poisson.pmf(self.dense_data0, lambda0_ija_0) * poisson.pmf(self.dense_data0T, lambda0_ijaT_0)  #* np.exp(-(lambda0_ija_0+lambda0_ijaT_0))
				Q_ij_dense_0_d += Q_ij_dense_0
				non_zeros = Q_ij_dense_0_d > 0
				Q_ij_dense_0[non_zeros] /=  Q_ij_dense_0_d[non_zeros]  
			else:
				"""
					Compute Q_ij_dense for t=0, sptensor format
				# """      
				nz_recon_I_0 =  self.mupr * poisson.pmf(self.data0vals0, self.pibr) * poisson.pmf(self.data0_T_vals0, self.pibr)
				nz_recon_Id_0 = nz_recon_I_0 + (1-self.mupr) * poisson.pmf(self.data0vals0, self.lambda0_nz_0) * poisson.pmf(self.data0_T_vals0, self.lambda0_nzT_0) 

				non_zeros = nz_recon_Id_0 > 0
				nz_recon_I_0[non_zeros] /=  nz_recon_Id_0[non_zeros]

				Q_ij_dense_0 = np.ones(lambda0_ija_0.shape)
				Q_ij_dense_0 *=  self.mupr * np.exp(-self.pibr*2)
				Q_ij_dense_d_0 = Q_ij_dense_0 + (1-self.mupr) * np.exp(-lambda0_ija_0) * np.exp(-lambda0_ijaT_0)
				non_zeros = Q_ij_dense_d_0 > 0
				Q_ij_dense_0[non_zeros] /= Q_ij_dense_d_0[non_zeros]

				Q_ij_dense_0[subs_nz] = np.copy(nz_recon_I_0) 


			self.Q_ij_dense_0 = np.maximum(Q_ij_dense_0, transpose_tensor(Q_ij_dense_0)) # make it symmetric
			assert np.allclose(self.Q_ij_dense_0[0], self.Q_ij_dense_0[0].T, rtol=1e-05, atol=1e-08)
			np.fill_diagonal(self.Q_ij_dense_0[0], 0.)
			assert (self.Q_ij_dense_0[0] > 1).sum() == 0
			self.Q_ij_nz_0 = self.Q_ij_dense_0[subs_nz]

			# self.Q0A0  = (Q_ij_dense_0[subs_nz] * dense_data0[subs_nz]).sum()	  # needed in the update of Likelihood
			self.Qsum = (self.Q_ij_dense_0).sum()  # needed in the update of Likelihood
			self.b1mQsum = (1-self.Q_ij_dense_0).sum()   # needed in the update of Likelihood
			# self.Q_ij_dense_0 = np.copy(Q_ij_dense_0)
			# self.Q_ij_nz_0 = np.copy(Q_ij_dense_0[subs_nz])
			self.Q0A0  = (data0.vals * self.Q_ij_nz_0).sum() 



		elif self.T > 0:

			"""
				Compute Q_ij_dense for t>0, full dataset, poisson
			"""
			# s = time.time()
			Q_ij_dense_tot_0   = (self.mupr * poisson.pmf(self.dense_data0, self.pibr) * poisson.pmf(self.dense_data0T, self.pibr))[0]#* np.exp(-self.pibr*2)
			Q_ij_dense_d_tot_0 = ((1-self.mupr) * poisson.pmf(self.dense_data0, lambda0_ija_0) * poisson.pmf(self.dense_data0T, lambda0_ijaT_0) )[0]  #* np.exp(-(lambda0_ija_0+lambda0_ijaT_0)) 

			# s = time.time()
			sum_a = (sum_b1mAtm1At + sum_bAtm11mAt + sum_b1mAtm1At_T + sum_bAtm11mAt_T)
			sum_b = (sum_bAtm1At + sum_bAtm1At_T)
			if self.phi > 0:
				# print(sum_b1mAtm1At.shape)
				log_Q_ij_dense  = (sum_a) * np.log(self.phi+EPS)
			else:
				raise ValueError('Invalid value', self.phi)
			
			if (1-self.phi) > 0:
				log_Q_ij_dense  += (sum_b) * np.log(1-self.phi+EPS)
			else:
				raise ValueError('Invalid value', self.phi)
			
			if self.ell > 0:
				log_Q_ij_dense  += (sum_b1mAtm1At + sum_b1mAtm1At_T) * np.log(self.ell+EPS)
			else:
				raise ValueError('Invalid value', self.ell)
			# print('log_Q_ij_dense.shape', log_Q_ij_dense.shape)

			Q_ij_dense_tot[0] =  np.exp(log_Q_ij_dense) * np.exp(-2. * self.T * self.ell * self.phi)  * Q_ij_dense_tot_0 
			# e = time.time()
			# print(f"Q_ij_dense_tot[0] created in {e - s} sec.")

			# s = time.time()
			if self.beta > 0:
				log_Q_ij_dense_d = (sum_a) * np.log(self.beta + EPS)
			else:
				raise ValueError('Invalid value', self.beta)

			if 1 - self.beta > 0:
				log_Q_ij_dense_d += (sum_b) * np.log(1 - self.beta + EPS)
			else:
				raise ValueError('Invalid value', self.beta)
			
			tmp = np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1At, np.log(lambda0_ija + EPS))
			log_Q_ij_dense_d += (tmp + tmp.T)
			# assert np.allclose(tmp.T,np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1AtT, np.log(lambda0_ijaT + EPS)))
			# log_Q_ij_dense_d += np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1At, np.log(lambda0_ija + EPS))
			# log_Q_ij_dense_d += np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1AtT, np.log(lambda0_ijaT + EPS))
			log_Q_ij_dense_d -= self.beta * (lambda0_ija.sum(axis=0) + lambda0_ijaT.sum(axis=0))
			Q_ij_dense_d_tot[0] = Q_ij_dense_d_tot_0 * np.exp(log_Q_ij_dense_d)

			Q_ij_dense_d_tot += Q_ij_dense_tot 
			non_zeros = Q_ij_dense_d_tot > 0  
			Q_ij_dense_tot[non_zeros] /= Q_ij_dense_d_tot[non_zeros]     

			Q_ij_dense_tot = np.maximum( Q_ij_dense_tot, transpose_tensor(Q_ij_dense_tot)) # make it symmetric
			np.fill_diagonal(Q_ij_dense_tot[0], 0.)

			# e = time.time()
			# print(f"Q_ij_dense_tot created in {e - s} sec.")
			assert (Q_ij_dense_tot[0] > 1).sum() == 0
			assert np.allclose(Q_ij_dense_tot[0], Q_ij_dense_tot[0].T, rtol=1e-05, atol=1e-08)


			# Q_ij_dense_0 = np.zeros(dense_data0.shape)
			# Q_ij_dense_0 = np.copy(Q_ij_dense_tot)

			s = time.time() 
			self.QAtm1At     = np.einsum('ij, aij-> ', Q_ij_dense_tot[0], self.dense_data_bAtm1At)   # needed in the update of Likelihood
			self.QAtm11mAt   = np.einsum('ij, aij-> ', Q_ij_dense_tot[0], self.dense_data_bAtm11mAt) # needed in the update of Likelihood
			self.Q1mAtm1At   = np.einsum('ij, aij-> ', Q_ij_dense_tot[0], self.dense_data_b1mAtm1At) # needed in the update of self.ell
			# self.Qsum       = np.einsum('ij -> ', Q_ij_dense_tot[0]) # needed in the update of self.ell
			
			self.b1mQAtm1At   = np.einsum('ij, aij-> ', (1-Q_ij_dense_tot[0]), self.dense_data_bAtm1At)   # needed in the update of beta, Likelihood
			self.b1mQAtm11mAt = np.einsum('ij, aij-> ', (1-Q_ij_dense_tot[0]), self.dense_data_bAtm11mAt) # needed in the update of beta
			self.b1mQ1mAtm1At = np.einsum('ij, aij-> ', (1-Q_ij_dense_tot[0]), self.dense_data_b1mAtm1At) # needed in the update of beta 

			self.Qsum = (Q_ij_dense_tot).sum()  # needed in the update of Likelihood
			self.b1mQsum = (1-Q_ij_dense_tot).sum()   # needed in the update of Likelihood
			# assert (Q_ij_dense_0 > 1).sum() == 0
			# self.Q_ij_dense_0 = np.copy(Q_ij_dense_tot)
			self.Q_ij_dense_0 = Q_ij_dense_tot
			self.Q_ij_nz_0 = Q_ij_dense_tot[subs_nz]
			self.Q0A0  = (data0.vals * Q_ij_dense_tot[subs_nz]).sum()

			# e = time.time()
			# print(f"Q_ij_dense_tot_npz self in {e - s} sec.")

			# s = time.time()

			Q_ij_dense_tot_npz = np.tile(Q_ij_dense_tot, (self.T, 1, 1)) 
			# e = time.time()
			# print(f"Q_ij_dense_tot_npz in {e - s} sec.")
			

		if self.T > 0:  
			return Q_ij_dense_tot_npz , Q_ij_dense_tot_npz[subs_nzp]
		else:
			return Q_ij_dense_0, Q_ij_dense_0[subs_nz]


	# @gl.timeit('lambda0_nz')
	# @timer_func
	def _lambda0_nz(self, subs_nz, u, v, w): 
		"""
			Compute the mean lambda0_ij for only non-zero entries.
			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.
			Returns
			-------
			nz_recon_I : ndarray
						 Mean lambda0_ij for only non-zero entries.
		"""   


		if not self.assortative:
			nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
		else:
			nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
		nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

		return nz_recon_I
	
	# def _lambda0_nz_0(self, subs_nz, u, v, w): 
	# 	"""
	# 		Compute the mean lambda0_ij for only non-zero entries.
	# 		Parameters
	# 		----------
	# 		subs_nz : tuple
	# 				  Indices of elements of data that are non-zero.
	# 		u : ndarray
	# 			Out-going membership matrix.
	# 		v : ndarray
	# 			In-coming membership matrix.
	# 		w : ndarray
	# 			Affinity tensor.
	# 		Returns
	# 		-------
	# 		nz_recon_I : ndarray
	# 					 Mean lambda0_ij for only non-zero entries.
	# 	"""      

	# 	if not self.assortative:
	# 		nz_recon_IQ = np.einsum('Ik,Ikq->Iq', u[subs_nz[1], :], w[subs_nz[0], :, :])
	# 	else:
	# 		nz_recon_IQ = np.einsum('Ik,Ik->Ik', u[subs_nz[1], :], w[subs_nz[0], :])
	# 	nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :])

	# 	return nz_recon_I
	
	# @gl.timeit('update_em')
	# @timer_func
	def _update_em(self,data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T):
		# data_t,data_b1mAtm1At_t, data_T_vals_t,subs_nz,subs_nzp
		"""
			Update parameters via EM procedure.
			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			denominator : float
						  Denominator used in the update of the eta parameter.
			Returns
			-------
			d_u : float
				  Maximum distance between the old and the new membership matrix u.
			d_v : float
				  Maximum distance between the old and the new membership matrix v.
			d_w : float
				  Maximum distance between the old and the new affinity tensor w. 
		""" 

		self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)

		# gl.timer.cum('uvw')
		if self.fix_communities == False:
			d_u = self._update_U(subs_nzp,subs_nz)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T) 
			if self.undirected:
				self.v = self.u
				self.v_old = self.v
				d_v = d_u
			else:
				d_v = self._update_V(subs_nzp,subs_nz)
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
		else:
			d_u = 0
			d_v = 0

		if self.fix_w == False:
			if not self.assortative:
				d_w = self._update_W(subs_nzp)
			else:
				d_w = self._update_W_assortative(subs_nzp,subs_nz)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
		else:
			d_w = 0

		if self.fix_pibr == False:
			d_pibr = self._update_pibr( data0, subs_nz)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)

		else: 
			d_pibr = 0.
		
		if self.fix_mupr == False:
			# s = time.time()
			d_mupr = self._update_mupr(data0, subs_nz)
			# e = time.time()
			# print('mu',e-s)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
		else:
			d_mupr = 0. 

		if self.fix_beta == False:
			if self.T  > 0:  
				d_beta = self._update_beta() 
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
			else:  d_beta = 0. 
		else:  
			d_beta = 0. 
		
		if self.fix_phi == False: 
			if self.T > 0:
				d_phi = self._update_phi()
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
			else:
				d_phi = 0.
		else:
			d_phi = 0. 
		

		if self.fix_ell == False:
			if self.T > 0:
				# denominator = (data_T_vals * self.beta_hat[subs_nzp[0]]).sum()  
				d_ell = self._update_ell(data_b1mAtm1At, subs_nzp)
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,sum_b1mAtm1At, sum_bAtm11mAt, sum_bAtm1At,sum_b1mAtm1At_T,sum_bAtm11mAt_T, sum_bAtm1At_T)
			else:
				d_ell = 0.
		else:
			d_ell = 0.


		return d_u, d_v, d_w, d_beta, d_phi, d_ell, d_pibr, d_mupr


	# @gl.timeit('pibr')

	# @gl.timeit('pibr')
	def _update_pibr(self,  data, subs_nz):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor. 

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		"""  
		if isinstance(data, skt.dtensor):
			Adata = (data[subs_nz] * self.Q_ij_nz_0).sum()
		elif isinstance(data, skt.sptensor):   
			Adata = np.copy(self.Q0A0) 
		
		self.pibr = Adata / self.Qsum

		if (self.pibr > 1):  
			self.pibr = 0.9999999
			print('self.pibr:', self.pibr)
			dist_pibr = self.pibr - self.pibr_old
			self.pibr_old = np.copy(self.pibr)
		elif (self.pibr < 1e-12):
			self.pibr = 0.00000001
			dist_pibr = abs(self.pibr - self.pibr_old) 
			self.pibr_old = np.copy(self.pibr)  
		else:
			dist_pibr = self.pibr - self.pibr_old
			self.pibr_old = np.copy(self.pibr)
		
		return dist_pibr

	# @gl.timeit('mupr')
	def _update_mupr(self, data, subs_nz):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j). 

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		"""
		self.mupr = (self.Qsum / (self.N * (self.N-1)))
		
		if (self.mupr > 1): 
			print('self.mupr:', self.mupr)
			self.mupr = 0.9999999
			dist_mupr = self.mupr - self.mupr_old
			self.mupr_old = np.copy(self.mupr) 
			# dist_mupr = 0. 
		elif (self.mupr < 1e-12):
			self.mupr = 0.00000001
			dist_mupr = abs(self.mupr - self.mupr_old) 
			self.mupr_old = np.copy(self.mupr)  
		else:
			dist_mupr = self.mupr - self.mupr_old
			self.mupr_old = np.copy(self.mupr) 
	 
		  

		return dist_mupr 

	# @gl.timeit('ell')
	def _update_ell(self, data, subs_nz):
		"""
			Update reciprocity coefficient eta.

			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T_vals : ndarray
						  Array with values of entries A[j, i] given non-zero entry (i, j).
			denominator : float
						  Denominator used in the update of the eta parameter.

			Returns
			-------
			dist_eta : float
					   Maximum distance between the old and the new reciprocity coefficient eta.
		# """    
		self.ell = self.Q1mAtm1At / (self.T*self.phi*self.Qsum)   
		# self.ell_hat[:] = np.copy(self.ell) 
		   
		
		if self.ell > 1:  
			print('self.ell:', self.ell) 
			self.ell = 0.9999999
			dist_ell = self.ell - self.ell_old
			self.ell_old = np.copy(self.ell) 
		elif self.ell < 1e-12:
			self.ell = 0.00000001
			dist_ell = abs(self.ell - self.ell_old) 
			self.ell_old = np.copy(self.ell)  
		else:
			dist_ell = self.ell - self.ell_old
			self.ell_old = np.copy(self.ell) 

		return dist_ell 
	
 
	# @gl.timeit('update_phi')
	def _update_phi(self): 
		res = root(func_phi_dynamic_t, self.phi_old, args=(self))  
		self.phi = (res.x)[0]   

		# self.phi = brentq(func_phi_dynamic_t, a=0.000001,b=0.99999, args=(self))   
		if self.phi > 1:  
			print('self.phi:', self.phi)
			self.phi = 0.9999999
			dist_phi = abs(self.phi - self.phi_old) 
			self.phi_old = np.copy(self.phi)
			# self.phi = np.copy(self.phi_old)
			# dist_phi = 0
		elif self.phi < 1e-12: 
			self.phi = 0.00000001
			dist_phi = abs(self.phi - self.phi_old) 
			self.phi_old = np.copy(self.phi) 
		else:
			dist_phi = abs(self.phi - self.phi_old) 
			self.phi_old = np.copy(self.phi)

		return dist_phi
	
	# @gl.timeit('update_beta')
	def _update_beta(self):
		res = root(func_beta_dynamic_t, self.beta_old, args=(self))  
		self.beta = res.x   
		# self.beta_hat[:] = np.copy(self.beta)

		if self.beta > 1:   
			print('beta_hat:', self.beta)
			self.beta = 0.9999
			dist_beta = abs(self.beta - self.beta_old) 
			self.beta_old = np.copy(self.beta) 
		elif self.beta < 1e-12:
			raise ValueError('The anomaly coefficient beta has to be in [0, 1]!')  
		else:
			dist_beta = abs(self.beta - self.beta_old) 
			self.beta_old = np.copy(self.beta) 

		return dist_beta
	
	
	# @gl.timeit_cum('update_U')
	# @timer_func
	def _update_U(self,subs_nzp,subs_nz):
		"""
			Update out-going membership matrix.
			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			Returns
			-------
			dist_u : float
					 Maximum distance between the old and the new membership matrix u. 
		"""    
		if self.flag_anomaly == True:
			if self.T > 0:
				Du = np.einsum('ij,jk->ik', 1 - self.Qij_dense[-1], self.v)
				beta_hat = np.ones(self.T+1) * self.beta
				beta_hat[0] = 1

				if self.assortative == False:
					w_k = np.einsum('akq,a->kq', self.w, beta_hat)
					Z_uk = np.einsum('iq,kq->ik', Du, w_k)
				else:
					w_k = np.einsum('ak,a->k', self.w, beta_hat)
					Z_uk = np.einsum('ik,k->ik', Du, w_k) 

				self.u = self._update_membership_anomaly(subs_nzp, self.u, self.v, self.w[1:], 1, data=self.data_M_nz_Q) + \
						 self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 1, data=self.data_M_nz_Q_0)
			else: 
				Du = np.einsum('ij,jk->ik', 1 - self.Q_ij_dense_0[0], self.v) 
				if self.assortative == False:
					w_k = np.einsum('akq->kq', self.w)(self.w[0])[np.newaxis]
					Z_uk = np.einsum('iq,kq->ik', Du, w_k)
				else:
					w_k = np.einsum('ak->k', self.w)
					Z_uk = np.einsum('ik,k->ik', Du, w_k)  
				self.u = self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 1, data=self.data_M_nz_Q_0) 

			Z_uk += self.bg
			self.u = (self.ag - 1) + self.u_old * self.u
			non_zeros = Z_uk > 0.
			self.u[Z_uk == 0] = 0.
			self.u[non_zeros] /= Z_uk[non_zeros]


		else: #self.flag_anomaly == False 
			if self.T > 0:  
				self.u = self._update_membership_anomaly(subs_nzp, self.u, self.v, self.w[1:], 1, data=self.data_M_nz_Q) + \
						 self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 1, data=self.data_M_nz_Q_0)

				if not self.constrained:
					Du = np.einsum('iq->q', self.v)
					beta_hat = np.ones(self.T+1) * self.beta
					beta_hat[0] = 1
					if not self.assortative: 
						w_k = np.einsum('aqk,a->qk', self.w, beta_hat) 
						Z_uk = np.einsum('q,kq->k', Du, w_k)
					else: 
						w_k = np.einsum('ak,a->k', self.w, beta_hat) 
						Z_uk = np.einsum('k,k->k', Du, w_k)   
					# gl.timer.cum('update_mem')
				else: 
					raise ValueError('Constrained should be False')
			else: #self.T == 0 
				self.u = self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 1, data=self.data_M_nz_Q_0) 
				if not self.constrained:
					Du = np.einsum('iq->q', self.v)
					if not self.assortative:
						# w_k = np.einsum('akq->kq', self.w)
						Z_uk = np.einsum('q,kq->k', Du, self.w[0])
					else:
						# w_k = np.einsum('ak->k', self.w)
						Z_uk = np.einsum('k,k->k', Du, self.w[0])  
					# gl.timer.cum('update_mem')
				else: 
					raise ValueError('Constrained should be False')
			self.u = (self.ag - 1) + self.u_old * self.u
			non_zeros = Z_uk > 0.   
			self.u[:, Z_uk == 0] = 0.     
			self.u[:, non_zeros] /= (self.bg+Z_uk[np.newaxis ,non_zeros]) 

		low_values_indices = self.u < self.err_max  # values are too low
		self.u[low_values_indices] = 0.  # and set to 0.
 
		dist_u = np.amax(abs(self.u - self.u_old))  
		self.u_old = np.copy(self.u) 

		return dist_u

	# @gl.timeit_cum('update_V')
	def _update_V(self,subs_nzp,subs_nz):
		"""
			Update in-coming membership matrix.
			Same as _update_U but with:
			data <-> data_T
			w <-> w_T
			u <-> v
			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			Returns
			-------
			dist_v : float
					 Maximum distance between the old and the new membership matrix v.
		"""

		if self.flag_anomaly == True: 
			if self.T > 0: 
				Dv = np.einsum('ji,ik->jk', 1-self.Qij_dense[-1], self.u)
				beta_hat = np.ones(self.T+1) * self.beta
				beta_hat[0] = 1

				if self.assortative == False:
					w_k = np.einsum('aqk,a->qk', self.w, beta_hat)
					Z_vk = np.einsum('ik,qk->ik', Dv, w_k)
				else:
					w_k = np.einsum('ak,a->k', self.w, beta_hat)
					Z_vk = np.einsum('ik,k->ik', Dv, w_k)
				Z_vk += self.bg
				self.v = self._update_membership_anomaly(subs_nzp, self.u, self.v, (self.w[1:]), 2,data=self.data_M_nz_Q) + \
						 self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 2,data=self.data_M_nz_Q_0) 
			else: 
				Dv = np.einsum('ji,ik->jk', 1 - self.Q_ij_dense_0[0], self.u) 

				if self.assortative == False: 
					Z_vk = np.einsum('iq,kq->ik', Dv, self.w[0,:])
				else: 
					Z_vk = np.einsum('ik,k->ik', Dv, self.w[0])

				Z_vk += self.bg 
				self.v = self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 2,data=self.data_M_nz_Q_0)
			
			self.v = (self.ag - 1) + self.v_old * self.v  
			non_zeros = Z_vk > 0.
			self.v[Z_vk == 0] = 0.
			self.v[non_zeros] /= Z_vk[non_zeros]


		else: #flag_anomaly == False 
			if self.T > 0:  
				if not self.constrained:
					Dv = np.einsum('iq->q', self.u)
					beta_hat = np.ones(self.T+1) * self.beta
					beta_hat[0] = 1
					if not self.assortative: 
						w_k = np.einsum('aqk,a->qk', self.w, beta_hat)  
						Z_vk = np.einsum('q,qk->k', Dv, w_k)
					else: 
						w_k = np.einsum('ak,a->k', self.w, beta_hat) 
						Z_vk = np.einsum('k,k->k', Dv, w_k)   
					self.v = self._update_membership_anomaly(subs_nzp, self.u, self.v, (self.w[1:]), 2,data=self.data_M_nz_Q) + \
						 self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 2,data=self.data_M_nz_Q_0)  
				else:
					raise ValueError('Constrained should be False')  
			else:
				self.v = self._update_membership_anomaly(subs_nz, self.u, self.v, (self.w[0])[np.newaxis], 2,data=self.data_M_nz_Q_0)  
				if not self.constrained:
					Dv = np.einsum('iq->q', self.u)
					if not self.assortative:
						# w_k = np.einsum('aqk->qk', self.w)
						Z_vk = np.einsum('q,qk->k', Dv, self.w[0])
					else:
						# w_k = np.einsum('ak->k', self.w)
						Z_vk = np.einsum('k,k->k', Dv, self.w[0])   
				else:
					raise ValueError('Constrained should be False')
			self.v = (self.ag - 1) + self.v_old * self.v
			non_zeros = Z_vk > 0
			self.v[:, Z_vk == 0] = 0.
			self.v[:, non_zeros] /=(self.bg+Z_vk[np.newaxis, non_zeros])
			
 
		low_values_indices = self.v < self.err_max  # values are too low
		self.v[low_values_indices] = 0.  # and set to 0.
		dist_v = np.amax(abs(self.v - self.v_old))
		self.v_old = np.copy(self.v) 

		return dist_v
	
	
	# @gl.timeit_cum('update_W_ass')
	# @timer_func
	def _update_W_assortative(self,subs_nzp,subs_nz):
		"""
			Update affinity tensor (assuming assortativity).
			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			Returns
			-------
			dist_w : float
					 Maximum distance between the old and the new affinity tensor w.
		""" 
		if self.flag_anomaly == True:

			# sub_w_nz = self.w.nonzero()
			uttkrp_DKQ = np.zeros_like(self.w) 
			for idx,(a,i,j) in enumerate(zip(*subs_nzp)):   
				uttkrp_DKQ[1+a, :] += self.data_M_nz_Q[idx] * self.u[i] * self.v[j]
			for idx, (a, i, j) in enumerate(zip(*subs_nz)): 
				uttkrp_DKQ[0, :] += self.data_M_nz_Q_0[idx] * self.u[i] * self.v[j] 

			self.w = (self.ag - 1) + self.w_old * uttkrp_DKQ
			
			UQk = np.einsum('ij,ik->jk', (1-self.Qij_dense[-1]), self.u)
			Zk  = np.einsum('jk,jk->k', UQk, self.v)

			# Zk += self.bg
			# non_zeros = Zk > 0
			# L = self.w.shape[0] 
			# for l in range(L):
			# 	if l == 0:
			# 		self.w[l, non_zeros] /= Zk[non_zeros]
			# 	else:  
			# 		self.w[l, non_zeros] /= (self.beta * Zk[non_zeros])
			
			# Z *=  self.beta_hat[self.T] * self.T 
			if self.T > 0:
				beta_hat = np.ones(self.T+1) * self.beta
				beta_hat[0] = 1 
				Zk = np.einsum('a,k->ak', beta_hat, Zk)   
				Zk += self.bg 
				non_zeros = Zk > 0    
				self.w[non_zeros] /= Zk[non_zeros] 
			else:
				Zk += self.bg 
				non_zeros = Zk > 0    
				self.w[:,non_zeros] /= Zk[non_zeros] 


		else: # flag_anomaly == False  

			uttkrp_DKQ = np.zeros_like(self.w)   

			for idx,(a,i,j) in enumerate(zip(*subs_nzp)):  
				uttkrp_DKQ[a+1,:] += self.data_M_nz_Q[idx] * self.u[i] *self.v[j]    
			
			for idx, (a, i, j) in enumerate(zip(*subs_nz)):  
				uttkrp_DKQ[0, :] += self.data_M_nz_Q_0[idx] * self.u[i] * self.v[j]   
			
			self.w =   (self.ag - 1) + self.w_old * uttkrp_DKQ    

			Z = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0))) 
			# Z *=  self.beta_hat[self.T] * self.T 
			if self.T > 0:
				beta_hat = np.ones(self.T+1) * self.beta
				beta_hat[0] = 1 
				Z = np.einsum('a,k->ak', beta_hat, Z)   
				Z += self.bg 
				non_zeros = Z > 0    
				self.w[non_zeros] /= Z[non_zeros] 
			else:
				Z += self.bg 
				non_zeros = Z > 0    
				self.w[:,non_zeros] /= Z[non_zeros] 

			# low_values_indices = self.w < self.err_max  # values are too low
			# self.w[low_values_indices] = 0.  # and set to 0. 

		 
		low_values_indices = self.w < self.err_max  # values are too low
		self.w[low_values_indices] = 0.  # and set to 0.
		dist_w = np.amax(abs(self.w - self.w_old))
		self.w_old = np.copy(self.w) 

		return dist_w


	def _update_membership_anomaly(self, subs_nz, u, v, w, m, data=None):
		"""
			Return the Khatri-Rao product (sparse version) used in the update of the membership matrices.
			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.
			m : int
				Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
				works with the matrix u; if 2 it works with v.
			Returns
			-------
			uttkrp_DK : ndarray
						Matrix which is the result of the matrix product of the unfolding of the tensor and the
						Khatri-Rao product of the membership matrix.
		"""
		if data is None:
			data = self.data_M_nz_Q
		if not self.assortative:
			uttkrp_DK = sp_uttkrp(data, subs_nz, m, u, v, w)
		else:
			uttkrp_DK = sp_uttkrp_assortative(data, subs_nz, m, u, v, w)

		return uttkrp_DK

	def _check_for_convergence(self,datam0, data0, subs_nzp, subs_nz,T,r, it, loglik, coincide, convergence, data_T=None):
		"""
			Check for convergence by using the pseudo log-likelihood values.
			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			it : int
				 Number of iteration.
			loglik : float
					 Pseudo log-likelihood value.
			coincide : int
					   Number of time the update of the pseudo log-likelihood respects the tolerance.
			convergence : bool
						  Flag for convergence.
			data_T : sptensor/dtensor
					 Graph adjacency tensor (transpose).
			Returns
			-------
			it : int
				 Number of iteration.
			loglik : float
					 Log-likelihood value.
			coincide : int
					   Number of time the update of the pseudo log-likelihood respects the tolerance.
			convergence : bool
						  Flag for convergence.
		""" 
		if it % 10 == 0:
			old_L = loglik
			loglik = self.__Likelihood( datam0,data0, data_T, subs_nzp, subs_nz,T)
			if abs(loglik - old_L) < self.tolerance:
				coincide += 1
			else:
				coincide = 0
		if coincide > self.decision:
			convergence = True
		it += 1

		return it, loglik, coincide, convergence

	# @gl.timeit('Likelihood')
	# @timer_func
	def __Likelihood(self, datam0, data0, data_T, subs_nzp, subs_nz,T,EPS=1e-12):
		"""
			Compute the pseudo log-likelihood of the data.
			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T : sptensor/dtensor
					 Graph adjacency tensor (transpose).
			Returns
			-------
			l : float
				Pseudo log-likelihood value.
		"""   
		lambda0_ija   = self._lambda0_full(self.u, self.v, self.w[1:])   
		lambda0_ija_0 = self._lambda0_full(self.u, self.v, (self.w[0])[np.newaxis])   

		if self.flag_anomaly == True:  
			l = 0
			'''
			Term containing Q and mu at t=0,  
			'''  
			if 1 - self.mupr >= 0 :
				l += np.log(1-self.mupr+EPS) * (self.b1mQsum)  # (1-Q[0])*log (1-mu)
			if self.mupr >= 0 :
				l += np.log(self.mupr+EPS) * (self.Qsum)    # Q[0]*log mu
			
			'''
			Entropy of Bernoulli in Q  (1)
			'''

			if self.T > 0: 
				non_zeros = (self.Qij_dense[0]) > 0
				non_zeros1 = (1-(self.Qij_dense[0])) > 0   

				l -= ((self.Qij_dense[0])[non_zeros] * np.log((self.Qij_dense[0])[non_zeros]+EPS)).sum()    # Q*log Q
				l -= ((1-(self.Qij_dense[0]))[non_zeros1]*np.log((1-(self.Qij_dense[0]))[non_zeros1]+EPS)).sum()   # (1-Q)*log(1-Q)  
			else: 
				non_zeros = self.Q_ij_dense_0 > 0
				non_zeros1 = (1-self.Q_ij_dense_0) > 0   

				l -= ((self.Q_ij_dense_0)[non_zeros] * np.log((self.Q_ij_dense_0)[non_zeros]+EPS)).sum()    # Q*log Q
				l -= ((1-self.Q_ij_dense_0)[non_zeros1]*np.log((1-self.Q_ij_dense_0)[non_zeros1]+EPS)).sum()   # (1-Q)*log(1-Q)  
			
			# non_zeros = self.Q_ij_dense_0 > 0
			# non_zeros1 = (1-self.Q_ij_dense_0) > 0   

			# l -= (self.Q_ij_dense_0[non_zeros] * np.log(self.Q_ij_dense_0[non_zeros]+EPS)).sum()    # Q*log Q
			# l -= ((1-self.Q_ij_dense_0)[non_zeros1] * np.log((1-self.Q_ij_dense_0)[non_zeros1]+EPS)).sum()   # (1-Q)*log(1-Q) 

			'''
			Term containing  (1-Q), lambda, beta, and A, (3)
			'''  
			if self.T > 0:   
				logM = np.log(self.lambda0_nz+EPS)  
				if isinstance(datam0, skt.dtensor):
					Alog = datam0[datam0.nonzero()] * (1-self.Qij_dense[0])[datam0.subs] * logM
				elif isinstance(datam0, skt.sptensor):
					Alog = (datam0.vals * (1-self.Qij_nz) * logM).sum()
				l += Alog 

			
				if (1 - self.beta) >= 0:     
					l += np.log(1-self.beta+EPS) * self.b1mQAtm1At  
				
				if self.beta >= 0:     
					l += np.log(self.beta+EPS) * self.b1mQ1mAtm1At 
					l += np.log(self.beta+EPS) * self.b1mQAtm11mAt 
				l -= self.beta  * ((1-self.Qij_dense[0]) * lambda0_ija).sum()
				# l -= self.beta  * ((1-self.Qij_dense[-1]) * lambda0_ija).sum()

			
			logM_0 = np.log(self.lambda0_nz_0+EPS)  
			if isinstance(data0, skt.dtensor):
				Alog_0 = data0[data0.nonzero()] * ((1-self.Q_ij_dense_0)[data0.subs]) * logM_0 
			elif isinstance(data0, skt.sptensor):
				Alog_0 = (data0.vals * (1-self.Q_ij_nz_0) * logM_0).sum()   
			l += Alog_0 

			l -= ((1-self.Q_ij_dense_0) * lambda0_ija_0).sum()  
			'''
			Term containing Q, phi, ell, and pi (4)
			''' 
			if  self.pibr > 0:
				l += np.log(self.pibr+EPS) * self.Q0A0 # A(0) *Q*log pi
			l -= (self.Qsum * self.pibr)
		
			if self.T > 0:
				if self.ell >= 0:
					# l += np.einsum('a, a-> ', self.Q1mAtm1At, np.log(self.ell_hat+EPS))		# (1-A(t-1)) * A(t) *Q*log l
					l += self.Q1mAtm1At * np.log(self.ell+EPS)  		# (1-A(t-1)) * A(t) *Q*log l
			
			
				if (1 - self.phi) >= 0:    
					l += np.log(1-self.phi+EPS) * self.QAtm1At # Q * A(t-1) * A(t) * log (1-phi)
					
				
				if  self.phi >= 0:      
					l += np.log(self.phi+EPS) * self.Q1mAtm1At # Q * (1-A(t-1)) * A(t) * log phi
					l += np.log(self.phi+EPS) * self.QAtm11mAt       # Q * A(t-1) * (1-A(t)) * log phi
				l-= self.phi * self.Qsum * self.T * self.ell   # Q * phi * ell 
				# l-= self.phi * np.einsum('ij, a-> ', self.Qij_dense[-1], self.ell_hat) # Q * phi * ell 

		
		else:   # flag_anomaly = False
			l = 0
			if self.T > 0: 
				l -= self.beta * lambda0_ija.sum() 
				logM = np.log(self.M_nz)  
				if isinstance(datam0, skt.dtensor):
					Alog = datam0[datam0.nonzero()] * logM 
				elif isinstance(datam0, skt.sptensor):
					Alog = (datam0.vals * logM).sum()  
				l += Alog 
				if self.beta > 0:
					l += np.log(self.beta+EPS) * self.b1mAtm1At  
					l += np.log(self.beta+EPS) * self.bAtm11mAt
				if (1- self.beta) > 0:
					l += np.log(1- self.beta+EPS) * self.bAtm1At


			logM_0 = np.log(self.M_nz_0)  
			if isinstance(data0, skt.dtensor):
				Alog_0 = data0[data0.nonzero()] * logM_0 
			elif isinstance(data0, skt.sptensor): 
				Alog_0 = (data0.vals * logM_0).sum()  
			l += Alog_0  
			l -= lambda0_ija_0.sum()
		
		if self.ag > 1.: 
			l += (self.ag -1) * np.log(self.u+EPS).sum()
			l += (self.ag -1) * np.log(self.v+EPS).sum()
		if self.bg > 0.:
			l -= self.bg * self.u.sum()
			l -= self.bg * self.v.sum()
		
		if np.isnan(l):
			print("Likelihood is NaN!!!!")
			sys.exit(1)
		else:
			return l

	def _lambda0_full(self, u, v, w):
		"""
			Compute the mean lambda0 for all entries.
			Parameters
			----------
			u : ndarray
				Out-going membership matrix.
			v : ndarray
				In-coming membership matrix.
			w : ndarray
				Affinity tensor.
			Returns
			-------
			M : ndarray
				Mean lambda0 for all entries.
		"""  
		M = np.einsum('ik,jk->ijk', u, v)
		M = np.einsum('ijk,ak->aij', M, w) 
		return M

	def _update_optimal_parameters(self):
		"""
			Update values of the parameters after convergence.
		"""
		self.u_f = np.copy(self.u) 
		self.v_f = np.copy(self.v)
		self.w_f = np.copy(self.w)
		if self.T > 0:  
			if self.flag_anomaly:
				self.ell_f = np.copy(self.ell)
				self.phi_f = np.copy(self.phi)
		else:  
			if not self.fix_ell:
				if self.flag_anomaly == True: 
					self.ell_f = float(self.pibr)
				else: 
					self.ell_f = 0.  
			if not self.fix_phi: 
				self.phi_f = 1. 
		self.pibr_f = np.copy(self.pibr)
		self.mupr_f = np.copy(self.mupr)

		if self.flag_anomaly == True:
			self.Q_ij_dense_f = np.copy(self.Qij_dense[-1])
		else:
			self.Q_ij_dense_f = np.zeros((1,self.N,self.N)) 
		
		if self.fix_beta == False:
			self.beta_f = np.copy(self.beta) 

	def output_results(self, nodes):
		"""
			Output results.
			Parameters
			----------
			nodes : list
					List of nodes IDs.
		""" 

		outfile = self.out_folder + 'theta_' + self.end_file 
		np.savez_compressed(outfile + '.npz', u=self.u_f, v=self.v_f, w=self.w_f, beta = self.beta_f,max_it=self.final_it, pi=self.pibr_f, mu=self.mupr_f, phi=self.phi_f, ell=self.ell_f,
							maxL=self.maxL, nodes=nodes)
		print(f'\nInferred parameters saved in: {outfile + ".npz"}')
		print('To load: theta=np.load(filename), then e.g. theta["u"]')


def sp_uttkrp(vals, subs, m, u, v, w):
	"""
		Compute the Khatri-Rao product (sparse version).
		Parameters
		----------
		vals : ndarray
			   Values of the non-zero entries.
		subs : tuple
			   Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
			   equal to the dimension of tensor.
		m : int
			Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
			works with the matrix u; if 2 it works with v.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		Returns
		-------
		out : ndarray
			  Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
			  of the membership matrix.
	""" 

	if m == 1:
		D, K = u.shape
		out = np.zeros_like(u)
	elif m == 2:
		D, K = v.shape
		out = np.zeros_like(v)

	for k in range(K):
		
		tmp = vals.copy()
		if m == 1:  # we are updating u
			w_I = w[0, k, :]
			tmp *= (w_I[np.newaxis,:].astype(tmp.dtype) * v[subs[2], :].astype(tmp.dtype)).sum(axis=1)
		elif m == 2:  # we are updating v
			w_I = w[0, :, k]
			tmp *= (w_I[np.newaxis,:].astype(tmp.dtype) * u[subs[1], :].astype(tmp.dtype)).sum(axis=1)
		out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D) 

	return out


def sp_uttkrp_assortative(vals, subs, m, u, v, w):
	"""
		Compute the Khatri-Rao product (sparse version) with the assumption of assortativity.
		Parameters
		----------
		vals : ndarray
			   Values of the non-zero entries.
		subs : tuple
			   Indices of elements that are non-zero. It is a n-tuple of array-likes and the length of tuple n must be
			   equal to the dimension of tensor.
		m : int
			Mode in which the Khatri-Rao product of the membership matrix is multiplied with the tensor: if 1 it
			works with the matrix u; if 2 it works with v.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		Returns
		-------
		out : ndarray
			  Matrix which is the result of the matrix product of the unfolding of the tensor and the Khatri-Rao product
			  of the membership matrix.
	"""

	if m == 1:
		D, K = u.shape
		out = np.zeros_like(u)
	elif m == 2:
		D, K = v.shape
		out = np.zeros_like(v)

	for k in range(K):
		tmp = vals.copy()
		if m == 1:  # we are updating u
			tmp *= w[subs[0], k].astype(tmp.dtype) * v[subs[2], k].astype(tmp.dtype)
		elif m == 2:  # we are updating v
			tmp *= w[subs[0], k].astype(tmp.dtype) * u[subs[1], k].astype(tmp.dtype)
		out[:, k] += np.bincount(subs[m], weights=tmp, minlength=D)

	return out


def get_item_array_from_subs(A, ref_subs):
	"""
		Get values of ref_subs entries of a dense tensor.
		Output is a 1-d array with dimension = number of non zero entries.
	"""

	return np.array([A[a, i, j] for a, i, j in zip(*ref_subs)])


def preprocess(X):
	"""
		Pre-process input data tensor.
		If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
		Parameters
		----------
		X : ndarray
			Input data (tensor).
		Returns
		-------
		X : sptensor/dtensor
			Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
	"""

	if not X.dtype == np.dtype(int).type:
		X = X.astype(int)
	if isinstance(X, np.ndarray) and is_sparse(X):
		X = sptensor_from_dense_array(X)
	else:
		X = skt.dtensor(X)

	return X


def is_sparse(X):
	"""
		Check whether the input tensor is sparse.
		It implements a heuristic definition of sparsity. A tensor is considered sparse if:
		given
		M = number of modes
		S = number of entries
		I = number of non-zero entries
		then
		N > M(I + 1)
		Parameters
		----------
		X : ndarray
			Input data.
		Returns
		-------
		Boolean flag: true if the input tensor is sparse, false otherwise.
	"""

	M = X.ndim
	S = X.size
	I = X.nonzero()[0].size

	return S > (I + 1) * M


def sptensor_from_dense_array(X):
	"""
		Create an sptensor from a ndarray or dtensor.
		Parameters
		----------
		X : ndarray
			Input data.
		Returns
		-------
		sptensor from a ndarray or dtensor.
	"""

	subs = X.nonzero()
	vals = X[subs]

	return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)

def transpose_tensor(A):
	'''
	Assuming the first index is for the layer, it transposes the second and third
	'''
	return np.einsum('aij->aji',A)

def plot_L(values, indices = None, k_i = 5, figsize=(5, 3), int_ticks=False, xlab='Iterations'):

	fig, ax = plt.subplots(1,1, figsize=figsize)
	#print('\n\nL: \n\n',values[k_i:])

	if indices is None:
		ax.plot(values[k_i:])
	else:
		ax.plot(indices[k_i:], values[k_i:])
	ax.set_xlabel(xlab)
	ax.set_ylabel('Log-likelihood values')
	if int_ticks:
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.grid()

	plt.tight_layout()
	plt.show()

def u_with_lagrange_multiplier(u,x,y):
	denominator = x.sum() - (y * u).sum()
	f_ui = x / (y + denominator)
	# if np.allclose(u.sum() ,0): return 0.
	if (u < 0).sum() > 0 : return 100. * np.ones(u.shape)
	# return (f_u - u).flatten()
	return (f_ui - u) 



def func_beta_dynamic_t(beta_t, obj):
	EPS = 1e-45
	# TO DO: generalise for anomaly=False 
	# self.b1mAtm1At,self.bAtm11mAt,self.bAtm1At
	assert type(obj) is Dyn_ACD_wtemp  
	if obj.flag_anomaly: 
		Du = np.einsum('ij,jk->ik', (1-obj.Qij_dense[0]),obj.v)
		if not obj.assortative:
			w_k  = np.einsum('akq->kq', obj.w[1:])
			Z_uk = np.einsum('iq,kq->ik', Du, w_k)
		else:
			w_k = np.einsum('ak->k', obj.w[1:])  
			Z_uk = np.einsum('ik,k->ik', Du, w_k)
		lambda0_ija = np.einsum('ik,ik->',obj.u, Z_uk)

		bt  = - (lambda0_ija)  # (1-Q) * \lambda   
		bt -=   obj.b1mQAtm1At  / (1-beta_t+EPS)  # adding Aij(t-1)*Aij(t)  

		bt += obj.b1mQ1mAtm1At / (beta_t+EPS)   
		bt += obj.b1mQAtm11mAt / (beta_t+EPS)   # adding Aij(t-1)*(1-Aij(t))  

	else: 
		# print((obj.w).sum(axis=0))
		lambda0_ija = np.einsum('k,k->k',(obj.u).sum(axis=0), (obj.w[1:]).sum(axis=0))  
		lambda0_ija = np.einsum('k,k->',(obj.v).sum(axis=0), lambda0_ija) 

		bt =  - (lambda0_ija) 
		bt -=  obj.bAtm1At / (1-beta_t)  # adding Aij(t-1)*Aij(t) 

		bt += obj.b1mAtm1At / (beta_t+EPS)  # adding sum A_hat from 1 to T 
		bt += obj.bAtm11mAt / (beta_t+EPS)  # adding Aij(t-1)*(1-Aij(t))   
	return bt

def func_phi_dynamic_t(phi_t, obj):
	EPS = 1e-12 
	assert type(obj) is Dyn_ACD_wtemp
	# pt  = - np.einsum('a,ij->', obj.ell_hat, obj.Qij_dense[-1])
	pt  = - obj.ell * obj.T * obj.Qsum
	pt -=  (obj.QAtm1At)  / (1-phi_t+EPS)  # adding Aij(t-1)*Aij(t)  
	pt += (obj.Q1mAtm1At) / (phi_t+EPS)  # adding sum A_hat from 1 to T. (1-A(t-1))A(t)
	pt += (obj.QAtm11mAt) / (phi_t+EPS)  # adding Aij(t-1)*(1-Aij(t))
	return pt

def calculate_lambda_full(u, v, w):
	"""
		Compute the mean lambda0 for all entries.
		Parameters
		----------
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		Returns
		-------
		M : ndarray
			Mean lambda0 for all entries.
	"""
	M = np.einsum('ik,jk->ijk', u, v)
	M = np.einsum('ijk,ak->aij', M, w)
	return M



