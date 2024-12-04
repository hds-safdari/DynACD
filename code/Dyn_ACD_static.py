"""
	Class definition of Dynamic Anomaly Detection, the algorithm to perform inference on dynamical networks.
	The latent variables are related to community memberships and anomaly parameters parameters.
	The latent variables at T=0 are drived separately from those at T>0. 
	In this version of the algorithm, all the latent variavles are static. 
"""

from __future__ import print_function
import time
import sys
import sktensor as skt
import numpy as np
import pandas as pd
from termcolor import colored
from scipy.stats import poisson 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq, root,root_scalar


EPS = 1e-12

class Dyn_ACD_static:
	def __init__(self, N=100, L=1, K=3, undirected=False, initialization=0, ag=1.0,bg=0., rseed=100, inf=1e20, err_max=1e-18, err=0.0001, 
				 N_real=1, tolerance=0.0001, decision=10, max_iter=500, out_inference=False,
				 in_parameters = '../data/input/synthetic/theta_500_3_5.0_6_0.05_0.2_10',
				 fix_communities=False,fix_w=False,plot_loglik=False,beta0 = 0.1,phi0 = 0.05, ell0= 0.1, pibr0 = 0.05, mupr0= 0.05,
				 fix_pibr=True, fix_mupr=False,fix_phi = False, fix_ell= False,
				 out_folder='../data/output/', end_file='.dat', assortative=True,  fix_beta=False,
				 constrained=False, verbose=False, flag_anomaly = False, weighted = True):

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
		self.decision = decision  # convergence parameter
		self.max_iter = max_iter  # maximum number of EM steps before aborting
		self.out_inference = out_inference  # flag for storing the inferred parameters
		self.out_folder = out_folder  # path for storing the output
		self.end_file = end_file  # output file suffix
		self.assortative = assortative  # if True, the network is assortative
		self.fix_pibr = fix_pibr  # if True, the pibr parameter is fixed
		self.fix_mupr = fix_mupr  # if True, the mupr parameter is fixed
		self.fix_phi = fix_phi  # if True, the phi parameter is fixed
		self.fix_ell = fix_ell  # if True, the l parameter is fixed 
		self.fix_beta = fix_beta  # if True, the beta parameter is fixed
		self.fix_communities = fix_communities
		self.fix_w = fix_w
		self.constrained = constrained  # if True, use the configuration with constraints on the updates
		self.verbose = verbose  # flag to print details

		self.ag = ag # shape of gamma prior
		self.bg = bg # rate of gamma prior 
		self.beta0 = beta0
		self.pibr0 = pibr0  # initial value for the mu in 
		self.mupr0 = mupr0  # initial value for the pi in Bernolie dist
		self.phi0 = phi0  # initial value for the reciprocity coefficient 
		self.ell0 = ell0
		self.plot_loglik = plot_loglik 
		self.in_parameters = in_parameters
		self.flag_anomaly = flag_anomaly 
		self.weighted = weighted

		if initialization not in {0, 1, 2, 3}:  # indicator for choosing how to initialize u, v and w
			raise ValueError('The initialization parameter can be either 0, 1 or 2. It is used as an indicator to '
							 'initialize the membership matrices u and v and the affinity matrix w. If it is 0, they '
							 'will be generated randomly, otherwise they will upload from file.')
		self.initialization = initialization
		


		if self.initialization >= 2:
			self.theta = np.load(self.in_parameters + '.npz',allow_pickle=True) 
			self.N, self.K = self.theta['u'].shape  
			self.beta0  = self.theta['beta']
		if self.initialization >= 2:
			self.phi0   = self.theta['phi']
			self.mupr0  = self.theta['mu']
			self.pibr0  = self.theta['pi']
			self.ell0   = self.theta['ell'] 
		
		if self.pibr0 is not None:
			if (self.pibr0 < 0) or (self.pibr0 > 1):
				raise ValueError('The reciprocity coefficient pibr0 has to be in [0, 1]!')

		if self.mupr0 is not None:
			if (self.mupr0 < 0) or (self.mupr0 > 1):
				raise ValueError('The reciprocity coefficient mupr0 has to be in [0, 1]!')
		
		self.u0 = np.zeros((self.N, self.K), dtype=float)  # out-going membership
		self.v0 = np.zeros((self.N, self.K), dtype=float)  # in-going membership
		self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
		self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership
		
		# values of the parameters in the previous iteration
		self.u_old = np.zeros((self.L, self.N, self.K), dtype=float)  # out-going membership
		self.v_old = np.zeros((self.L, self.N, self.K), dtype=float)  # in-going membership
		self.u0_old = np.zeros((self.L, self.N, self.K), dtype=float)  # out-going membership
		self.v0_old = np.zeros((self.L, self.N, self.K), dtype=float)  # in-going membership
		
		# values of the affinity tensor: in this case w is always ASSORTATIVE 
		if self.assortative:  # purely diagonal matrix
			self.w = np.zeros((self.L, self.K), dtype=float)
			self.w_old = np.zeros((self.L, self.K), dtype=float)
			self.w_f = np.zeros((self.L, self.K), dtype=float)
			self.w0 = np.zeros((self.L, self.K), dtype=float)
			self.w0_old = np.zeros((self.L, self.K), dtype=float)
			self.w0_f = np.zeros((self.L, self.K), dtype=float)
		else:
			self.w = np.zeros((self.L, self.K, self.K), dtype=float)
			self.w_old = np.zeros((self.L, self.K, self.K), dtype=float)
			self.w_f = np.zeros((self.L, self.K, self.K), dtype=float) 
			self.w0 = np.zeros((self.L, self.K, self.K), dtype=float)
			self.w0_old = np.zeros((self.L, self.K, self.K), dtype=float)
			self.w0_f = np.zeros((self.L, self.K, self.K), dtype=float) 

		if self.ag < 1 :
			self.ag = 1.
		if self.bg < 0:
			self.bg = 0. 
		
		
		if self.fix_beta:
			self.beta = self.beta_old = self.beta_f = self.beta0
		

		if self.flag_anomaly == False:
			# TO DO: check these values. 
			self.phi =  self.phi_old = self.phi_f =self.phi0 =  1.
			self.ell =  self.ell_old = self.ell_f = self.ell0= 0.
			self.pibr =  self.pibr_old = self.pibr_f = self.pibr0= 0.
			self.mupr = self.mupr_old = self.mupr_f = self.mupr0 = 0.
			self.fix_phi = self.fix_ell = self.fix_pibr = self.fix_mupr = True
			self.weighted = False

		if self.fix_phi and self.flag_anomaly == True:
			self.phi = self.phi_old = self.phi_f = self.phi0
		
		if self.fix_ell and self.flag_anomaly == True:
			self.ell = self.ell_old = self.ell_f = self.ell0
		

		if self.fix_pibr and self.flag_anomaly == True:
			self.pibr = self.pibr_old = self.pibr_f = self.pibr0
		
		if self.fix_mupr and self.flag_anomaly == True:
			self.mupr = self.mupr_old = self.mupr_f = self.mupr0 

	def fit(self, data, T, nodes, mask=None):
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
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
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
		# self.K = K
		self.N = data.shape[-1]  
		T = max(0, min(T, data.shape[0]-1))
		self.T = T  
		print('='*60)
		print('self.T:', self.T)
		data = data[:T+1,:,:]  
		'''
		Pre-process data
		'''
		data_b1mAtm1At = np.zeros(data.shape)	# Aij(t)*(1-Aij(t-1))
		data0 = np.zeros(data.shape[0])
		# data_tm1 = np.zeros_like(data) 
		data_bAtm11mAt = np.zeros(data.shape)	# A(t-1) (1-A(t))
		data_bAtm1At = np.zeros(data.shape)	# A(t-1) A(t) 

		data_b1mAtm1At[0,:,:] = data[0,:,:] # to calculate numerator containing Aij(t)*(1-Aij(t-1))   
		# self.E0 = 0. # to calculate denominator eta
		# self.Etg0 = np.sum(data[:-1]) # to calculate denominator eta

		self.b1mAtm1At = 0
		self.bAtm11mAt = 0
		self.bAtm1At= 0

		data_Td = np.einsum('aij->aji', data) # to calculate  Aji(t) s in Q_{ij}
		data0  = data[0,:,:]
		data0_T = np.einsum('ij->ji', data0) # to calculate Pois terms by Aji(t) s in Q_{ij}
		data0   = data0[np.newaxis,:,:]
		data0_T = data0_T[np.newaxis,:,:]

		if T > 0:
			b1mAtm1At_l = 0	 # Aij(t)*(1-Aij(t-1)) 
			bAtm11mAt_l = 0	 # A(t-1) (1-A(t))
			bAtm1At_l = 0	 # A(t-1) A(t)
			for i in range(T):  
				data_b1mAtm1At[i+1,:,:] = data[i+1,:,:] * (1 - data[i,:,:]) # (1-A(t-1)) A(t) 
				data_bAtm11mAt[i+1,:,:] = (1 - data[i+1,:,:]) * data[i,:,:] # A(t-1) (1-A(t))
				data_bAtm1At[i+1,:,:]   = data[i+1,:,:] * data[i,:,:]       # A(t-1) A(t)
				# calculate (1-Aij(t-1))*Aij(t) 
				sub_nz_and = np.logical_and(data[i+1,:,:]>0,(1-data[i,:,:])>0) 
				b1mAtm1At_l += (((1-data[i,:,:])[sub_nz_and] * data[i+1,:,:][sub_nz_and])).sum() 
				# calculate A(t-1) * (1-A(t))
				sub_nz_and = np.logical_and(data[i,:,:]>0,(1-data[i+1,:,:])>0) 
				bAtm11mAt_l += (((1-data[i+1,:,:])[sub_nz_and] * data[i,:,:][sub_nz_and])).sum() 
				# calculate Aij(t)*Aij(t-1) 
				sub_nz_and = np.logical_and(data[i+1,:,:]>0,data[i,:,:]>0) 
				bAtm1At_l +=  ((data[i+1,:,:][sub_nz_and] * data[i,:,:][sub_nz_and])).sum() 
			self.b1mAtm1At = b1mAtm1At_l
			self.bAtm11mAt = bAtm11mAt_l 	
			self.bAtm1At   = bAtm1At_l   
			 

		self.sum_data_hat = data_b1mAtm1At[1:].sum()  # needed in the update of ell and phi   

		'''
		transposes needed in Q. 
		''' 
		data_b1mAtm1AtT = np.einsum('aij->aji', data_b1mAtm1At[1:]) # needed in the update of Q 	 	 
		data_bAtm11mAtT = np.einsum('aij->aji', data_bAtm11mAt[1:]) # needed in the update of Q	 
		data_bAtm1AtT   = np.einsum('aij->aji', data_bAtm1At[1:]) # needed in the update of Q 

		data_b1mAtm1AtT = np.einsum('aji->ji', data_b1mAtm1AtT) # needed in the update of Q  	 
		data_bAtm11mAtT = np.einsum('aji->ji', data_bAtm11mAtT) # needed in the update of Q	 
		data_bAtm1AtT   = np.einsum('aji->ji', data_bAtm1AtT) # needed in the update of Q 

		# data_b1mAtm1At_All = np.einsum('aij->ij', data_b1mAtm1At) # needed in the update of Q  	
		data_b1mAtm1At = np.einsum('aij->ij', data_b1mAtm1At[1:]) # needed in the update of Q  	 
		data_bAtm11mAt = np.einsum('aij->ij', data_bAtm11mAt[1:]) # needed in the update of Q	 
		data_bAtm1At   = np.einsum('aij->ij', data_bAtm1At[1:]) # needed in the update of Q 

		# data_b1mAtm1At_All = data_b1mAtm1At_All[np.newaxis,:,:]
		data_b1mAtm1At = data_b1mAtm1At[np.newaxis,:,:]
		data_bAtm11mAt = data_bAtm11mAt[np.newaxis,:,:]
		data_bAtm1At   = data_bAtm1At[np.newaxis,:,:]    
 
		data_b1mAtm1AtT = data_b1mAtm1AtT[np.newaxis,:,:]
		data_bAtm11mAtT = data_bAtm11mAtT[np.newaxis,:,:]
		data_bAtm1AtT   = data_bAtm1AtT[np.newaxis,:,:] 

		'''
		values of A(t) (1-A(t-1)), A(t-1) (1-A(t)), and A(t-1) A(t); and their transposes  nedded in Q.  
		''' 
		# data_bAtm11mAtvals   = get_item_array_from_subs(data_bAtm11mAt, data_b1mAtm1At.nonzero())  		
		# data_bAtm1Atvals     = get_item_array_from_subs(data_bAtm1At, data_b1mAtm1At.nonzero())		
		
		# data_b1mAtm1AtT_vals = get_item_array_from_subs(data_b1mAtm1AtT, data_b1mAtm1At.nonzero()) # to calculate denominator containing Aji(t), needed in Q(ij)  
		# data_bAtm11mAtT_vals = get_item_array_from_subs(data_bAtm11mAtT, data_b1mAtm1At.nonzero()) # to calculate denominator containing Aji(t), needed in Q(ij)  
		# data_bAtm1AtT_vals   = get_item_array_from_subs(data_bAtm1AtT, data_b1mAtm1At.nonzero()) # to calculate denominator containing Aji(t), needed in Q(ij)    

		# data0vals    = get_item_array_from_subs(data0, data_b1mAtm1At.nonzero())      
		# data0_T_vals = get_item_array_from_subs(data0_T, data_b1mAtm1At.nonzero()) 

		self.data0vals0    = get_item_array_from_subs(data0, data0.nonzero())      
		self.data0_T_vals0 = get_item_array_from_subs(data0_T, data0.nonzero())  
		'''
		sptensor of A(t) (1-A(t-1)), A(t-1) (1-A(t)), and A(t-1) A(t) nedded in Q. 
		'''  
		
		# data_b1mAtm1At_All = preprocess(data_b1mAtm1At_All)  # to calculate numerator containing Aij(t)*(1-Aij(t-1))  
		data0 = preprocess(data0)   
		data0_T = preprocess(data0_T)   
		data_b1mAtm1At = preprocess(data_b1mAtm1At)  # to calculate numerator containing Aij(t)*(1-Aij(t-1))  
		data_bAtm11mAt = preprocess(data_bAtm11mAt) 
		data_bAtm1At   = preprocess(data_bAtm1At)  

		data_b1mAtm1AtT = preprocess(data_b1mAtm1AtT)  # to calculate numerator containing Aij(t)*(1-Aij(t-1))  
		data_bAtm11mAtT = preprocess(data_bAtm11mAtT) 
		data_bAtm1AtT   = preprocess(data_bAtm1AtT)   

		# data_bAtm11mAtvals_I = data_bAtm11mAt.vals  
		# data_bAtm1Atvals_I   = data_bAtm1At.vals 
		# self.data_bAtm11mAtvals_IS = data_bAtm11mAtvals_I.sum()			
		# self.data_bAtm1Atvals_IS   = data_bAtm1Atvals_I.sum() 

		# self.data_bAtm11mAtsubs_IS = data_bAtm11mAt.subs	 	
		# self.data_bAtm1Atsubs_IS   = data_bAtm1At.subs 

		# save the indexes of the nonzero entries of Aij(t)*(1-Aij(t-1))
		if isinstance(data_b1mAtm1At, skt.dtensor):
			subs_nzp = data_b1mAtm1At.nonzero() 
		elif isinstance(data_b1mAtm1At, skt.sptensor):
			subs_nzp = data_b1mAtm1At.subs   
		 

		# if isinstance(data, skt.dtensor):
		# 	sub_nz_all = data.nonzero() 
		# elif isinstance(data, skt.sptensor):
		# 	sub_nz_all = data.subs    

		# save the indexes of the nonzero entries of  Aij(0)
		if isinstance(data0, skt.dtensor): 
			subs_nz = data0.nonzero() 
		elif isinstance(data0, skt.sptensor): 
			subs_nz = data0.subs 
		

		self.beta_hat = np.ones(T+1)
		self.phi_hat  = np.ones(T+1)
		self.ell_hat  = np.ones(T+1) 
		if T > 0: self.beta_hat[1:] = self.beta0     
		if T > 0: self.phi_hat[1:] = self.phi0     
		if T > 0: self.ell_hat[1:] = self.ell0  

		# self.Q_ij_T = np.ones((T+1, self.N, self.N))   
		# self.Q_ij_dense_T = {}  

		'''
		INFERENCE
		'''
		maxL = -self.inf  # initialization of the maximum log-likelihood
		rng = np.random.RandomState(self.rseed)

		for r in range(self.N_real): 

			self._initialize(rng=rng)   
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

				delta_u, delta_v, delta_w, delta_beta, delta_phi, delta_ell, delta_pibr, delta_mupr = self._update_em(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)
				it, loglik, coincide, convergence = self._check_for_convergence(data_b1mAtm1At,data0,subs_nzp, subs_nz, T, r, it, loglik, coincide, convergence,
																				data_T=data_b1mAtm1AtT, mask=mask)  

				loglik_values.append(loglik)

			if self.verbose:
				print('done!')
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

		return self.u_f, self.v_f, self.w_f, self.u0_f, self.v0_f, self.w0_f, self.beta_f, self.phi_f, self.ell_f, self.pibr_f, self.mupr_f, self.maxL


	def _initialize(self, rng=None):
		"""
			Random initialization of the parameters u, v, w, beta.
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""

		if rng is None:
			rng = np.random.RandomState(self.rseed) 
		if self.T > 1:
			if self.beta0 is not None:
				self.beta = self.beta0
			else:
				self._randomize_beta(rng) 


			if self.phi0 is not None:
				self.phi = self.phi0
			else:
				self._randomize_phi(rng)
			
			
			if self.ell0 is not None:
				self.ell = self.ell0
			else:
				self._randomize_ell(rng)
		else:
			self.beta = 1
			self.beta0 = 1
			self.ell0  = 0
			self.ell  = 0
			if self.flag_anomaly:
				self.phi0  = 1
				self.phi  = 1
			else:
				self.phi0  = 0
				self.phi  = 0
		
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

		if self.initialization == 0:
			if self.verbose:
				print('u, v and w are initialized randomly.') 
			self._randomize_w(rng=rng)
			self._randomize_u_v(rng=rng)

		elif self.initialization == 1:
			if self.verbose:
				print('u, v and w are initialized using the input files:')
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
				print('u, v and w are initialized using the input files:')
				print(self.in_parameters + '.npz')
			theta = np.load(self.in_parameters + '.npz',allow_pickle=True)
			self._randomize_u_v(rng=rng) 
			self._initialize_w(theta['w'])  

	def _initialize_u(self, u0): 
		if u0.shape[0] != self.N:
			raise ValueError('u.shape is different that the initialized one.',self.N,u0.shape[0]) 

		self.u0 = u0.copy()  
		max_entry = np.max(u0)
		self.u0 += max_entry * self.err * np.random.random_sample(self.u0.shape)  
		if self.T > 1.:
			self.u = np.copy(self.u0) 

	def _initialize_v(self, v0):
		if v0.shape[0] != self.N:
			raise ValueError('v.shape is different that the initialized one.',self.N,v0.shape[0]) 
		self.v0 = v0.copy() 
		max_entry = np.max(v0)
		self.v0 += max_entry * self.err  * np.random.random_sample(self.v0.shape)
		if self.T > 1: 
			self.v = np.copy(self.v0)
	
	def _initialize_w(self, w0):
		"""
			Initialize affinity tensor w from file.

			Parameters
			----------
			infile_name : str
						  Path of the input file.
		""" 
		# self.w = w0.copy()  
		# max_entry = np.max(w0)
		# self.w += max_entry * self.err * np.random.random_sample(self.w.shape)
		# if self.assortative:
		# 	self.w = np.diagonal(self.w)
		# 	if self.w.ndim < 2:
		# 		self.w = self.w[np.newaxis,:] 
		# else:
		# 	self.w = self.w[np.newaxis,:,:] 
		
		

		self.w0 = w0.copy()  
		max_entry = np.max(w0)
		self.w0 += max_entry * self.err * np.random.random_sample(self.w0.shape)
		if self.assortative:
			self.w0 = np.diagonal(self.w0)
			if self.w0.ndim < 2:
				self.w0 = self.w0[np.newaxis,:] 
		else:
			self.w0 = self.w0[np.newaxis,:,:]  
		if self.T > 1.:
			self.w = np.copy(self.w0)

	
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
		self.beta = rng.random_sample(1) 

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
		self.phi = rng.random_sample(1) 
	
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
		self.ell = rng.random_sample(1) 
	
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
		# self.pibr = rng.random_sample(1)[0] 
		self.pibr = rng.rand() * 0.5
		# self.pibr = self.pibr0
	
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
		# self.mupr = rng.random_sample(1) [0]
		self.mupr = rng.rand() * 0.5

	def _randomize_w(self, rng):
		"""
			Assign a random number in (0, 1.) to each entry of the affinity tensor w.
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""
		# if self.assortative == True:
		# 	self.w = np.zeros((self.L,self.K))
		# else:
		# 	self.w = np.zeros((self.L,self.K,self.K))

		if rng is None:
			rng = np.random.RandomState(self.rseed) 
		for i in range(self.L):
			for k in range(self.K):
				if self.assortative:
					self.w[i, k] = rng.random_sample(1) 
				else:
					for q in range(k, self.K):
						if q == k: 
							self.w[i, k, q] = rng.random_sample(1)[0]
						else:
							self.w[i, k, q] = self.w[i, q, k] = self.err * rng.random_sample(1)[0]  
		
		# if self.assortative == True:
		# 	self.w0 = np.zeros((self.L,self.K))
		# else:
		# 	self.w0 = np.zeros((self.L,self.K,self.K))

		if rng is None:
			rng = np.random.RandomState(self.rseed) 
		for i in range(self.L):
			for k in range(self.K):
				if self.assortative:
					self.w0[i, k] = rng.random_sample(1) 
				else:
					for q in range(k, self.K):
						if q == k: 
							self.w0[i, k, q] = rng.random_sample(1)[0]
						else:
							self.w0[i, k, q] = self.w0[i, q, k] = self.err * rng.random_sample(1)[0] 


	def _randomize_u_v(self, rng=None):
		"""
			Assign a random number in (0, 1.) to each entry of the membership matrices u and v, and normalize each row.
			Parameters
			----------
			rng : RandomState
				  Container for the Mersenne Twister pseudo-random number generator.
		"""
		if self.T > 1.:
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
		

		if rng is None:
			rng = np.random.RandomState(self.rseed)
		self.u0 = rng.random_sample((self.N,self.K))
		row_sums = self.u0.sum(axis=1) 
		self.u0[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

		if not self.undirected:
			self.v0 = rng.random_sample((self.N,self.K))
			row_sums = self.v0.sum(axis=1)
			self.v0[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		else:
			self.v0 = self.u0


	def _update_old_variables(self):
		"""
			Update values of the parameters in the previous iteration.
		"""
		
		# self.u_old[self.u > 0] = np.copy(self.u[self.u > 0])
		# self.v_old[self.v > 0] = np.copy(self.v[self.v > 0])
		# self.w_old[self.w > 0] = np.copy(self.w[self.w > 0]) 
		self.u_old = np.copy(self.u)
		self.v_old = np.copy(self.v)
		self.w_old = np.copy(self.w) 
		self.u0_old = np.copy(self.u0)
		self.v0_old = np.copy(self.v0)
		self.w0_old = np.copy(self.w0) 
		self.pibr_old = np.copy(self.pibr)
		self.mupr_old = np.copy(self.mupr)
		self.beta_old = np.copy(self.beta) 
		self.phi_old = np.copy(self.phi)  
		self.ell_old = np.copy(self.ell) 

	def _update_cache(self,data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,update_Q=True):
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
		self.lambda0_nz   = self._lambda0_nz(subs_nzp, self.u , self.v , self.w)
		self.lambda0_nz_0 = self._lambda0_nz(subs_nz, self.u0 , self.v0 , self.w0)    

		if self.assortative == False:
			self.lambda0_nzT = self._lambda0_nz(subs_nzp, self.v, self.u, np.einsum('akq->aqk',self.w))
			self.lambda0_nzT_0 = self._lambda0_nz(subs_nz, self.v0, self.u0, np.einsum('akq->aqk',self.w0))  
		else:
			self.lambda0_nzT = self._lambda0_nz(subs_nzp, self.v, self.u,self.w) 
			self.lambda0_nzT_0 = self._lambda0_nz(subs_nz, self.v0, self.u0,self.w0)
		 
		   
		self.M_nz = self.lambda0_nz   
		self.M_nz[self.M_nz == 0] = 1 

		self.M_nz_0 = self.lambda0_nz_0   
		self.M_nz_0[self.M_nz_0 == 0] = 1 

		if self.flag_anomaly == True: 
			self.Qij_dense,self.Qij_nz = self._QIJ(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)   
		
		# if isinstance(data, skt.dtensor):
		# 	self.data_M_nz = data[subs_nzp] / self.M_nz 
		# elif isinstance(data, skt.sptensor):
		# 	self.data_M_nz = data.vals / self.M_nz   
		
		#TODO: check 
		if isinstance(data_b1mAtm1At, skt.dtensor):
			if self.flag_anomaly == True:
				self.data_M_nz_Q  = (data_b1mAtm1At[subs_nzp] * (1-self.Qij_nz)[subs_nzp])/ self.M_nz   
			else:
				self.data_M_nz_Q = data_b1mAtm1At[subs_nzp] / self.M_nz
		 
		elif isinstance(data_b1mAtm1At, skt.sptensor):
			if self.flag_anomaly == True:
				self.data_M_nz_Q = data_b1mAtm1At.vals * (1-self.Qij_nz) / self.M_nz   
			else:
				self.data_M_nz_Q = data_b1mAtm1At.vals / self.M_nz 

		
		if isinstance(data0, skt.dtensor):
			if self.flag_anomaly == True:
				self.data_M_nz_Q_0  = (data0[subs_nz] * (1-self.Q_ij_nz_0)[subs_nz])/ self.M_nz_0   
			else:
				self.data_M_nz_Q_0 = data0[subs_nz] / self.M_nz_0 
		elif isinstance(data0, skt.sptensor):
			if self.flag_anomaly == True:
				self.data_M_nz_Q_0 = data0.vals * (1-self.Q_ij_nz_0) / self.M_nz_0   
			else:
				self.data_M_nz_Q_0 = data0.vals / self.M_nz_0 
	
	def _QIJ(self, data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,EPS=1e-12):
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
		lambda0_ija_0  = self._lambda0_full(self.u0, self.v0, self.w0) 
		lambda0_ijaT_0 = self._lambda0_full(self.v0, self.u0, self.w0)   
		 

		dense_data0 = data0.toarray() 
		dense_data0T = data0_T.toarray() 
		# lambda0_ijaT_0 = transpose_tensor(lambda0_ija_0) 

		"""
			Compute Q_ij_dense for t=0, need to update pi and mu
		"""      


		# Q_ij_dense_0   = np.ones(lambda0_ija_0.shape)
		# Q_ij_dense_0  *=  self.mupr * poisson.pmf(dense_data0, self.pibr) * poisson.pmf(dense_data0T, self.pibr)
		# Q_ij_dense_d_0 = Q_ij_dense_0 + (1-self.mupr) * poisson.pmf(dense_data0, lambda0_ija_0) * poisson.pmf(dense_data0T, lambda0_ijaT_0)
		# non_zeros = Q_ij_dense_d_0 > 0
		# Q_ij_dense_0[non_zeros] /= Q_ij_dense_d_0[non_zeros]  


		nz_recon_I_0 =  self.mupr * poisson.pmf(data0.vals, self.pibr) * poisson.pmf(self.data0_T_vals0, self.pibr)  
		nz_recon_Id_0 = nz_recon_I_0 + (1-self.mupr) * poisson.pmf(data0.vals, self.lambda0_nz_0) * poisson.pmf(self.data0_T_vals0, self.lambda0_nzT_0)  
		

		# nz_recon_I_0 =  self.mupr * np.power(self.data0vals0, self.pibr) * np.power(self.data0_T_vals0, self.pibr) * np.exp(-self.pibr*2)   
		# nz_recon_Id_0 = nz_recon_I_0 + (1-self.mupr) * np.power(self.data0vals0, self.lambda0_nz_0) * np.power(self.data0_T_vals0, self.lambda0_nzT_0) * np.exp(-(self.lambda0_nz_0+self.lambda0_nzT_0))   


		non_zeros = nz_recon_Id_0 > 0 
		nz_recon_I_0[non_zeros] /=  nz_recon_Id_0[non_zeros]  
		Q_ij_dense_0 = np.ones(lambda0_ija_0.shape)   

		Q_ij_dense_0 *=  self.mupr * np.exp(-self.pibr*2)  
		Q_ij_dense_d_0 = Q_ij_dense_0 + (1-self.mupr) * np.exp(-(lambda0_ija_0+lambda0_ijaT_0))  
		non_zeros = Q_ij_dense_d_0 > 0
		Q_ij_dense_0[non_zeros] /= Q_ij_dense_d_0[non_zeros]  
			
		Q_ij_dense_0[subs_nz] = nz_recon_I_0  

		Q_ij_dense_0 = np.maximum(Q_ij_dense_0[0], transpose_tensor(Q_ij_dense_0)) # make it symmetric
		assert np.allclose(Q_ij_dense_0[0], Q_ij_dense_0[0].T, rtol=1e-05, atol=1e-08) 
		np.fill_diagonal(Q_ij_dense_0[0], 0.) 

		# self.Q0A0  = (Q_ij_dense_0 * dense_data0).sum()	
		self.Q0A0  = (Q_ij_dense_0[subs_nz] * self.data0vals0).sum()	
		self.Q0sum = (Q_ij_dense_0).sum() 
		self.b1mQ0sum = (1-Q_ij_dense_0[0]).sum() 
		assert (Q_ij_dense_0 > 1).sum() == 0  
		self.Q_ij_dense_0 = np.copy(Q_ij_dense_0)  
		self.Q_ij_nz_0 = Q_ij_dense_0[subs_nz]  


		if self.T > 1.:
			lambda0_ija   = self._lambda0_full(self.u, self.v, self.w)  
			lambda0_ijaT = transpose_tensor(lambda0_ija) 

			"""
				Compute Q_ij_dense at zeros of (1-Aij(t-1)) * Aij(t) by dense Aij(t-1) * (1-Aij(t)) and Aij(t-1) * Aij(t) 
			"""    
			# dense_data_b1mAtm1At_All = data_b1mAtm1At_All.toarray() 
			dense_data_b1mAtm1At = data_b1mAtm1At.toarray() 
			dense_data_bAtm11mAt = data_bAtm11mAt.toarray() 
			dense_data_bAtm1At   = data_bAtm1At.toarray() 


			data_b1mAtm1AtT = data_b1mAtm1AtT.toarray()  # to calculate numerator containing Aij(t)*(1-Aij(t-1))  
			data_bAtm11mAtT = data_bAtm11mAtT.toarray() 
			data_bAtm1AtT   = data_bAtm1AtT.toarray()   

			# Q_ij_dense  = np.ones(dense_data_b1mAtm1At.shape)    

			# Q_ij_dense  =  self.mupr * poisson.pmf(dense_data0, self.pibr) * poisson.pmf(dense_data0T, self.pibr)  *  np.exp(-2.*self.T*self.ell*self.phi_hat[self.T])

			# if self.T > 1:
			# 	log_Q_ij_dense = ((dense_data_bAtm11mAt+data_bAtm11mAtT)*np.log(self.phi+EPS)+(dense_data_bAtm1At+data_bAtm1AtT)*np.log(1-self.phi+EPS)+ (dense_data_b1mAtm1At+data_b1mAtm1AtT)*np.log((self.ell*self.phi)+EPS))
			# 	result = np.exp(log_Q_ij_dense)  
			# 	Q_ij_dense *= result
			
			# Q_ij_dense_d = (1-self.mupr) * poisson.pmf(dense_data0, lambda0_ija_0) * poisson.pmf(dense_data0T, lambda0_ijaT_0)  * np.exp(- self.T * self.beta_hat[self.T]* (lambda0_ija+lambda0_ijaT) )  

			# if self.T > 1:
			# 	log_Q_ij_dense_d = np.log(self.beta_hat[self.T]+EPS)*(dense_data_bAtm11mAt+data_bAtm11mAtT)+np.log(1-self.beta_hat[self.T]+EPS)*(dense_data_bAtm1At+data_bAtm1AtT)+np.log((lambda0_ija*self.beta_hat[self.T])+EPS)*dense_data_b1mAtm1At+np.log((lambda0_ijaT*self.beta_hat[self.T])+EPS)*data_b1mAtm1AtT
			# 	result = np.exp(log_Q_ij_dense_d) 
			# 	Q_ij_dense_d *= result 


			# Q_ij_dense  =  self.mupr * poisson.pmf(dense_data0, self.pibr) * poisson.pmf(dense_data0T, self.pibr)  
			Q_ij_dense  =  self.mupr * np.power(dense_data0, self.pibr) * np.power(dense_data0T, self.pibr) * np.exp(-self.pibr*2)
			Q_ij_dense  *= np.exp(-2.*self.T*self.ell_hat[self.T]*self.phi_hat[self.T])
			Q_ij_dense  *= np.power(self.phi_hat[self.T],dense_data_bAtm11mAt) * np.power(1-self.phi_hat[self.T],dense_data_bAtm1At)
			Q_ij_dense  *= np.power(self.phi_hat[self.T],data_bAtm11mAtT) * np.power(1-self.phi_hat[self.T],data_bAtm1AtT)
			Q_ij_dense  *= np.power(self.ell_hat[self.T]*self.phi_hat[self.T],dense_data_b1mAtm1At)* np.power(self.ell_hat[self.T]*self.phi,data_b1mAtm1AtT)
			
			# Q_ij_dense_d = (1-self.mupr) * poisson.pmf(dense_data0, lambda0_ija_0) * poisson.pmf(dense_data0T, lambda0_ijaT_0)  
			Q_ij_dense_d = (1-self.mupr) * np.power(dense_data0, lambda0_ija_0) * np.power(dense_data0T, lambda0_ijaT_0) * np.exp(-(lambda0_ija_0+lambda0_ijaT_0)) 
 
			Q_ij_dense_d *= np.exp(- self.T * self.beta_hat[self.T]* (lambda0_ija+lambda0_ijaT) ) 
			Q_ij_dense_d *= np.power(self.beta_hat[self.T],dense_data_bAtm11mAt) * np.power(1-self.beta_hat[self.T],dense_data_bAtm1At) 
			Q_ij_dense_d *= np.power(self.beta_hat[self.T],data_bAtm11mAtT) * np.power(1-self.beta_hat[self.T],data_bAtm1AtT)
			Q_ij_dense_d *= np.power(lambda0_ija*self.beta_hat[self.T],dense_data_b1mAtm1At)* np.power(lambda0_ijaT*self.beta_hat[self.T],data_b1mAtm1AtT) 


			Q_ij_dense_d += Q_ij_dense 
			non_zeros = Q_ij_dense_d > 0
			Q_ij_dense[non_zeros] /= Q_ij_dense_d[non_zeros]   
			Q_ij_dense = np.maximum( Q_ij_dense, transpose_tensor(Q_ij_dense)) # make it symmetric
			np.fill_diagonal(Q_ij_dense[0], 0.)  
			
			assert np.allclose(Q_ij_dense[0], Q_ij_dense[0].T, rtol=1e-05, atol=1e-08)   
			self.b1mQsum = (1-Q_ij_dense).sum()

			# self.Qb1mAtm1At_All     = np.einsum('aij, aij-> ', Q_ij_dense, dense_data_b1mAtm1At_All)   # needed in the update of Likelihood 
			self.QAtm1At     = np.einsum('aij, aij-> ', Q_ij_dense, dense_data_bAtm1At)   # needed in the update of Likelihood 	
			self.QAtm11mAt   = np.einsum('aij, aij-> ', Q_ij_dense, dense_data_bAtm11mAt) # needed in the update of Likelihood 
			self.Q1mAtm1At   = np.einsum('aij, aij-> ', Q_ij_dense, dense_data_b1mAtm1At) # needed in the update of pi 
			
			self.b1mQAtm1At   = np.einsum('aij, aij-> ', (1-Q_ij_dense), dense_data_bAtm1At)   # needed in the update of beta, Likelihood
			self.b1mQAtm11mAt = np.einsum('aij, aij-> ', (1-Q_ij_dense), dense_data_bAtm11mAt) # needed in the update of beta 
			self.b1mQ1mAtm1At = np.einsum('aij, aij-> ', (1-Q_ij_dense), dense_data_b1mAtm1At) # needed in the update of beta 
			self.Qsum = Q_ij_dense.sum() # needed in the update of mu,pi, and ell 
		else:
			Q_ij_dense = np.copy(Q_ij_dense_0) 
			Q_ij_dense[subs_nz] = nz_recon_I_0
		
		
		return Q_ij_dense, Q_ij_dense[subs_nzp]


	# @gl.timeit('lambda0_nz')
	# TO DO: check w dimension
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
			nz_recon_IQ = np.einsum('Ik,kq->Iq', u[subs_nz[1], :], w[0, :, :]) 
		else: 
			nz_recon_IQ = np.einsum('Ik,k->Ik', u[subs_nz[1], :], w[0, :])   
		nz_recon_I = np.einsum('Iq,Iq->I', nz_recon_IQ, v[subs_nz[2], :]) 

		return nz_recon_I

	@gl.timeit('update_em')
	def _update_em(self,data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz,mask=None,subs_nz_mask=None):
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
			d_eta : float
					Maximum distance between the old and the new reciprocity coefficient eta.
		""" 

		self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)

		# gl.timer.cum('uvw')
		if self.fix_communities == False:
			d_u = self._update_U(subs_nzp,subs_nz,mask=None,subs_nz_mask=None) 
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz) 
			if self.undirected:
				self.v = self.u
				self.v_old = self.v
				d_v = d_u
			else: 
				d_v = self._update_V(subs_nzp,subs_nz,mask=None,subs_nz_mask=None)  
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz) 
		else:
			d_u = 0
			d_v = 0

		if self.fix_w == False:
			if not self.assortative:
				d_w = self._update_W(subs_nzp,mask=None,subs_nz_mask=None)
			else:
				d_w = self._update_W_assortative(subs_nzp,subs_nz,mask=None,subs_nz_mask=None)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)
		else:
			d_w = 0

		if self.fix_beta == False:
			if self.T  > 1: 
				d_beta = self._update_beta() 
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)
			else:  d_beta = 0. 
		else:  
			d_beta = 0. 
		
		#To DO: correct hte following.
		if self.fix_phi == False: 
			if self.T > 1:
				d_phi = self._update_phi()
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)
			else:
				d_phi = 0.
		else:
			d_phi = 0.

		if self.fix_ell == False:
			if self.T > 1:
				# denominator = (data_T_vals * self.beta_hat[subs_nzp[0]]).sum()  
				d_ell = self._update_ell(data_b1mAtm1At, subs_nzp,mask=mask,subs_nz_mask=subs_nz_mask)
				self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)
			else:
				d_ell = 0.
		else:
			d_ell = 0.
		


		if self.fix_pibr == False: 
			# s = time.time() 
			d_pibr = self._update_pibr( data0, subs_nz,mask=mask,subs_nz_mask=subs_nz_mask) 
			# e = time.time()
			# print('pi',e-s)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)

		else: 
			d_pibr = 0.
		
		if self.fix_mupr == False:
			# s = time.time()
			d_mupr = self._update_mupr(data0, subs_nz,mask=mask,subs_nz_mask=subs_nz_mask)
			# e = time.time()
			# print('mu',e-s)
			self._update_cache(data_b1mAtm1At,data_bAtm11mAt,data_bAtm1At,data0,data_b1mAtm1AtT,data_bAtm11mAtT,data_bAtm1AtT,data0_T,subs_nzp,subs_nz)
		else:
			d_mupr = 0.


		return d_u, d_v, d_w, d_beta, d_phi, d_ell, d_pibr, d_mupr


	# @gl.timeit('pibr')
	def _update_pibr(self,  data, subs_nz,mask=None,subs_nz_mask=None):
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
			# Adata = (data.vals *self.Q_ij_nz_0).sum() 
			Adata = self.Q0A0
			# Adata   = (data.vals * self.Qij_dense[subs_nz]).sum()  
		
		if mask is None:     
			self.pibr = Adata / self.Q0sum 
			# self.pibr = Adata / self.Qij_dense[subs_nz].sum() 
		else:
			self.pibr = Adata / self.Q_ij_dense_0[subs_nz_mask].sum()
		self.ell_hat[0] = self.pibr  
		
		if (self.pibr < 0) or (self.pibr > 1): 
			print('self.pibr:', self.pibr)
			raise ValueError('The anomaly coefficient pi has to be in [0, 1]!')
		dist_pibr = self.pibr - self.pibr_old
		self.pibr_old = np.copy(self.pibr) 

		return dist_pibr

	# @gl.timeit('mupr')
	def _update_mupr(self, data, subs_nz,mask=None,subs_nz_mask=None):
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
		if mask is None:
			self.mupr = self.Q0sum / ( (self.N * (self.N-1)) / 1. ) 
			# self.mupr = self.Q0sum / (self.N * (self.N-1)) 
		else:
			self.mupr = self.Q_ij_dense_0[subs_nz_mask].sum() /( (self.N * (self.N-1))/1. )  
		
		if (self.mupr < 0) or (self.mupr > 1):
			raise ValueError('The reciprocity coefficient mu has to be in [0, 1]!')

		dist_mupr = self.mupr - self.mupr_old 
		self.mupr_old = np.copy(self.mupr)  

		return dist_mupr 

	# @gl.timeit('ell')
	def _update_ell(self, data, subs_nz,mask=None,subs_nz_mask=None):
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
	
		if isinstance(data, skt.dtensor):
			Adata = (data[subs_nz] * self.Qij_nz).sum() 
		if mask is None:    
			# self.ell =  self.Qb1mAtm1At_All  / ( self.Q0sum + self.T * self.phi_hat[self.T] * self.Qsum)  
			self.ell =  self.Q1mAtm1At  / ( (self.T * self.phi_hat[-1]) * self.Qsum)  
			# self.ell =  self.Q1mAtm1At  / ( (self.T * self.phi_hat[self.T]) * self.Qsum) 
		# else:
			# self.ell = Adata / ((1+self.T*self.phi_hat[self.T])*self.Qij_dense[subs_nz_mask]).sum() 

		self.ell_hat[1:] = self.ell      
		self.ell_hat[0] = self.pibr   
		
		if (self.ell < 0) or (self.ell > 1): 
			print('self.ell:', self.ell)
			raise ValueError('The anomaly coefficient ell has to be in [0, 1]!')
		dist_ell = self.ell - self.ell_old
		self.ell_old = np.copy(self.ell)

		return dist_ell 
	

	#TO DO: check the Eq==0
	# @gl.timeit('update_phi')
	def _update_phi(self):
		# phi_result = root(func_phi_static, self.phi_old, args=(self))  
		# self.phi = phi_result.x[0] 
		self.phi = brentq(func_phi_static, a=0.000001,b=0.999, args=(self)) 
		self.phi_hat[1:] = self.phi 
		self.phi_hat[0]  = 0 

		dist_phi = abs(self.phi - self.phi_old) 
		self.phi_old = np.copy(self.phi)

		return dist_phi
	
	# @gl.timeit('update_beta')
	def _update_beta(self):
		res = root(func_beta_static, self.beta_old, args=(self)) 
		self.beta = res.x[0] 
		self.beta_hat[1:] = self.beta 
		self.beta_hat[0]  = 1  
		

		dist_beta = abs(self.beta - self.beta_old) 
		self.beta_old = np.copy(self.beta)

		return dist_beta
	
	# @gl.timeit_cum('update_U')
	def _update_U(self,subs_nzp,subs_nz,mask=None,subs_nz_mask=None):
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
			# self.T == 0: 
			if mask is None:
				Du = np.einsum('aij,jq->iq', 1-self.Q_ij_dense_0,self.v0) 
			else:
				Du = np.einsum('aij,jq->iq', mask * (1-self.Q_ij_dense_0),self.v0) 
			
			if not self.constrained:  
				# Du = np.einsum('iq->q', Du)
				if not self.assortative:
					w_k = np.einsum('akq->kq', self.w0)
					Z_uk = np.einsum('aiq,kq->ik', Du, w_k)
				else:
					w_k = np.einsum('ak->k', self.w0) 
					# w_k = np.einsum('a,ak->k', self.beta_hat, self.w)
					Z_uk = np.einsum('ik,k->ik', Du, w_k)   
				Z_uk += self.bg    
				self.u0 = (self.ag - 1) + self.u0_old * self._update_membership_Q_0(subs_nz, self.u0, self.v0, self.w0, 1) 
				non_zeros = Z_uk > 0.  
				self.u0[Z_uk == 0] = 0.  
				self.u0[non_zeros] /= Z_uk[non_zeros]
				# gl.timer.cum('update_mem')
			else:
				# Du = np.einsum('iq->q', Du)
				if not self.assortative:
					w_k = np.einsum('akq->kq', self.w0)
					Z_uk = np.einsum('q,kq->k', Du, w_k)
				else:
					w_k = np.einsum('ak->k', self.w0)
					Z_uk = np.einsum('k,k->k', Du, w_k) 
				for i in range(self.u0.shape[0]):
					if self.u0[i].sum() > self.err_max: 
						u_root = root(u_with_lagrange_multiplier, self.u0_old[i], args=(self.u[i],Z_uk))
						self.u0[i] = u_root.x
			
			if self.T > 1: 
				if mask is None:
					Du = np.einsum('aij,jq->iq', 1-self.Qij_dense,self.v)
				else:
					Du = np.einsum('aij,jq->iq', mask * (1-self.Qij_dense),self.v) 
				
				if not self.constrained:  
					# Du = np.einsum('iq->q', Du)
					if not self.assortative:
						w_k = np.einsum('akq->kq', self.w)
						Z_uk = np.einsum('iq,kq->ik', Du, w_k)
					else:
						w_k = np.einsum('ak->k', self.w) 
						# w_k = np.einsum('a,ak->k', self.beta_hat, self.w)
						Z_uk = np.einsum('ik,k->ik', Du, w_k)  
					Z_uk *= (self.beta_hat[self.T] * self.T) 
					Z_uk += self.bg   
					self.u = (self.ag - 1) + self.u_old * self._update_membership_Q(subs_nzp, self.u, self.v, self.w, 1)
					non_zeros = Z_uk > 0. 
					self.u[Z_uk == 0] = 0.  
					self.u[non_zeros] /= Z_uk[non_zeros]
					# gl.timer.cum('update_mem')
				else:
					# Du = np.einsum('iq->q', Du)
					if not self.assortative:
						w_k = np.einsum('akq->kq', self.w)
						Z_uk = np.einsum('q,kq->k', Du, w_k)
					else:
						w_k = np.einsum('ak->k', self.w)
						Z_uk = np.einsum('k,k->k', Du, w_k)
					Z_uk *= (self.beta_hat[self.T] * self.T)
					for i in range(self.u.shape[0]):
						if self.u[i].sum() > self.err_max:
							u_root = root(u_with_lagrange_multiplier, self.u_old[i], args=(self.u[i],Z_uk))
							self.u[i] = u_root.x 
				
			
			# row_sums = self.u.sum(axis=1)
			# self.u[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		else: #self.flag_anomaly == False 
			#self.T == 0 
			self.u0 = (self.ag - 1) + self.u0_old * ( self._update_membership_0(subs_nz, self.u0, self.v0, self.w0, 1) )  
			
			if not self.constrained:
				Du = np.einsum('iq->q', self.v0)
				if not self.assortative:
					w_k = np.einsum('akq->kq', self.w0)
					Z_uk = np.einsum('q,kq->k', Du, w_k)
				else:
					w_k = np.einsum('ak->k', self.w0)
					Z_uk = np.einsum('k,k->k', Du, w_k) 
				non_zeros = Z_uk > 0. 
				self.u0[:, Z_uk == 0] = 0.
				self.u0[:, non_zeros] /= (self.bg+Z_uk[np.newaxis,non_zeros])  
				# gl.timer.cum('update_mem')
			else:
				Du = np.einsum('iq->q', self.v0)
				if not self.assortative:
					w_k = np.einsum('akq->kq', self.w0)
					Z_uk = np.einsum('q,kq->k', Du, w_k)
				else:
					w_k = np.einsum('ak->k', self.w0)
					Z_uk = np.einsum('k,k->k', Du, w_k) 
				for i in range(self.u0.shape[0]):
					if self.u0[i].sum() > self.err_max:
						u_root = root(u_with_lagrange_multiplier, self.u0_old[i], args=(self.u0[i],Z_uk))
						self.u0[i] = u_root.x
			if self.T > 1: 
				self.u = (self.ag - 1) + self.u_old * ( + self._update_membership(subs_nzp, self.u, self.v, self.w, 1) )  
				if not self.constrained:
					Du = np.einsum('iq->q', self.v)
					if not self.assortative:
						w_k = np.einsum('akq->kq', self.w)
						Z_uk = np.einsum('q,kq->k', Du, w_k)
					else:
						w_k = np.einsum('ak->k', self.w)
						Z_uk = np.einsum('k,k->k', Du, w_k) 
					Z_uk *= (self.beta_hat[self.T] * self.T)
					non_zeros = Z_uk > 0.   
					self.u[:, Z_uk == 0] = 0.     
					self.u[:, non_zeros] /= (self.bg+Z_uk[np.newaxis ,non_zeros])  
					# gl.timer.cum('update_mem')
				else:
					Du = np.einsum('iq->q', self.v)
					if not self.assortative:
						w_k = np.einsum('akq->kq', self.w)
						Z_uk = np.einsum('q,kq->k', Du, w_k)
					else:
						w_k = np.einsum('ak->k', self.w)
						Z_uk = np.einsum('k,k->k', Du, w_k)
					Z_uk *= (self.beta_hat[self.T] * self.T)
					for i in range(self.u.shape[0]):
						if self.u[i].sum() > self.err_max:
							u_root = root(u_with_lagrange_multiplier, self.u_old[i], args=(self.u[i],Z_uk))
							self.u[i] = u_root.x
				

		

		low_values_indices = self.u0 < self.err_max  # values are too low
		self.u0[low_values_indices] = 0.  # and set to 0.
		self.u0_old = np.copy(self.u0)  
 
		if self.T > 1:
			low_values_indices = self.u < self.err_max  # values are too low
			self.u[low_values_indices] = 0.  # and set to 0.

			dist_u = np.amax(abs(self.u - self.u_old))  
			self.u_old = np.copy(self.u)
		else:
			dist_u = np.amax(abs(self.u0 - self.u0_old)) 
			self.u = np.copy(self.u0) 
			self.u_old = np.copy(self.u0)

		return dist_u

	# @gl.timeit_cum('update_V')
	def _update_V(self,subs_nzp,subs_nz,mask=None,subs_nz_mask=None):
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
			# self.T == 0
			if mask is None:
				Dv = np.einsum('aji,ik->ajk', 1-self.Q_ij_dense_0, self.u0)
			else:
				Dv = np.einsum('aji,ik->ajk', mask * (1-self.Q_ij_dense_0), self.u0) 
			
			if not self.constrained: 
				if not self.assortative:
					# w_k = np.einsum('aqk->qk', self.w0)
					Z_vk = np.einsum('aik,aqk->ik', Dv, self.w0)
				else:
					# w_k = np.einsum('ak->k', self.w0)
					# w_k = np.einsum('a,ak->k', self.beta_hat, self.w)
					Z_vk = np.einsum('aik,ak->ik', Dv, self.w0) 
				Z_vk += self.bg
				self.v0 = (self.ag - 1) + self.v0_old * self._update_membership_Q_0(subs_nz, self.u0, self.v0, self.w0, 2)
				non_zeros = Z_vk > 0
				self.v0[Z_vk == 0] = 0.
				self.v0[non_zeros] /=(Z_vk[non_zeros])
			else:
				# Dv = np.einsum('iq->q', Dv)
				if not self.assortative:
					# w_k = np.einsum('aqk->qk', self.w0)
					Z_vk = np.einsum('ak,aqk->k', Dv, self.w0)
				else:
					# w_k = np.einsum('ak->k', self.w0)
					Z_vk = np.einsum('ak,ak->k', Dv, self.w0) 

				for i in range(self.v0.shape[0]):
					if self.v0[i].sum() > self.err_max:
						v_root = root(u_with_lagrange_multiplier, self.v0_old[i], args=(self.v0[i],Z_vk))
						self.v0[i] = v_root.x
			
			if self.T > 1: 
				if mask is None:
					Dv = np.einsum('aji,ik->ajk', 1-self.Qij_dense, self.u)
				else:
					Dv = np.einsum('aji,ik->ajk', mask * (1-self.Qij_dense), self.u) 
				
				if not self.constrained: 
					if not self.assortative:
						# w_k = np.einsum('aqk->qk', self.w)
						Z_vk = np.einsum('aik,aqk->ik', Dv, self.w)
					else:
						# w_k = np.einsum('ak->k', self.w)
						# w_k = np.einsum('a,ak->k', self.beta_hat, self.w)
						Z_vk = np.einsum('aik,ak->ik', Dv, self.w)
					Z_vk *= (self.beta_hat[self.T] * self.T )
					Z_vk += self.bg
					self.v = (self.ag - 1) + self.v_old * self._update_membership_Q(subs_nzp, self.u, self.v, self.w, 2) 					
					non_zeros = Z_vk > 0
					self.v[Z_vk == 0] = 0.
					self.v[non_zeros] /=(Z_vk[non_zeros])
				else:
					# Dv = np.einsum('iq->q', Dv)
					if not self.assortative:
						# w_k = np.einsum('aqk->qk', self.w)
						Z_vk = np.einsum('ak,aqk->k', Dv, self.w)
					else:
						# w_k = np.einsum('ak->k', self.w)
						Z_vk = np.einsum('ak,ak->k', Dv, self.w)
					Z_vk *= (self.beta_hat[self.T] * self.T )

					for i in range(self.v.shape[0]):
						if self.v[i].sum() > self.err_max:
							v_root = root(u_with_lagrange_multiplier, self.v_old[i], args=(self.v[i],Z_vk))
							self.v[i] = v_root.x
		else: 
			# self.T == 0
			self.v0 = (self.ag - 1) + self.v0_old * self._update_membership_0(subs_nz, self.u0, self.v0, self.w0, 2) 
			if not self.constrained:
				Dv = np.einsum('iq->q', self.u0)
				if not self.assortative:
					w_k = np.einsum('aqk->qk', self.w0)
					Z_vk = np.einsum('q,qk->k', Dv, w_k)
				else:
					w_k = np.einsum('ak->k', self.w0)
					Z_vk = np.einsum('k,k->k', Dv, w_k) 
				non_zeros = Z_vk > 0
				self.v0[:, Z_vk == 0] = 0.
				self.v0[:, non_zeros] /=(self.bg+Z_vk[np.newaxis, non_zeros])
			else:
				Dv = np.einsum('iq->q', self.u0)
				if not self.assortative:
					w_k = np.einsum('aqk->qk', self.w0[np.newaxis,:,:])
					Z_vk = np.einsum('q,qk->k', Dv, w_k)
				else:
					w_k = np.einsum('ak->k', self.w0[np.newaxis,:])
					Z_vk = np.einsum('k,k->k', Dv, w_k) 

				for i in range(self.v.shape[0]):
					if self.v0[i].sum() > self.err_max:
						v_root = root(u_with_lagrange_multiplier, self.v0_old[i], args=(self.v0[i],Z_vk))
						self.v0[i] = v_root.x
				# row_sums = self.v.sum(axis=1)
				# self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
			
			if self.T > 1: 
				self.v = (self.ag - 1) + self.v_old * self._update_membership(subs_nzp, self.u, self.v, self.w, 2) 
				if not self.constrained:
					Dv = np.einsum('iq->q', self.u)
					if not self.assortative:
						w_k = np.einsum('aqk->qk', self.w)
						Z_vk = np.einsum('q,qk->k', Dv, w_k)
					else:
						w_k = np.einsum('ak->k', self.w)
						Z_vk = np.einsum('k,k->k', Dv, w_k)
					Z_vk *= (self.beta_hat[self.T] * self.T )
					non_zeros = Z_vk > 0
					self.v[:, Z_vk == 0] = 0.
					self.v[:, non_zeros] /=(self.bg+Z_vk[np.newaxis, non_zeros])
				else:
					Dv = np.einsum('iq->q', self.u)
					if not self.assortative:
						w_k = np.einsum('aqk->qk', self.w)
						Z_vk = np.einsum('q,qk->k', Dv, w_k)
					else:
						w_k = np.einsum('ak->k', self.w)
						Z_vk = np.einsum('k,k->k', Dv, w_k)
					Z_vk *= (self.beta_hat[self.T] * self.T )

					for i in range(self.v.shape[0]):
						if self.v[i].sum() > self.err_max:
							v_root = root(u_with_lagrange_multiplier, self.v_old[i], args=(self.v[i],Z_vk))
							self.v[i] = v_root.x
					# row_sums = self.v.sum(axis=1)
					# self.v[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
		

		low_values_indices = self.v0 < self.err_max  # values are too low
		self.v0[low_values_indices] = 0.  # and set to 0.

		
		self.v0_old = np.copy(self.v0) 

		if self.T > 1:
			low_values_indices = self.v < self.err_max  # values are too low
			self.v[low_values_indices] = 0.  # and set to 0.
			dist_v = np.amax(abs(self.v - self.v_old))
			self.v_old = np.copy(self.v)  
		else:
			self.v = np.copy(self.v0) 
			self.v_old = np.copy(self.v0) 
			dist_v = np.amax(abs(self.v - self.v_old))

		# print('self.v:', np.mean(self.v))

		return dist_v

	# @gl.timeit_cum('update_W')
	def _update_W(self, subs_nz,mask=None,subs_nz_mask=None):
		"""
			Update affinity tensor.
			Parameters
			----------
			subs_nz : tuple
					  Indices of elements of data that are non-zero.
			Returns
			-------
			dist_w : float
					 Maximum distance between the old and the new affinity tensor w.
		"""
		# TO DO: add anomaly effects to this part. 
		sub_w_nz = self.w.nonzero()
		uttkrp_DKQ = np.zeros_like(self.w) 

		UV = np.einsum('Ik,Iq->Ikq', self.u[subs_nz[1], :], self.v[ subs_nz[2], :])  
		uttkrp_I = self.data_M_nz_Q[:, np.newaxis, np.newaxis] * UV
		# uttkrp_I = self.data_M_nz[:, np.newaxis, np.newaxis] * UV
		
		for a, k, q in zip(*sub_w_nz):
			uttkrp_DKQ[:, k, q] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k, q], minlength=1)[0] 
		
		self.w = (self.ag - 1) + self.w * uttkrp_DKQ

		if self.flag_anomaly == True:
			if mask is None:
				UQk = np.einsum('aij,ik->jk', (1-self.Qij_dense), self.u)
			else:
				UQk = np.einsum('aij,ik->jk', mask * (1-self.Qij_dense), self.u)
			Z = np.einsum('jk,jq->kq', UQk, self.v)
		else: # flag_anomaly == False
			Z = np.einsum('k,q->kq', self.u.sum(axis=0), self.v.sum(axis=0))[np.newaxis, :, :]
			# Z *= (1.+self.beta_hat[self.T] * self.T) 
		Z = np.einsum('a,k->ak', self.beta_hat, Z)
		Z += self.bg

		non_zeros = Z > 0    
		self.w[non_zeros]  /= Z[non_zeros] 

		low_values_indices = self.w < self.err_max  # values are too low
		self.w[low_values_indices] = 0. #self.err_max  # and set to 0.

		dist_w = np.amax(abs(self.w - self.w_old)) 
		self.w_old = np.copy(self.w_old)

		return dist_w
	
	# @gl.timeit_cum('update_W_ass')
	def _update_W_assortative(self,subs_nzp,subs_nz,mask=None,subs_nz_mask=None):
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
		# TODO
		if self.T > 1:
			sub_w_nz = self.w.nonzero()
			uttkrp_DKQ = np.zeros_like(self.w) 

			UV = np.einsum('Ik,Ik->Ik', self.u[subs_nzp[1], :], self.v[subs_nzp[2], :]) 
			uttkrp_I = self.data_M_nz_Q[:, np.newaxis] * UV  
			# for k in range(self.K):
			# 	uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L) 


			for a, k in zip(*sub_w_nz):
				uttkrp_DKQ[:, k] += np.bincount(subs_nzp[0], weights=uttkrp_I[:, k], minlength=1)[0] 

			self.w = (self.ag - 1) + self.w_old * uttkrp_DKQ 

			if self.flag_anomaly == True:
				if mask is None:
					UQk = np.einsum('aij,ik->jk', (1-self.Qij_dense), self.u)
					Zk = np.einsum('jk,jk->k', UQk, self.v)
					Zk = Zk[np.newaxis,:] 
				else:
					Zk = np.einsum('aij,ijk->ak',mask * (1-self.Qij_dense),np.einsum('ik,jk->ijk',self.u,self.v))
					raise ValueError('update w: Check mask for flag_anomaly == True')
					# Zk = np.zeros_like(self.w)
					# for a,i,j in zip(*subs_nz_mask):
					#     Zk[a,:] += (1-self.Qij_dense[a,i,j]) * self.u[i] * self.v[j]
			else: # flag_anomaly == False 
				Zk = ((self.u_old.sum(axis=0)) * (self.v_old.sum(axis=0)))[np.newaxis, :]
			
			Zk *= (self.beta_hat[self.T] * self.T) 
			Zk += self.bg
			non_zeros = Zk > 0   
			self.w[Zk == 0] = 0
			self.w[non_zeros] /= Zk[non_zeros] 
			low_values_indices = self.w < self.err_max  # values are too low
			self.w[low_values_indices] = 0.  # and set to 0.


			low_values_indices = self.w < self.err_max  # values are too low
			self.w[low_values_indices] = 0.  # and set to 0.
			self.w_old = np.copy(self.w)
		


		# self.T == 0 
		sub_w_nz = self.w0.nonzero()
		uttkrp_DKQ = np.zeros_like(self.w0) 

		UV = np.einsum('Ik,Ik->Ik', self.u0[subs_nz[1], :], self.v0[subs_nz[2], :]) 
		uttkrp_I = self.data_M_nz_Q_0[:, np.newaxis] * UV  
		# for k in range(self.K):
		# 	uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=self.L) 


		for a, k in zip(*sub_w_nz):
			uttkrp_DKQ[:, k] += np.bincount(subs_nz[0], weights=uttkrp_I[:, k], minlength=1)[0] 

		self.w0 = (self.ag - 1) + self.w0 * uttkrp_DKQ 

		if self.flag_anomaly == True:
			if mask is None:
				UQk = np.einsum('aij,ik->jk', (1-self.Q_ij_dense_0), self.u0)
				Zk = np.einsum('jk,jk->k', UQk, self.v0)
				# Zk = Zk[np.newaxis,:] 
			else:
				Zk = np.einsum('aij,ijk->k',mask * (1-self.Q_ij_dense_0),np.einsum('ik,jk->ijk',self.u0,self.v0))
				# Zk = np.zeros_like(self.w)
				# for a,i,j in zip(*subs_nz_mask):
				#     Zk[a,:] += (1-self.Qij_dense[a,i,j]) * self.u[i] * self.v[j]
		else: # flag_anomaly == False 
			Zk = ((self.u0_old.sum(axis=0)) * (self.v0_old.sum(axis=0)))#[np.newaxis, :]
		 
		Zk += self.bg 
		non_zeros = Zk > 0   
		self.w0[:,Zk == 0] = 0
		self.w0[:,non_zeros] /= Zk[non_zeros]  

		
		low_values_indices = self.w0 < self.err_max  # values are too low
		self.w0[low_values_indices] = 0.  # and set to 0.
		self.w0_old = np.copy(self.w0)



		
		if self.T > 1: 
			dist_w = np.amax(abs(self.w - self.w_old))
			self.w_old = np.copy(self.w)
		else:
			dist_w = np.amax(abs(self.w0 - self.w0_old))
			self.w = np.copy(self.w0)
			self.w_old = np.copy(self.w0)
		

		return dist_w

	
	def _update_membership_Q(self, subs_nz, u, v, w, m):
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

		if not self.assortative:
			uttkrp_DK = sp_uttkrp(self.data_M_nz_Q, subs_nz, m, u, v, w)
		else:
			uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q, subs_nz, m, u, v, w)

		return uttkrp_DK
	
	def _update_membership_Q_0(self, subs_nz, u, v, w, m): 
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

		if not self.assortative:
			uttkrp_DK = sp_uttkrp(self.data_M_nz_Q_0, subs_nz, m, u, v, w)
		else:
			uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q_0, subs_nz, m, u, v, w)

		return uttkrp_DK
	
	def _update_membership_0(self, subs_nz, u, v, w, m):
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
		if not self.assortative:
			uttkrp_DK = sp_uttkrp(self.data_M_nz_Q_0, subs_nz, m, u, v, w)
		else:
			uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q_0, subs_nz, m, u, v, w)

		return uttkrp_DK
	
	def _update_membership(self, subs_nz, u, v, w, m):
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
		if not self.assortative:
			uttkrp_DK = sp_uttkrp(self.data_M_nz_Q, subs_nz, m, u, v, w)
		else:
			uttkrp_DK = sp_uttkrp_assortative(self.data_M_nz_Q, subs_nz, m, u, v, w)

		return uttkrp_DK

	def _check_for_convergence(self,datam0, data0, subs_nzp, subs_nz,T,r, it, loglik, coincide, convergence, data_T=None, mask=None):
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
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
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
			loglik = self.__Likelihood( datam0,data0, data_T, subs_nzp, subs_nz,T, mask=mask)
			if abs(loglik - old_L) < self.tolerance:
				coincide += 1
			else:
				coincide = 0
		if coincide > self.decision:
			convergence = True
		it += 1

		return it, loglik, coincide, convergence

	def __Likelihood(self, datam0, data0, data_T, subs_nzp, subs_nz,T,mask=None,subs_nz_mask=None,EPS=1e-12):
		"""
			Compute the pseudo log-likelihood of the data.
			Parameters
			----------
			data : sptensor/dtensor
				   Graph adjacency tensor.
			data_T : sptensor/dtensor
					 Graph adjacency tensor (transpose).
			mask : ndarray
				   Mask for selecting the held out set in the adjacency tensor in case of cross-validation.
			Returns
			-------
			l : float
				Pseudo log-likelihood value.
		""" 
		if self.T > 1.:   
			self.lambda0_ija  = self._lambda0_full(self.u, self.v, self.w)   
		self.lambda0_ija_0 = self._lambda0_full(self.u0, self.v0, self.w0)  


		# if mask is not None:
		# 	Adense = data.toarray()
		l = 0.

		if self.flag_anomaly == True:   
			'''
			Term containing Q and mu at t=0,  
			'''  
			if mask is None: 
				if 1 - self.mupr >= 0 :
					l += np.log(1-self.mupr+EPS) * (self.b1mQ0sum)  # (1-Q[0])*log (1-mu)
				if self.mupr >= 0 :
					l += np.log(self.mupr+EPS) * (self.Q0sum)    # Q[0]*log mu
			else:
				raise ValueError('Complete thsi part.') 
				# if 1 - self.mupr > 0 :
				# 	l += np.log(1-self.mupr+EPS) * (1-self.Q_ij_dense_0)[subs_nz_mask].sum()
				# if self.mupr > 0 :
				# 	l += np.log(self.mupr+EPS) * (self.Q_ij_dense_0[subs_nz_mask]).sum()
				
			
			'''
			Entropy of Bernoulli in Q  (1)
			'''
			if self.T > 1.:
				if mask is None:
					non_zeros = self.Qij_dense > 0
					non_zeros1 = (1-self.Qij_dense) > 0  
				else:
					non_zeros = np.logical_and( mask > 0,self.Qij_dense > 0 )
					non_zeros1 = np.logical_and( mask > 0, (1-self.Qij_dense ) > 0 ) 

				l -= (self.Qij_dense[non_zeros] * np.log(self.Qij_dense[non_zeros]+EPS)).sum()    # Q*log Q
				l -= ((1-self.Qij_dense)[non_zeros1]*np.log((1-self.Qij_dense)[non_zeros1]+EPS)).sum()   # (1-Q)*log(1-Q) 
			else:
				if mask is None:
					non_zeros = self.Q_ij_dense_0 > 0
					non_zeros1 = (1-self.Q_ij_dense_0) > 0  
				else:
					non_zeros = np.logical_and( mask > 0,self.Q_ij_dense_0 > 0 )
					non_zeros1 = np.logical_and( mask > 0, (1-self.Q_ij_dense_0 ) > 0 ) 

				l -= (self.Q_ij_dense_0[non_zeros] * np.log(self.Q_ij_dense_0[non_zeros]+EPS)).sum()    # Q*log Q
				l -= ((1-self.Q_ij_dense_0)[non_zeros1] * np.log((1-self.Q_ij_dense_0)[non_zeros1]+EPS)).sum()   # (1-Q)*log(1-Q) 

			 


			'''
			Term containing  (1-Q), lambda, beta, and A, (3)
			''' 
			logM_0 = np.log(self.lambda0_nz_0+EPS)  
			if isinstance(data0, skt.dtensor):
				Alog_0 = data0[data0.nonzero()] * ((1-self.Q_ij_dense_0)[data0.subs]) * logM_0 
			elif isinstance(data0, skt.sptensor):
				Alog_0 = (data0.vals * (1-self.Q_ij_nz_0) * logM_0).sum()   
			l += Alog_0  

			l -= ((1-self.Q_ij_dense_0) * self.lambda0_ija_0).sum()

			if self.T > 1:
				if mask is None:  
					logM = np.log(self.lambda0_nz+EPS)  
					if isinstance(datam0, skt.dtensor):
						Alog = datam0[datam0.nonzero()] * ((1-self.Qij_dense)[datam0.subs]) * logM 
					elif isinstance(datam0, skt.sptensor):
						Alog = (datam0.vals * (1-self.Qij_nz) * logM).sum()  
					l += Alog   

				
					if (1 - self.beta_hat[self.T] )>= 0:     
						l += np.log(1-self.beta_hat[self.T]+EPS) * self.b1mQAtm1At
					
					if (self.beta_hat[self.T]) >= 0:     
						l += np.log(self.beta_hat[self.T]+EPS) * self.b1mQ1mAtm1At
						l += np.log(self.beta_hat[self.T]+EPS) * self.b1mQAtm11mAt
					
					# l -=  (np.einsum('i,ijk->jk', self.beta_hat, ((1-self.Qij_dense) * self.lambda0_ija))).sum()  
					l -= ((1-self.Qij_dense) * self.lambda0_ija * (self.T*self.beta_hat[self.T])).sum()    
					# l -= self.beta_hat[-1] * ((1-self.Qij_dense) * self.lambda0_ija).sum()
				else:
					raise ValueError('Complete thsi part.')   
 			
			'''
			Term containing Q, pi, phi, ell, and A, (4)
			'''
			if mask is None: 
				if  self.pibr > 0:
					l += np.log(self.pibr+EPS) * self.Q0A0 # (1-A(t-1)) * A(t) *Q*log l
				l -= (self.Q0sum * self.pibr)

				if self.T > 1:
					if  self.ell >= 0:
						l += np.log(self.ell+EPS) * self.Q1mAtm1At # (1-A(t-1)) * A(t) *Q*log l
				
				
					if 1 - self.phi_hat[self.T] >= 0:     
						l += np.log(1-self.phi_hat[self.T]+EPS) * self.QAtm1At
					
					if self.phi_hat[-1] >= 0:      
						l += np.log(self.phi_hat[self.T]+EPS) * self.Q1mAtm1At 
						l += np.log(self.phi_hat[self.T]+EPS) * self.QAtm11mAt 
					# l -= self.phi_hat[-1] * self.ell * (self.Qij_dense).sum()
					l -= (self.Qsum * self.ell * (self.T*self.phi_hat[self.T])) 
			else:
				raise ValueError('Complete thsi part.') 
			# print('L4:', l)

		
		else:   # flag_anomaly = False
			if mask is not None:
				sub_mask_nz = mask.nonzero() 
				l  -=  self.lambda0_ija_0[sub_mask_nz].sum()   
			else:
				l -=   self.lambda0_ija_0.sum() 
			
			logM_0 = np.log(self.M_nz_0)  
			if isinstance(data0, skt.dtensor):
				Alog_0 = data0[data0.nonzero()] * logM_0 
			elif isinstance(data0, skt.sptensor):
				Alog_0 = (data0.vals * logM_0).sum()  
			l += Alog_0  


			if self.T > 1:
				logM = np.log(self.M_nz)  
				if isinstance(datam0, skt.dtensor):
					Alog = datam0[datam0.nonzero()] * logM 
				elif isinstance(datam0, skt.sptensor):
					Alog = (datam0.vals * logM).sum()  
				l += Alog


				if mask is not None:
					sub_mask_nz = mask.nonzero() 
					l  -=  self.T * self.beta_hat[self.T] * (self.lambda0_ija[sub_mask_nz].sum())
				else:
					l  -=  self.T * self.beta_hat[self.T] * (self.lambda0_ija.sum())
			
				l += np.log(self.beta_hat[self.T]+EPS) * (datam0.vals).sum()  
				l += np.log(1- self.beta_hat[self.T]+EPS) * (self.bAtm1At).sum()
				l += np.log(self.beta_hat[self.T]+EPS) * (self.bAtm11mAt).sum()
			
		
		if self.ag > 1.:
			if self.T > 1.:
				l += (self.ag -1) * np.log(self.u+EPS).sum()
				l += (self.ag -1) * np.log(self.v+EPS).sum()
			else:
				l += (self.ag -1) * np.log(self.u0+EPS).sum()
				l += (self.ag -1) * np.log(self.v0+EPS).sum()
		if self.bg > 0. :
			if self.T > 1.:
				l -= self.bg * self.u.sum()
				l -= self.bg * self.v.sum()
			else:
				l -= self.bg * self.u0.sum()
				l -= self.bg * self.v0.sum()  
		
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
		if w.ndim == 2:
			M = np.einsum('ik,jk->ijk', u, v)
			M = np.einsum('ijk,ak->aij', M, w)
		else:
			M = np.einsum('ik,jq->ijkq', u, v)
			M = np.einsum('ijkq,akq->aij', M, w)
		return M

	def _update_optimal_parameters(self):
		"""
			Update values of the parameters after convergence.
		""" 
		if self.T > 1:
			self.u_f = np.copy(self.u)+np.copy(self.u0)
			self.v_f = np.copy(self.v)+np.copy(self.v0)
			self.w_f = np.copy(self.w)+np.copy(self.w0)
		elif self.T  <= 1:
			self.u_f = np.copy(self.u)
			self.v_f = np.copy(self.v)
			self.w_f = np.copy(self.w)
		self.u0_f = np.copy(self.u0)
		self.v0_f = np.copy(self.v0)
		self.w0_f = np.copy(self.w0) 
		self.ell_f = np.copy(self.ell)
		self.phi_f = np.copy(self.phi)
		self.pibr_f = np.copy(self.pibr)
		self.mupr_f = np.copy(self.mupr)
		if self.flag_anomaly == True:
			self.Q_ij_dense_f = np.copy(self.Qij_dense)
		else:
			self.Q_ij_dense_f = np.zeros((1,self.N,self.N))
		
		if self.fix_beta == False:
			self.beta_f = np.copy(self.beta_hat[self.T])

	def output_results(self, nodes):
		"""
			Output results.
			Parameters
			----------
			nodes : list
					List of nodes IDs.
		"""

		outfile = self.out_folder + 'theta' + self.end_file
		np.savez_compressed(outfile + '.npz', u=self.u_f, v=self.v_f, w=self.w_f, beta = self.beta_f,max_it=self.final_it, pibr=self.pibr_f, mupr=self.mupr_f, phi=self.phi_f, ell=self.ell_f,
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

def plot_L(values, indices = None, k_i = 5, figsize=(8, 7), int_ticks=False, xlab='Iterations'):

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

def func_beta_static(beta_t, obj):
	EPS = 1e-12
	# TO DO: generalise for anomaly=False 
	# self.b1mAtm1At,self.bAtm11mAt,self.bAtm1At
	assert type(obj) is Dyn_ACD_static  
	if obj.flag_anomaly:
		Du = np.einsum('aij,jk->ik', 1-obj.Qij_dense,obj.v)  
		if not obj.assortative:
			w_k  = np.einsum('akq->kq', obj.w)
			Z_uk = np.einsum('iq,kq->ik', Du, w_k)
		else:
			w_k = np.einsum('ak->k', obj.w)
			Z_uk = np.einsum('ik,k->ik', Du, w_k)
		lambda0_ija = np.einsum('ik,ik->',obj.u, Z_uk)  

		bt  = - (obj.T * lambda0_ija)  # (1-Q) * \lambda  
		# bt  = - (lambda0_ija*(1-obj.Qij_dense)).sum()  # (1-Q) * \lambda 
		bt -=   obj.b1mQAtm1At  / (1-beta_t)  # adding Aij(t-1)*Aij(t)  

		# bt += obj.b1mAtm1At * (1-obj.Qij_nz).sum()  / beta_t  # adding sum A_hat from 1 to T. (1-A(t-1))A(t) 
		bt += obj.b1mQ1mAtm1At / beta_t   
		bt += obj.b1mQAtm11mAt / beta_t   # adding Aij(t-1)*(1-Aij(t))  

	else: 
		lambda0_ija = np.einsum('k,k->k',(obj.u).sum(axis=0), (obj.w).sum(axis=0)) 
		lambda0_ija = np.einsum('k,k->',(obj.v).sum(axis=0), lambda0_ija) 

		bt =  - ( obj.T * lambda0_ija)
		bt -=  obj.bAtm1At / (1-beta_t)  # adding Aij(t-1)*Aij(t)

		bt += obj.sum_data_hat / (beta_t+EPS)  # adding sum A_hat from 1 to T
		bt += obj.bAtm11mAt / (beta_t+EPS)  # adding Aij(t-1)*(1-Aij(t)) 
	return bt


def func_phi_static(phi_t, obj):
	EPS = 1e-12
	# TO DO: generalise for anomaly=False  
	assert type(obj) is Dyn_ACD_static  

	pt  =  - obj.ell * obj.Qsum * obj.T   # - \ell * Q
	pt -=  (obj.QAtm1At)  / (1-phi_t)  # adding Aij(t-1)*Aij(t)  

	pt += (obj.Q1mAtm1At) / (phi_t+EPS)  # adding sum A_hat from 1 to T. (1-A(t-1))A(t)
	pt += (obj.QAtm11mAt) / (phi_t+EPS)  # adding Aij(t-1)*(1-Aij(t))
	return pt