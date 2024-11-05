"""
	Class for generation and management of synthetic networks with anomalies
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.stats import poisson
import random

from scipy.optimize import brentq, root

EPS = 1e-12

class SyntNetDynAnomaly(object):

	def __init__(self, m = 1, N = 200, K = 2, T = 8, prng = 10, avg_degree = 4., rho_anomaly = 0.1,
				structure = 'assortative', label = None, mu = None, pi = 0.4, phi = 0.2, ell = 0.2,beta=0.2,
				gamma = 0.5, eta = 0.5, L1=False,ag = 0.6, bg = 1., corr = 0., over = 0.,rho_node=0.9,
				verbose = 0, folder = '../../data/input', output_parameters = False,
				output_adj = False, outfile_adj = None,flag_anomalies=True):

		self.T = T 
		# Set network size (node number)
		self.N = N 
		# Set number of communities
		self.K = K
		# Set number of networks to be generated
		self.m = m
		# Set seed random number generator
		self.prng = prng
		# Set label (associated uniquely with the set of inputs)
		if label is not None:
			self.label = label
		else:
			self.label = ('_').join([str(N),str(K),str(avg_degree),str(T),str(beta),str(pi).split('.')[1],str(phi).split('.')[1],str(flt(rho_anomaly,d=2))])
			print(self.label ) 
		# Initialize data folder path
		self.folder = folder
		# Set flag for storing the parameters
		self.output_parameters = output_parameters
		# Set flag for storing the generated adjacency matrix
		self.output_adj = output_adj
		# Set name for saving the adjacency matrix
		self.outfile_adj = outfile_adj
		# Set required average degree
		self.avg_degree = avg_degree
		self.rho_anomaly = rho_anomaly  
		self.flag_anomalies = flag_anomalies


		# Set verbosity flag
		if verbose > 2 and not isinstance(verbose, int):
			raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
		self.verbose = verbose

		# Set Bernoullis parameters
		# if mu < 0 or mu > 1:
			# raise ValueError('The   parameter mu has to be in [0, 1]!')

		if pi < 0 or pi > 1:
			raise ValueError('The   parameter pi has to be in [0, 1]!')
		if pi == 1: pi = 1 - EPS
		if pi == 0: pi = EPS
		self.pi = pi
		
		if phi < 0 or phi > 1:
			raise ValueError('The parameter phi has to be in [0, 1]!')
		if phi == 1: phi = 1 - EPS
		if phi == 0: phi = EPS
		self.phi = phi

		if ell < 0 or ell > 1:
			raise ValueError('The parameter ell has to be in [0, 1]!')
		if ell == 1: ell = 1 - EPS
		if ell == 0: ell = EPS
		self.ell = ell

		if beta < 0 or beta > 1:
			raise ValueError('The parameter beta has to be in [0, 1]!')
		if beta == 1: beta = 1 - EPS
		if beta == 0: beta = EPS
		self.beta = beta

		if rho_anomaly < 0 or rho_anomaly > 1:
			raise ValueError('The rho anomaly has to be in [0, 1]!')
		
		self.ExpM = self.avg_degree * self.N * 0.5 
		mu = self.rho_anomaly * self.ExpM / ((1-np.exp(-self.pi)) * (self.N**2-self.N)) 
		# mu = self.rho_anomaly * self.ExpM / ((self.pi) * (self.N**2-self.N))
		if mu == 1: mu = 1 - EPS
		if mu == 0: mu = EPS 
		assert mu > 0. and mu < 1.
		self.mu = mu

		
		### Set MT inputs
		# Set the affinity matrix structure
		if structure not in ['assortative', 'disassortative', 'core-periphery', 'directed-biased']:
			raise ValueError('The available structures for the affinity matrix w '
							 'are: assortative, disassortative, core-periphery '
							 'and directed-biased!')
		self.structure = structure

		# Set eta parameter of the  Dirichlet distribution
		if eta <= 0 and L1:
			raise ValueError('The Dirichlet parameter eta has to be positive!')
		self.eta = eta
		# Set alpha parameter of the Gamma distribution
		if ag <= 0 and not L1:
			raise ValueError('The Gamma parameter alpha has to be positive!')
		self.ag = ag
		# Set beta parameter of the Gamma distribution
		if bg <= 0 and not L1:
			raise ValueError('The Gamma parameter beta has to be positive!')
		self.bg = bg
		# Set u,v generation preference
		self.L1 = L1
		# Set correlation between u and v synthetically generated
		if (corr < 0) or (corr > 1):
			raise ValueError('The correlation parameter has to be in [0, 1]!')
		self.corr = corr
		# Set fraction of nodes with mixed membership
		if (over < 0) or (over > 1):
				raise ValueError('The overlapping parameter has to be in [0, 1]!')
		self.over = over

	def Exp_ija_matrix(self,u,v,w): 
		if w.ndim == 2:
			M = np.einsum('ik,jk->ijk', u, v)
			M = np.einsum('ijk,ak->ij', M, w)
		else:
			M = np.einsum('ik,jq->ijkq', u, v)
			M = np.einsum('ijkq,akq->ij', M, w)

		# Exp_ija=np.einsum('ik,kq->iq',u,w)
		# Exp_ija=np.einsum('iq,jq->ij',Exp_ija,v) 
		return M

	def anomaly_network_PB(self, parameters = None):
		"""
			Generate a directed, possibly weighted network by using the anomaly model Poisson-Poisson
			Steps:
				1. Generate or load the latent variables Z_ij.
				2. Extract A_ij entries (network edges) from a Poisson (M_ij) distribution if Z_ij=0; from a Poisson (pi) distribution if Z_ij=1
			INPUT
			----------
			parameters : object
						 Latent variables z, s, u, v and w.
			OUTPUT
			----------
			G : Digraph
				MultiDiGraph NetworkX object. Self-loops allowed.
		"""

		# Set seed random number generator
		prng = np.random.RandomState(self.prng)

		### Latent variables
		if parameters is None:
			# Generate latent variables
			self.z, self.u, self.v, self.w = self._generate_lv(prng)
		else:
			# Set latent variables
			self.z, self.u, self.v, self.w = parameters
		
		if self.flag_anomalies:
			if len(self.z.shape) == 2: 
				self.z_t = np.zeros((self.T+1,self.N,self.N)) 
				self.z_t[0] = np.copy(self.z.toarray())
			elif len(self.z.shape) > 2:
				# self.z_t = np.zeros((self.T,self.N,self.N))
				self.z_t = np.copy(self.z.toarray())
		else:
			self.z_t = np.zeros((self.T+1,self.N,self.N)) 

		


		### Network generation
		G = [nx.DiGraph() for t in range(self.T+1)]
		for t in range(self.T+1):
			for i in range(self.N):
				G[t].add_node(i) 

		# # Compute M_ij
		# M = np.einsum('ik,jq->ijkq', self.u, self.v)
		# M = np.einsum('ijkq,kq->ij', M, self.w)

		# Compute M_ij
		M = self.Exp_ija_matrix(self.u, self.v,self.w)
		np.fill_diagonal(M, 0)

		if self.flag_anomalies: 
			# Set c sparsity parameter 
			c = brentq(eq_c, EPS, 20, args = (M, self.N,self.ExpM,self.rho_anomaly,self.mu))
		else:

			c = brentq(eq_c, EPS, 20, args = (M, self.N,self.ExpM,0.,0.))

		self.w *= c
 

		'''
		a) Build network: regular edges at t=0.
		'''
		# A = prng.poisson(c * M)
		# # A[A>0] = 1 # binarize the adjacency matrix
		# np.fill_diagonal(A, 0)
		# G0 = nx.to_networkx_graph(A, create_using=nx.MultiDiGraph)
 
		# '''
		# b) Build network: anomalous edges at t=0. 
		# '''
		# # weighted anomaly
		# A[self.z.nonzero()] = prng.poisson(self.pi * self.z.count_nonzero())
		# # A[self.z.nonzero()] = prng.binomial(1,self.pi,self.z.count_nonzero())
		# # A[A>0] = 1 # binarize the adjacency matrix
		# np.fill_diagonal(A, 0)
		# G = nx.to_networkx_graph(A, create_using=nx.MultiDiGraph)  

		if self.flag_anomalies:
			'''
			Build network: transitions at t = 0. 
			'''  
			list_t = np.arange(self.N)
			list_t = list_t.tolist()  
			random.shuffle(list_t)
			for i in list_t:
				for j in list_t:
					if i != j: 
						if not self.z[i,j]: 
							lambda_ij = c * M[i,j]  
						else:
							lambda_ij = self.pi 
							
						A_ij = prng.poisson(lambda_ij, 1)[0]

						if A_ij > 0:
							G[0].add_edge(i, j, weight = 1) # binarized?   
			
			'''
			Build network: transitions at t> 0. 
			'''  
			list_t = np.arange(self.N)
			list_t = list_t.tolist()  
			random.shuffle(list_t)
			for t in range(self.T):
				for i in list_t:
					for j in list_t: 
						if i != j:
							if not self.z[i,j]:
								lambda_ij = c * M[i,j] 
								exp_bl = np.exp(- self.beta * lambda_ij) 
								if G[t].has_edge(i,j): # the edge at previous time step: determines the transition rate 
									q = (1 - self.beta) 
								else:
									q = self.beta * lambda_ij 
								r = prng.rand() 
								if r <= q:
									G[t+1].add_edge(i, j, weight = 1) # binarized
							else:  
								r = prng.rand()  
								exp_pl = np.exp(-self.phi*self.ell)
								# exp_pl = 1
								if G[t].has_edge(i,j): # the edge at previous time step: determines the transition rate 
									q = (1 - self.phi) 
								else:
									q = self.phi * self.ell 
								r = prng.rand() 
								if r <= q:
									G[t+1].add_edge(i, j, weight = 1) # binarized
									self.z_t[t+1,i,j] = 1

			### Network post-processing
			nodes = list(G[0].nodes())
			assert len(nodes) == self.N 
			A = [nx.to_scipy_sparse_array(G[t], nodelist=nodes, weight='weight') for t in range(len(G))]
		else:#self.flag_anomalies:False
			'''
			Build network: transitions at t = 0. 
			'''  
			list_t = np.arange(self.N)
			list_t = list_t.tolist()  
			random.shuffle(list_t)
			for i in list_t:
				for j in list_t:
					if i != j: 
						lambda_ij = c * M[i,j] 
						A_ij = prng.poisson(lambda_ij, 1)[0] 
						if A_ij > 0:
							G[0].add_edge(i, j, weight = 1) # binarized?   
			
			'''
			Build network: transitions at t> 0. 
			'''  
			list_t = np.arange(self.N)
			list_t = list_t.tolist()  
			random.shuffle(list_t)
			for t in range(self.T):
				for i in list_t:
					for j in list_t: 
						if i != j: 
							lambda_ij = c * M[i,j] 
							exp_bl = np.exp(- self.beta * lambda_ij) 
							if G[t].has_edge(i,j): # the edge at previous time step: determines the transition rate 
								q = (1 - self.beta) 
							else:
								q = self.beta * lambda_ij 
							r = prng.rand() 
							if r <= q:
								G[t+1].add_edge(i, j, weight = 1) # binarized 

			### Network post-processing
			nodes = list(G[0].nodes())
			assert len(nodes) == self.N 
			A = [nx.to_scipy_sparse_array(G[t], nodelist=nodes, weight='weight') for t in range(len(G))]

		

		# Keep largest connected component
		A_sum = A[0].copy()
		for t in range(1,len(A)): A_sum += A[t] 
		G_sum = nx.from_scipy_sparse_array(A_sum,create_using=nx.DiGraph)
		Gc = max(nx.weakly_connected_components(G_sum), key=len)
		nodes_to_remove = set(G_sum.nodes()).difference(Gc)
		if self.flag_anomalies:
			G_sum.remove_nodes_from(list(nodes_to_remove))

		if self.output_adj:
			self._output_adjacency(nodes,A_sum,A,nodes_to_keep=list(G_sum.nodes()), outfile = self.outfile_adj)

		nodes = list(G_sum.nodes())

		if self.flag_anomalies: 
			for t in range(len(G)):
				G[t].remove_nodes_from(list(nodes_to_remove))
 
		self.N = len(nodes)
		print('self.N :', self.N )

		# G0 = G0.subgraph(nodes)
		# A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight') #
		# A0 = nx.to_scipy_sparse_array(G0, nodelist=nodes, weight='weight') #
		if self.flag_anomalies:
			for t in range(self.T): 
				self.z_t[t] = np.take(self.z_t[t], nodes, axis=1)
				self.z_t[t] = np.take(self.z_t[t], nodes, axis=0)
		else:
			n_nodes = len(nodes)
			self.z_t = np.zeros((self.T,n_nodes,n_nodes)) 

			# try:
			# 	self.z_t[t] = np.take(self.z_t[t], nodes, axis=1)
			# 	self.z_t[t] = np.take(self.z_t[t], nodes, axis=0)
			# except:
			# 	self.z_t[t] = self.z_t[:,nodes]
			# 	self.z_t[t] = self.z_t[t][nodes]  


		if self.u is not None:
			self.u = self.u[nodes]
			self.v = self.v[nodes] 
		self.N = len(nodes)

		if self.verbose > 0:print(f'Removed {len(nodes_to_remove)} nodes, because not part of the largest connected component')

		if self.verbose > 0:
			for t in range(len(G)):
				print('-'*30)
				print('t=',t)
				ave_w_deg = np.round(2 * G[t].number_of_edges() / float(G[t].number_of_nodes()), 3)   
				print(f'Number of nodes: {G[t].number_of_nodes()} \n'
					  f'Number of edges: {G[t].number_of_edges()}')
				print(f'Average degree (2E/N): {ave_w_deg}')
				print(f'Reciprocity at t: {nx.reciprocity(G[t])}')
				print('-'*30)

			self.check_reciprocity_tm1(A,A_sum)
		rec_syn = []
		for t in range(len(G)):
			rec_syn.append(np.round(nx.reciprocity(G[t]), 4))

		if self.output_parameters:
			self._output_results(nodes)

		if self.verbose == 2:
			for i in range(self.T):
				self._plot_A(A[i])
			if M is not None: self._plot_M(M)

		return G, rec_syn


	def _generate_lv(self, prng = 42):
		"""
			Generate z, u, v, w latent variables.
			INPUT
			----------
			prng : int
				   Seed for the random number generator.
			OUTPUT
			----------
			z : Numpy array
				Matrix NxN of model indicators (binary).

			u : Numpy array
				Matrix NxK of out-going membership vectors, positive element-wise.
				With unitary L1 norm computed row-wise.

			v : Numpy array
				Matrix NxK of in-coming membership vectors, positive element-wise.
				With unitary L1 norm computed row-wise.

			w : Numpy array
				Affinity matrix KxK. Possibly None if in pure SpringRank.
				Element (k,h) gives the density of edges going from the nodes
				of group k to nodes of group h.
		"""
		# Generate z through binomial distribution  

		if self.mu < 0:
			density = EPS
		else: 
			density = self.mu
		z = sparse.random(self.N,self.N, density=density, data_rvs=np.ones)
		upper_z = sparse.triu(z) 
		z = upper_z + upper_z.T  

		# Generate u, v for overlapping communities
		u, v = membership_vectors(prng, self.L1, self.eta, self.ag, self.bg, self.K,
								 self.N, self.corr, self.over)
		# Generate w
		w = affinity_matrix(self.structure, self.N, self.K, self.avg_degree)
		return z, u, v, w

	def _build_multilayer_edgelist(self,nodes,A_tot,A,nodes_to_keep=None):
		A_coo = A_tot.tocoo()
		data_dict = {'source':A_coo.row,'target':A_coo.col}
		for t in range(len(A)):
			data_dict['weight_t'+str(t)] = np.squeeze(np.asarray(A[t][A_tot.nonzero()]))
		  
		df_res = pd.DataFrame(data_dict)
		# print(len(df_res))
		# if nodes_to_keep is not None:
		# 	df_res = df_res[df_res.source.isin(nodes_to_keep) & df_res.target.isin(nodes_to_keep)]

		# nodes = list(set(df_res.source).union(set(df_res.target)))
		# id2node = {}
		# for i,n in enumerate(nodes):id2node[i] = n

		# df_res['source'] = df_res.source.map(id2node)
		# df_res['target'] = df_res.target.map(id2node)
	
		return df_res

	def _output_results(self, nodes):
		"""
			Output results in a compressed file.
			INPUT
			----------
			nodes : list
					List of nodes IDs.
		""" 
		output_parameters = self.folder + 'theta_' + self.label + '_' + str(self.prng)  
		np.savez_compressed(output_parameters + '.npz', z=self.z_t, u=self.u, v=self.v,
						w=self.w, beta=self.beta, ell=self.ell, phi=self.phi, mu=self.mu, pi=self.pi, nodes=nodes)
		if self.verbose:
			print()
			print(f'Parameters saved in: {output_parameters}.npz')
			print('To load: theta=np.load(filename), then e.g. theta["u"]')

	def _output_adjacency(self,nodes,A_tot,A,nodes_to_keep=None, outfile = None):
		"""
			Output the adjacency matrix. Default format is space-separated .csv
			with 3 columns: node1 node2 weight
			INPUT
			----------
			G: Digraph
			   DiGraph NetworkX object.
			outfile: str
					 Name of the adjacency matrix.
		"""
		if outfile is None:
			outfile = 'syn_' + self.label + '_' + str(self.prng)  + '.dat'

		df = self._build_multilayer_edgelist(nodes,A_tot,A,nodes_to_keep=nodes_to_keep)
		df.to_csv(self.folder + outfile, index=False, sep=' ')
		if self.verbose:
			print(f'Adjacency matrix saved in: {self.folder + outfile}')

	def _plot_A(self, A, cmap = 'PuBuGn',title='Adjacency matrix'):
		"""
			Plot the adjacency matrix produced by the generative algorithm.
			INPUT
			----------
			A : Scipy array
				Sparse version of the NxN adjacency matrix associated to the graph.
			cmap : Matplotlib object
				Colormap used for the plot.
		"""
		Ad = A.todense() 
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.matshow(Ad, cmap = plt.get_cmap(cmap))
		ax.set_title(title, fontsize = 15)
		for PCM in ax.get_children():
			if isinstance(PCM, plt.cm.ScalarMappable):
				break
		plt.colorbar(PCM, ax=ax)
		plt.show()

	def _plot_Z(self, cmap = 'PuBuGn'):
		"""
			Plot the anomaly matrix produced by the generative algorithm.
			INPUT
			----------
			cmap : Matplotlib object
				Colormap used for the plot.
		""" 
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.matshow(self.z, cmap = plt.get_cmap(cmap))
		ax.set_title('Anomaly matrix', fontsize = 15)
		for PCM in ax.get_children():
			if isinstance(PCM, plt.cm.ScalarMappable):
				break
		plt.colorbar(PCM, ax=ax)
		plt.show()


	def _plot_M(self, M, cmap = 'PuBuGn',title='MT means matrix'):
		"""
			Plot the M matrix produced by the generative algorithm. Each entry is the
			poisson mean associated to each couple of nodes of the graph.
			INPUT
			----------
			M : Numpy array
				NxN M matrix associated to the graph. Contains all the means used
				for generating edges.
			cmap : Matplotlib object
				Colormap used for the plot.
		"""
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.matshow(M, cmap = plt.get_cmap(cmap))
		ax.set_title(title, fontsize = 15)
		for PCM in ax.get_children():
			if isinstance(PCM, plt.cm.ScalarMappable):
				break
		plt.colorbar(PCM, ax=ax)
		plt.show()


	def check_reciprocity_tm1(self,A,A_sum):
		for t in range(1,len(A)):
			ref_subs = A[t].nonzero()
			M_t_T = A[t].transpose()[ref_subs]
			M_tm1_T = A[t-1].transpose()[ref_subs]
			nnz = float(A[t].count_nonzero())
			print(nnz,M_t_T.nonzero()[0].shape[0]/nnz,M_tm1_T.nonzero()[0].shape[0]/nnz)



def membership_vectors(prng = 10, L1 = False, eta = 0.5, alpha = 0.6, beta = 1, K = 2, N = 100, corr = 0., over = 0.):
	"""
		Compute the NxK membership vectors u, v using a Dirichlet or a Gamma distribution.
		INPUT
		----------
		prng: Numpy Random object
			  Random number generator container.
		L1 : bool
			 Flag for parameter generation method. True for Dirichlet, False for Gamma.
		eta : float
			  Parameter for Dirichlet.
		alpha : float
			Parameter (alpha) for Gamma.
		beta : float
			Parameter (beta) for Gamma.
		N : int
			Number of nodes.
		K : int
			Number of communities.
		corr : float
			   Correlation between u and v synthetically generated.
		over : float
			   Fraction of nodes with mixed membership.
		OUTPUT
		-------
		u : Numpy array
			Matrix NxK of out-going membership vectors, positive element-wise.
			Possibly None if in pure SpringRank or pure Multitensor.
			With unitary L1 norm computed row-wise.

		v : Numpy array
			Matrix NxK of in-coming membership vectors, positive element-wise.
			Possibly None if in pure SpringRank or pure Multitensor.
			With unitary L1 norm computed row-wise.
	"""
	# Generate equal-size unmixed group membership
	size = int(N / K)
	u = np.zeros((N, K))
	v = np.zeros((N, K))
	for i in range(N):
		q = int(math.floor(float(i) / float(size)))
		if q == K:
			u[i:, K - 1] = 1.
			v[i:, K - 1] = 1.
		else:
			for j in range(q * size, q * size + size):
				u[j, q] = 1.
				v[j, q] = 1.
	# Generate mixed communities if requested
	if over != 0.:
		overlapping = int(N * over)  # number of nodes belonging to more communities
		ind_over = np.random.randint(len(u), size=overlapping)
		if L1:
			u[ind_over] = prng.dirichlet(eta * np.ones(K), overlapping)
			v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.dirichlet(eta * np.ones(K), overlapping)
			if corr == 1.:
				assert np.allclose(u, v)
			if corr > 0:
				v = normalize_nonzero_membership(v)
		else:
			u[ind_over] = prng.gamma(alpha, 1. / beta, size=(N, K))
			v[ind_over] = corr * u[ind_over] + (1. - corr) * prng.gamma(alpha, 1. / beta, size=(overlapping, K))
			u = normalize_nonzero_membership(u)
			v = normalize_nonzero_membership(v)
	return u, v

def affinity_matrix(structure = 'assortative', N = 100, K = 2, avg_degree = 4., a = 0.1, b = 0.3):
	"""
		Compute the KxK affinity matrix w with probabilities between and within groups.
		INPUT
		----------
		structure : string
					Structure of the network.
		N : int
			Number of nodes.
		K : int
			Number of communities.
		a : float
			Parameter for secondary probabilities.
		OUTPUT
		-------
		p : Numpy array
			Array with probabilities between and within groups. Element (k,h)
			gives the density of edges going from the nodes of group k to nodes of group h.
	"""

	b *= a
	p1 = avg_degree * K / N

	if structure == 'assortative':
		p = p1 * a * np.ones((K,K))  # secondary-probabilities
		np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities

	elif structure == 'disassortative':
		p = p1 * np.ones((K,K))   # primary-probabilities
		np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

	elif structure == 'core-periphery':
		p = p1 * np.ones((K,K))
		np.fill_diagonal(np.fliplr(p), a * p1)
		p[1, 1] = b * p1

	elif structure == 'directed-biased':
		p = a * p1 * np.ones((K,K))
		p[0, 1] = p1
		p[1, 0] = b * p1

	return p

def normalize_nonzero_membership(u):
	"""
		Given a matrix, it returns the same matrix normalized by row.
		INPUT
		----------
		u: Numpy array
		   Numpy Matrix.
		OUTPUT
		-------
		The matrix normalized by row.
	"""

	den1 = u.sum(axis=1, keepdims=True)
	nzz = den1 == 0.
	den1[nzz] = 1.

	return u / den1

def eq_c(c,M, N,E,rho_a,mu):

	return np.sum(np.exp(-c*M)) - (N**2 -N) + E * (1-rho_a) / (1-mu) 

def flt(x,d=1):
	return round(x, d)

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
