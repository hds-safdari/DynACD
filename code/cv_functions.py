"""
	Functions used in the k-fold cross-validation procedure.
"""
 
import numpy as np 
import pandas as pd
from sklearn import metrics 
import networkx as nx
import scipy.sparse as sparse
import yaml
import sys
from scipy.stats import poisson
import sktensor as skt
import matplotlib.pyplot as plt
# import Dyn_ACD_static 
import Dyn_ACD_wtemp

 
def QIJ_dense(data,data0,lambda0_ija, lambda0_ija_0, T, beta, phi, ell, pi, mu,EPS=1e-12):
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

	data0_spt = preprocess(data0)   
	data_spt = preprocess(data)  
	if isinstance(data_spt, skt.sptensor):
		subs_nzp = data_spt.subs
	
	if isinstance(data0_spt, skt.sptensor):
		subs_nz = data0_spt.subs 
	
	if T > 0:
		original_shape = data.shape 
		new_shape = (original_shape[0] - 1, original_shape[1], original_shape[2])  
		data_b1mAtm1At = np.zeros(new_shape) 	# Aij(t)*(1-Aij(t-1))    
		data_bAtm11mAt = np.zeros(new_shape)	# A(t-1) (1-A(t))
		data_bAtm1At   = np.zeros(new_shape)	# A(t-1) A(t)  
	
		Q_ij_dense_tot   = np.zeros(data0.shape)
		Q_ij_dense_d_tot = np.zeros(data0.shape) 
	
	data0T = np.einsum('aij->aji', data0) # to calculate Pois terms by Aji(t) s in Q_{ij} 
	if T > 0:
		for i in range(T):   
			data_b1mAtm1At[i,:,:] = data[i+1,:,:] * (1 - data[i,:,:]) # (1-A(t-1)) A(t)
			data_bAtm11mAt[i,:,:] = (1 - data[i+1,:,:]) * data[i,:,:] # A(t-1) (1-A(t))
			data_bAtm1At[i,:,:]   = data[i+1,:,:] * data[i,:,:]       # A(t-1) A(t)  
		
		sum_b1mAtm1At  =  data_b1mAtm1At.sum(axis=0)	
		sum_bAtm11mAt  =  data_bAtm11mAt.sum(axis=0)
		sum_bAtm1At    =  data_bAtm1At.sum(axis=0)


		data_b1mAtm1AtT = np.einsum('aij->aji', data_b1mAtm1At)   	 
		data_bAtm11mAtT = np.einsum('aij->aji', data_bAtm11mAt) 	 
		data_bAtm1AtT = np.einsum('aij->aji', data_bAtm1At) 

		data_b1mAtm1AtT = np.einsum('aij->aij', data_b1mAtm1AtT)   

		sum_b1mAtm1At_T  = data_b1mAtm1AtT.sum(axis=0)
		sum_bAtm11mAt_T  = data_bAtm11mAtT.sum(axis=0)
		sum_bAtm1At_T    = data_bAtm1AtT.sum(axis=0)
	 

	lambda0_ijaT = np.einsum('aij->aji', lambda0_ija)
	lambda0_ijaT_0 = np.einsum('aij->aji',lambda0_ija_0)
	# assert np.allclose(np.einsum('aij->aji',lambda0_ija),lambda0_ijaT)
	"""
		Compute Q_ij_dense at zeros of (1-Aij(t-1)) * Aij(t) by dense Aij(t-1) * (1-Aij(t)) and Aij(t-1) * Aij(t) 
	"""
	# if not hasattr('dense_data0'):
	dense_data0 = np.copy(data0) 
	dense_data0T = np.einsum('aij->aji', dense_data0) # to calculate Pois terms by Aji(t) s in Q_{ij}  

	if T > 0:
		# A(t) (1-A(t-1)), A(t-1) (1-A(t)), and A(t-1) A(t) and their tranposes to  dense array
		# if not hasattr('dense_data_b1mAtm1At'):
		dense_data_b1mAtm1At = np.copy(data_b1mAtm1At)
		dense_data_bAtm11mAt = np.copy(data_bAtm11mAt)
		dense_data_bAtm1At   = np.copy(data_bAtm1At)
		dense_data_b1mAtm1AtT = np.copy(data_b1mAtm1AtT)
		dense_data_bAtm11mAtT = np.copy(data_bAtm11mAtT)
		dense_data_bAtm1AtT   = np.copy(data_bAtm1AtT)

	# Q_ij_dense_tot   = np.zeros(dense_data0.shape)
	# Q_ij_dense_d_tot = np.zeros(dense_data0.shape)
	
	if T == 0:
		flag_dense = True
		if flag_dense:
			"""
				Compute Q_ij_dense for t=0, dense matrix 
			# """
			Q_ij_dense_0   = (mu * poisson.pmf(dense_data0, pi) * poisson.pmf(dense_data0T, pi))#* np.exp(-self.pibr*2)
			Q_ij_dense_0_d = (1-mu) * poisson.pmf(dense_data0, lambda0_ija_0) * poisson.pmf(dense_data0T, lambda0_ijaT_0)  #* np.exp(-(lambda0_ija_0+lambda0_ijaT_0))
			Q_ij_dense_0_d += Q_ij_dense_0
			non_zeros = Q_ij_dense_0_d > 0
			Q_ij_dense_0[non_zeros] /=  Q_ij_dense_0_d[non_zeros]  
		else:
			"""
				Compute Q_ij_dense for t=0, sptensor format
			# """      
			nz_recon_I_0 =  mu * poisson.pmf(data0vals0, pi) * poisson.pmf(data0_T_vals0, pi)
			nz_recon_Id_0 = nz_recon_I_0 + (1-mu) * poisson.pmf(data0vals0, lambda0_nz_0) * poisson.pmf(data0_T_vals0, lambda0_nzT_0) 

			non_zeros = nz_recon_Id_0 > 0
			nz_recon_I_0[non_zeros] /=  nz_recon_Id_0[non_zeros]

			Q_ij_dense_0 = np.ones(lambda0_ija_0.shape)
			Q_ij_dense_0 *=  mu * np.exp(-pibr*2)
			Q_ij_dense_d_0 = Q_ij_dense_0 + (1-mu) * np.exp(-lambda0_ija_0) * np.exp(-lambda0_ijaT_0)
			non_zeros = Q_ij_dense_d_0 > 0
			Q_ij_dense_0[non_zeros] /= Q_ij_dense_d_0[non_zeros]

			Q_ij_dense_0[subs_nz] = np.copy(nz_recon_I_0) 


		Q_ij_dense_0[0] = np.maximum(Q_ij_dense_0[0], transpose_tensor(Q_ij_dense_0[0])) # make it symmetric
		assert np.allclose(Q_ij_dense_0[0], Q_ij_dense_0[0].T, rtol=1e-05, atol=1e-08)
		np.fill_diagonal(Q_ij_dense_0[0], 0.)
		assert (Q_ij_dense_0[0] > 1).sum() == 0
		Q_ij_nz_0 = Q_ij_dense_0[subs_nz]

		# self.Q0A0  = (Q_ij_dense_0[subs_nz] * dense_data0[subs_nz]).sum()	  # needed in the update of Likelihood
		Qsum = (Q_ij_dense_0).sum()  # needed in the update of Likelihood
		b1mQsum = (1-Q_ij_dense_0).sum()   # needed in the update of Likelihood
		# self.Q_ij_dense_0 = np.copy(Q_ij_dense_0)
		# self.Q_ij_nz_0 = np.copy(Q_ij_dense_0[subs_nz])
		Q0A0  = (data0_spt.vals * Q_ij_nz_0).sum() 



	elif T > 0:

		"""
			Compute Q_ij_dense for t>0, full dataset, poisson
		"""
		# s = time.time()
		Q_ij_dense_tot_0   = (mu * poisson.pmf(dense_data0, pi) * poisson.pmf(dense_data0T, pi))[0]#* np.exp(-self.pibr*2)
		Q_ij_dense_d_tot_0 = ((1-mu) * poisson.pmf(dense_data0, lambda0_ija_0) * poisson.pmf(dense_data0T, lambda0_ijaT_0) )[0]  #* np.exp(-(lambda0_ija_0+lambda0_ijaT_0))
		# e = time.time()
		# print(f"Q_ij_dense_tot_0, Q_ij_dense_d_tot_0 created in {e - s} sec.")

		# sum_b1mAtm1At  =  dense_data_b1mAtm1At.sum(axis=0)
		# sum_bAtm11mAt  =  dense_data_bAtm11mAt.sum(axis=0)
		# sum_bAtm1At    =  dense_data_bAtm1At.sum(axis=0)

		# sum_b1mAtm1At_T  = dense_data_b1mAtm1AtT.sum(axis=0)
		# sum_bAtm11mAt_T  = dense_data_bAtm11mAtT.sum(axis=0)
		# sum_bAtm1At_T    = dense_data_bAtm1AtT.sum(axis=0)

		# s = time.time()
		sum_a = (sum_b1mAtm1At + sum_bAtm11mAt + sum_b1mAtm1At_T + sum_bAtm11mAt_T)
		sum_b = (sum_bAtm1At + sum_bAtm1At_T)
		if phi > 0:
			# print(sum_b1mAtm1At.shape)
			log_Q_ij_dense  = (sum_a) * np.log(phi+EPS)
		else:
			raise ValueError('Invalid value', phi)
		
		if (1-phi) > 0:
			log_Q_ij_dense  += (sum_b) * np.log(1-phi+EPS)
		else:
			raise ValueError('Invalid value', phi)
		
		if ell > 0:
			log_Q_ij_dense  += (sum_b1mAtm1At + sum_b1mAtm1At_T) * np.log(ell+EPS)
		else:
			raise ValueError('Invalid value', ell)
		# print('log_Q_ij_dense.shape', log_Q_ij_dense.shape)

		Q_ij_dense_tot[0] =  np.exp(log_Q_ij_dense) * np.exp(-2. * T * ell * phi)  * Q_ij_dense_tot_0 
		# e = time.time()
		# print(f"Q_ij_dense_tot[0] created in {e - s} sec.")

		# s = time.time()
		if beta > 0:
			log_Q_ij_dense_d = (sum_a) * np.log(beta + EPS)
		else:
			raise ValueError('Invalid value', beta)

		if 1 - beta > 0:
			log_Q_ij_dense_d += (sum_b) * np.log(1 - beta + EPS)
		else:
			raise ValueError('Invalid value', beta)

		tmp = np.einsum('aij,aij ->ij', dense_data_b1mAtm1At, np.log(lambda0_ija + EPS))
		log_Q_ij_dense_d += (tmp + tmp.T)
		# assert np.allclose(tmp.T,np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1AtT, np.log(lambda0_ijaT + EPS)))
		# log_Q_ij_dense_d += np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1At, np.log(lambda0_ija + EPS))
		# log_Q_ij_dense_d += np.einsum('aij,aij ->ij', self.dense_data_b1mAtm1AtT, np.log(lambda0_ijaT + EPS))
		log_Q_ij_dense_d -= beta * (lambda0_ija.sum(axis=0) + lambda0_ijaT.sum(axis=0))
		Q_ij_dense_d_tot[0] = Q_ij_dense_d_tot_0 * np.exp(log_Q_ij_dense_d)

		Q_ij_dense_d_tot += Q_ij_dense_tot 
		non_zeros = Q_ij_dense_d_tot > 0  
		Q_ij_dense_tot[non_zeros] /= Q_ij_dense_d_tot[non_zeros]     

		Q_ij_dense_tot[0] = np.maximum( Q_ij_dense_tot[0], transpose_tensor(Q_ij_dense_tot[0])) # make it symmetric
		np.fill_diagonal(Q_ij_dense_tot[0], 0.)

		# e = time.time()
		# print(f"Q_ij_dense_tot created in {e - s} sec.")
		assert (Q_ij_dense_tot[0] > 1).sum() == 0
		assert np.allclose(Q_ij_dense_tot[0], Q_ij_dense_tot[0].T, rtol=1e-05, atol=1e-08)


		# Q_ij_dense_0 = np.zeros(dense_data0.shape)
		# Q_ij_dense_0 = np.copy(Q_ij_dense_tot)

		# s = time.time()
		QAtm1At     = np.einsum('ij, aij-> ', Q_ij_dense_tot[0], dense_data_bAtm1At)   # needed in the update of Likelihood
		QAtm11mAt   = np.einsum('ij, aij-> ', Q_ij_dense_tot[0], dense_data_bAtm11mAt) # needed in the update of Likelihood
		Q1mAtm1At   = np.einsum('ij, aij-> ', Q_ij_dense_tot[0], dense_data_b1mAtm1At) # needed in the update of self.ell
		# self.Qsum       = np.einsum('ij -> ', Q_ij_dense_tot[0]) # needed in the update of self.ell
		
		b1mQAtm1At   = np.einsum('ij, aij-> ', (1-Q_ij_dense_tot[0]), dense_data_bAtm1At)   # needed in the update of beta, Likelihood
		b1mQAtm11mAt = np.einsum('ij, aij-> ', (1-Q_ij_dense_tot[0]), dense_data_bAtm11mAt) # needed in the update of beta
		b1mQ1mAtm1At = np.einsum('ij, aij-> ', (1-Q_ij_dense_tot[0]), dense_data_b1mAtm1At) # needed in the update of beta

		Qsum = (Q_ij_dense_tot).sum()  # needed in the update of Likelihood
		b1mQsum = (1-Q_ij_dense_tot).sum()   # needed in the update of Likelihood
		# assert (Q_ij_dense_0 > 1).sum() == 0
		# self.Q_ij_dense_0 = np.copy(Q_ij_dense_tot)
		Q_ij_dense_0 = Q_ij_dense_tot
		Q_ij_nz_0 = Q_ij_dense_tot[subs_nz]
		Q0A0  = (data0_spt.vals * Q_ij_dense_tot[subs_nz]).sum()

		# e = time.time()
		# print(f"Q_ij_dense_tot_npz self in {e - s} sec.")

		# s = time.time()

		Q_ij_dense_tot_npz = np.tile(Q_ij_dense_tot, (T, 1, 1))
		# e = time.time()
		# print(f"Q_ij_dense_tot_npz in {e - s} sec.")
		

	if T > 0:  
		return Q_ij_dense_tot[0] #, Q_ij_dense_tot_npz[subs_nzp]
	else:
		return Q_ij_dense_0[0]#, Q_ij_dense_0[subs_nz]

def QIJ_dense_old(data,data0,M0, M00, T, beta, phi, ell, pi, mu,EPS=1e-12): 
	cmap = 'PuBuGn' 
	if T > 0:
		original_shape = data.shape 
		new_shape = (original_shape[0] - 1, original_shape[1], original_shape[2])  
		data_b1mAtm1At = np.zeros(new_shape) 	# Aij(t)*(1-Aij(t-1))    
		data_bAtm11mAt = np.zeros(new_shape)	# A(t-1) (1-A(t))
		data_bAtm1At   = np.zeros(new_shape)	# A(t-1) A(t)  
	
		Q_ij_dense_tot   = np.zeros(data0.shape)
		Q_ij_dense_d_tot = np.zeros(data0.shape)
	
	data0T = np.einsum('ij->ji', data0) # to calculate Pois terms by Aji(t) s in Q_{ij} 
	if T > 0:
		for i in range(T):   
			data_b1mAtm1At[i,:,:] = data[i+1,:,:] * (1 - data[i,:,:]) # (1-A(t-1)) A(t)
			data_bAtm11mAt[i,:,:] = (1 - data[i+1,:,:]) * data[i,:,:] # A(t-1) (1-A(t))
			data_bAtm1At[i,:,:]   = data[i+1,:,:] * data[i,:,:]       # A(t-1) A(t)  
		
		sum_b1mAtm1At  =  data_b1mAtm1At.sum(axis=0)	
		sum_bAtm11mAt  =  data_bAtm11mAt.sum(axis=0)
		sum_bAtm1At    =  data_bAtm1At.sum(axis=0)


		data_b1mAtm1AtT = np.einsum('aij->aji', data_b1mAtm1At)   	 
		data_bAtm11mAtT = np.einsum('aij->aji', data_bAtm11mAt) 	 
		data_bAtm1AtT = np.einsum('aij->aji', data_bAtm1At) 

		data_b1mAtm1AtT = np.einsum('aij->aij', data_b1mAtm1AtT)   

		sum_b1mAtm1At_T  = data_b1mAtm1AtT.sum(axis=0)
		sum_bAtm11mAt_T  = data_bAtm11mAtT.sum(axis=0)
		sum_bAtm1At_T    = data_bAtm1AtT.sum(axis=0)
	 
	M0T  = np.einsum('aij->aji', M0)   
	M00T = np.einsum('aij->aji', M00)    

	Q_ij_dense_tot_0   = (mu * poisson.pmf(data0, pi) * poisson.pmf(data0T, pi))#* np.exp(-self.pibr*2)
	Q_ij_dense_d_tot_0 = ((1-mu) * poisson.pmf(data0, M00[0]) * poisson.pmf(data0T, M00T[0]) ) #* np.exp(-(lambda0_ija_0+lambda0_ijaT_0))
	if T > 0: 
		if phi > 0:
			# print(sum_b1mAtm1At.shape)
			log_Q_ij_dense  = (sum_b1mAtm1At + sum_bAtm11mAt + sum_b1mAtm1At_T + sum_bAtm11mAt_T) * np.log(phi+EPS)
		else:
			raise ValueError('Invalid value', phi)
		
		if (1-phi) > 0:
			log_Q_ij_dense  += (sum_bAtm1At + sum_bAtm1At_T) * np.log(1-phi+EPS)
		else:
			raise ValueError('Invalid value', phi)
		
		if ell > 0:
			log_Q_ij_dense  += (sum_b1mAtm1At + sum_b1mAtm1At_T) * np.log(ell+EPS)
		else:
			raise ValueError('Invalid value', ell)
		# print('log_Q_ij_dense.shape', log_Q_ij_dense.shape)

		Q_ij_dense_tot =  np.exp(log_Q_ij_dense) * np.exp(-2. * T * ell * phi)  * Q_ij_dense_tot_0 

		for idx in range(T):  
			# log_Q_ij_dense  = (data_bAtm11mAt[idx]+data_bAtm11mAtT[idx]) * np.log(phi+EPS) 
			# log_Q_ij_dense += (data_bAtm1At[idx]+data_bAtm1AtT[idx])    * np.log(1-phi+EPS)  
			# log_Q_ij_dense += (data_b1mAtm1At[idx]+data_b1mAtm1AtT[idx]) * np.log(ell*phi+EPS)
			# log_Q_ij_dense -= 2.*ell*phi  

			# Q_ij_dense_idx = np.exp(log_Q_ij_dense) 

			# if idx == 0:
			# 	Q_ij_dense_tot = np.copy(Q_ij_dense_idx)
			# 	Q_ij_dense_tot *= Q_ij_dense_tot_0
			# elif idx > 0:
			# 	Q_ij_dense_tot *= np.copy(Q_ij_dense_idx) 

			if beta > 0:
				log_Q_ij_dense_d  = (data_bAtm11mAt[idx]+data_bAtm11mAtT[idx]) * np.log(beta+EPS)
			else:
				raise ValueError('Invalid value', beta)
			
			if (1-beta) > 0:
				log_Q_ij_dense_d += (data_bAtm1At[idx]+data_bAtm1AtT[idx]) * np.log(1-beta+EPS) 
			else:
				raise ValueError('Invalid value', beta)
			log_Q_ij_dense_d += data_b1mAtm1At[idx] * np.log(M0[idx]*beta+EPS)
			log_Q_ij_dense_d += data_b1mAtm1AtT[idx] * np.log(M0T[idx]*beta+EPS)
			log_Q_ij_dense_d -= beta * (M0[idx]+M0T[idx])  

			Q_ij_dense_d_idx = np.exp(log_Q_ij_dense_d) 

			if idx == 0:
				Q_ij_dense_d_tot = np.copy(Q_ij_dense_d_idx)
				Q_ij_dense_d_tot *= Q_ij_dense_d_tot_0 
			elif idx > 0:
				Q_ij_dense_d_tot *= Q_ij_dense_d_idx
		
		
		Q_ij_dense_d_tot += Q_ij_dense_tot 
		non_zeros = Q_ij_dense_d_tot > 0  
		Q_ij_dense_tot[non_zeros] /= Q_ij_dense_d_tot[non_zeros]
		Q_ij_dense_tot = np.maximum( Q_ij_dense_tot, transpose_tensor(Q_ij_dense_tot)) # make it symmetric  
	  
		np.fill_diagonal(Q_ij_dense_tot, 0.) 
		assert (Q_ij_dense_tot > 1).sum() == 0
		assert np.allclose(Q_ij_dense_tot, Q_ij_dense_tot.T, rtol=1e-05, atol=1e-08)
		# plt.figure(figsize=(6,6))
		# cmap = 'PuBuGn' 
		# plt.imshow(Q_ij_dense_tot, cmap=cmap, interpolation='nearest')
		# # plt.imshow(B[0], cmap=cmap, interpolation='nearest')
		# plt.colorbar(fraction=0.046)
		# plt.title('Q_ij_dense_tot')
		# plt.show()


	elif T == 0:
		Q_ij_dense_d_tot_0 += Q_ij_dense_tot_0  
		non_zeros_0 = Q_ij_dense_d_tot_0 > 0    
		Q_ij_dense_tot_0[non_zeros_0] /= Q_ij_dense_d_tot_0[non_zeros_0]

	if T > 0:
		return Q_ij_dense_tot
	else: 
		return Q_ij_dense_tot_0



def QIJ_dense_aggre(data,data0,M0, pi, mu):  
	dataT = np.einsum('ij->ji', data) 
	data  = np.einsum('ij->ij', data) 
	data0T = np.einsum('ij->ji', data0)  
	M0T = np.einsum('ij->ji', M0)   

	Q_ij_dense  = np.ones(data[0].shape) 

	Q_ij_dense  =  mu * poisson.pmf(data, pi) * poisson.pmf(dataT, pi)  

	Q_ij_dense_d = (1-mu) * poisson.pmf(data, M0) * poisson.pmf(dataT, M0T)    

	Q_ij_dense_d += Q_ij_dense 
	non_zeros = Q_ij_dense_d > 0
	Q_ij_dense[non_zeros] /= Q_ij_dense_d[non_zeros]    

	assert np.allclose(Q_ij_dense, Q_ij_dense.T, rtol=1e-05, atol=1e-08) 
	Q_ij_dense = np.maximum( Q_ij_dense, transpose_ij(Q_ij_dense)) # make it symmetric
	np.fill_diagonal(Q_ij_dense, 0.) 
	return Q_ij_dense  

def QIJ_conditional_old(data,data_tm1,data0,M0, T, beta, phi, ell, pi, mu):     
	dataT = transpose_ij(data)    
	data_tm1T = transpose_ij(data_tm1)    
	data0T = transpose_ij(data0)
	M0T = transpose_ij(M0)
	subs_nzp =  data.nonzero() 
	
	sum1 = data[subs_nzp] * (1-data_tm1)[subs_nzp] #(1-A(t-1)) * A(t) 
	sum1T = dataT[subs_nzp] * (1-data_tm1T )[subs_nzp]

 
	sum2 = (1-data)[subs_nzp] * data_tm1[subs_nzp] #A(t-1) * (1-A(t)) 
	sum2T = (1-dataT)[subs_nzp] * data_tm1T[subs_nzp]
 
	sum3 = data[subs_nzp] * data_tm1[subs_nzp] #A(t-1) * A(t) 
	sum3T = dataT[subs_nzp] * data_tm1[subs_nzp]    

	nz_recon_I1 =  mu * poisson.pmf(data0[subs_nzp], ell) * poisson.pmf(data0T[subs_nzp], ell) * np.exp(-2*T*ell*phi) * np.power(ell*phi,sum1)* np.power(ell*phi,sum1T) * np.power(phi,sum2) 
	nz_recon_I = nz_recon_I1  * np.power(phi,sum2T)* np.power(1-phi,sum3)* np.power(1-phi,sum3T)

	nz_recon_Id1 = (1-mu) * poisson.pmf(data0[subs_nzp], M0[subs_nzp]) * poisson.pmf(data0T[subs_nzp], M0T[subs_nzp]) 

	nz_recon_Id2 = np.exp(-T*M0[subs_nzp]*beta) * np.exp(-T*M0T[subs_nzp]*beta) * np.power(M0[subs_nzp]*beta,sum1) * np.power(beta,sum2) * np.power(1-beta,sum3) 
		
	nz_recon_Id = nz_recon_Id1 * nz_recon_Id2 * np.power(M0T[subs_nzp]*beta,sum1T) * np.power(beta,sum2T) * np.power(1-beta,sum3T)  

	nz_recon_Id = nz_recon_I + nz_recon_Id
	non_zeros = nz_recon_Id > 0 
	nz_recon_I[non_zeros] /=  nz_recon_Id[non_zeros]  
	
	Q_ij_dense = np.ones(M0.shape) 

	Q_ij_dense *=  mu * np.exp(-ell*2) * np.exp(-2*T*ell*phi)
	Q_ij_dense_d = Q_ij_dense + (1-mu) * np.exp(-T*(M0+M0T)) * np.exp(-T*(M0+M0T)*beta) 
	non_zeros = Q_ij_dense_d > 0
	Q_ij_dense[non_zeros] /= Q_ij_dense_d[non_zeros] 

	
	Q_ij_dense[subs_nzp] = nz_recon_I  

	Q_ij_dense = np.maximum( Q_ij_dense, transpose_ij(Q_ij_dense)) # make it symmetric
	np.fill_diagonal(Q_ij_dense, 0.)

	assert (Q_ij_dense > 1).sum() == 0  
	return Q_ij_dense, Q_ij_dense[subs_nzp] 


def Likelihood_conditional_Q(M, Qij_dense,T, data,data_tm1, beta, phi, ell, pi, mu,mask=None,EPS=1e-12):
	"""
		Compute the pseudo log-likelihood of the data.
		Compute the log-likelihood of the data conditioned in the previous time step
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
		l : float, log-likelihood value.
	"""
	l = 0
	non_zeros = Qij_dense > 0
	non_zeros1 = (1-Qij_dense) > 0   
	l -= (Qij_dense[non_zeros] * np.log(Qij_dense[non_zeros]+EPS)).sum()    # Q[0]*log Q[0]
	l -= ((1-Qij_dense)[non_zeros1]*np.log((1-Qij_dense)[non_zeros1]+EPS)).sum()   # (1-Q[0])*log(1-Q[0])  

	if 1 - mu >= 0 :
		l += np.log(1-mu+EPS) * (1-Qij_dense).sum()   # (1-Q[0])*log (1-mu)
	if mu >= 0 :
		l += np.log(mu+EPS) * (Qij_dense).sum()    # Q[0]*log mu

	l = - (((T*beta+1)*M)  * (1-Qij_dense)).sum()
	if T > 0:
		l = - (((T*phi)*ell) * Qij_dense).sum() 
	
	sub_nz_and = np.logical_and( data>0,(1-data_tm1)>0 )  # (1-A(t-1)) * A(t)  

	Alog = (data[sub_nz_and] * (1-data_tm1)[sub_nz_and])  
	Alog *= np.log(M[sub_nz_and]+EPS)  
	Alog *= (1-(Qij_dense[-1])[sub_nz_and])
	l += Alog.sum() 
	Alog = (data[sub_nz_and] * (1-data_tm1)[sub_nz_and] * np.log(beta+EPS) ) * (1-(Qij_dense[-1])[sub_nz_and])
	l += Alog.sum() 
	if T > 0:
		Alog = (data[sub_nz_and] * (1-data_tm1)[sub_nz_and] * (np.log(phi+EPS)+np.log(ell+EPS)) ) * (Qij_dense[-1])[sub_nz_and] 
	l += Alog.sum()


	sub_nz_and = np.logical_and(data_tm1>0,(1-data)>0) #   A(t-1) * (1-A(t))
	l += np.log(beta+EPS) * ((1-data)[sub_nz_and] * data_tm1[sub_nz_and]* (1-(Qij_dense[-1]))[sub_nz_and]).sum()
	l += np.log(phi+EPS) * ((1-data)[sub_nz_and] * data_tm1[sub_nz_and]* (Qij_dense[-1])[sub_nz_and]).sum()

	sub_nz_and = np.logical_and(data>0,data_tm1>0)#   A(t-1) * A(t)
	l += np.log(1-beta+EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]* (1-(Qij_dense[-1]))[sub_nz_and]).sum()
	l += np.log(1-phi+EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]* (Qij_dense[-1])[sub_nz_and]).sum()

	if np.isnan(l):
		print("Likelihood is NaN!!!!")
		sys.exit(1)
	else:
		return l

def Likelihood_conditional(M, data,data_tm1, beta, phi, ell,mask=None,EPS=1e-12):
	"""
		Compute the pseudo log-likelihood of the data.
		Compute the log-likelihood of the data conditioned in the previous time step
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
		l : float, log-likelihood value.
	"""
	l = 0

	l = - ((beta*M)).sum() 
	sub_nz_and = np.logical_and( data>0,(1-data_tm1)>0 )  # (1-A(t-1)) * A(t)
	Alog = (data[sub_nz_and] * (1-data_tm1)[sub_nz_and] * np.log(M[sub_nz_and]+EPS) ) 
	l += Alog.sum() 
	Alog = (data[sub_nz_and] * (1-data_tm1)[sub_nz_and] * np.log(beta+EPS) )
	l += Alog.sum() 
	Alog = (data[sub_nz_and] * (1-data_tm1)[sub_nz_and] * (np.log(phi+EPS)+np.log(phi+EPS)) ) 
	l += Alog.sum()


	sub_nz_and = np.logical_and(data_tm1>0,(1-data)>0) #   A(t-1) * (1-A(t))
	l += np.log(beta+EPS) * ((1-data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()
	l += np.log(phi+EPS) * ((1-data)[sub_nz_and] * data_tm1[sub_nz_and]).sum()

	sub_nz_and = np.logical_and(data>0,data_tm1>0)#   A(t-1) * A(t)
	l += np.log(1-beta+EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()
	l += np.log(1-phi+EPS) * (data[sub_nz_and] * data_tm1[sub_nz_and]).sum()

	if np.isnan(l):
		print("Likelihood is NaN!!!!")
		sys.exit(1)
	else:
		return l


def _lambda0_full(u, v, w):
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


def transpose_ij(M):
	"""
		Compute the transpose of a matrix.

		Parameters
		----------
		M : ndarray
			Numpy matrix.

		Returns
		-------
		Transpose of the matrix.
	"""

	return np.einsum('ij->ji', M)

def transpose_aij(M):
	"""
		Compute the transpose of a matrix.

		Parameters
		----------
		M : ndarray
			Numpy matrix.

		Returns
		-------
		Transpose of the matrix.
	"""

	return np.einsum('aij->jia', M)

def calculate_conditional_expectation(M,beta=1.):
	"""
		Compute the conditional expectations, e.g. the parameters of the conditional distribution lambda_{ij}.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		eta : float
			  Reciprocity coefficient.
		mean : ndarray
			   Matrix with mean entries.

		Returns
		-------
		Matrix whose elements are lambda_{ij}.
	"""    
	M = (beta * M)/ (1 + beta * M) 
	return M


def calculate_conditional_expectation_Q(M0, Q,  beta=1., phi=0., ell=0.):
	"""
		Compute the conditional expectations, e.g. the parameters of the conditional distribution lambda_{ij}.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		u : ndarray
			Out-going membership matrix.
		v : ndarray
			In-coming membership matrix.
		w : ndarray
			Affinity tensor.
		eta : float
			  Reciprocity coefficient.
		mean : ndarray
			   Matrix with mean entries.

		Returns
		-------
		Matrix whose elements are lambda_{ij}.
	"""   
	# if T > 2:  
	# 	M = ((1-Q) * beta * M0)/ (1 + beta * M0) + (Q * phi * ell[-1]) / (1 + phi * ell[-1])  
	# else:
	# 	M = ((1-Q) * beta * M0)/ (1 + beta * M0) + (Q * phi * ell) / (1 + phi * ell)   
	# return M

	M = ((1-Q) * beta * M0)/ (1 + beta * M0) + (Q * phi * ell) / (1 + phi * ell)   
	return M

def calculate_AUC(pred, data0, mask=None):
	"""
		Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
		(true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
		(true negative).

		Parameters
		----------
		pred : ndarray
			   Inferred values.
		data0 : ndarray
				Given values.
		mask : ndarray
			   Mask for selecting a subset of the adjacency tensor.

		Returns
		-------
		AUC value.
	"""

	data = (data0 > 0).astype('int') 
	if mask is None:
		fpr, tpr, thresholds = metrics.roc_curve(data.flatten(), pred.flatten())
	else:
		fpr, tpr, thresholds = metrics.roc_curve(data[mask > 0], pred[mask > 0])

	return metrics.auc(fpr, tpr)


def shuffle_indices_all_matrix(N, L, rseed=10):
	"""
		Shuffle the indices of the adjacency tensor.

		Parameters
		----------
		N : int
			Number of nodes.
		L : int
			Number of layers.
		rseed : int
				Random seed.

		Returns
		-------
		indices : ndarray
				  Indices in a shuffled order.
	"""

	n_samples = int(N * N)
	indices = [np.arange(n_samples) for _ in range(L)]
	rng = np.random.RandomState(rseed)
	for l in range(L):
		rng.shuffle(indices[l])

	return indices

def calculate_Q_dense(At,Atm1,b1Atm1At,Atm11At,bAtAtm1,M,T,pi,mu, phi=1,ell=1,mask=None,EPS=1e-12):
	AT = transpose_ij(At)
	MT = transpose_ij(M)
	num = (mu+EPS) * poisson.pmf(At, (pi+EPS)) * poisson.pmf(AT, (pi+EPS)) 
	num *= np.exp(-2*T*ell*phi) * np.power(ell*phi,b1Atm1At) * np.power(phi,Atm11At) * np.power(1-phi,bAtAtm1)* np.power(ell*phi,transpose_ij(b1Atm1At)) * np.power(phi,transpose_ij(Atm11At)) * np.power(1-phi,transpose_ij(bAtAtm1))
	# num = poisson.pmf(A,pi) * poisson.pmf(AT,pi)* (mu+EPS)
	den = num + poisson.pmf(At,M) * poisson.pmf(AT,MT) * (1-mu+EPS)
	if mask is None:
		return num / den 
	else:
		return num[mask.nonzero()] / den[mask.nonzero()] 


def calculate_Q_dense_T1(At,M,pi,mu,mask=None,EPS=1e-12):
	AT = transpose_ij(At)
	MT = transpose_ij(M)
	num = (mu+EPS) * poisson.pmf(At, (pi+EPS)) * poisson.pmf(AT, (pi+EPS))   
	den = num + poisson.pmf(At,M) * poisson.pmf(AT,MT) * (1-mu+EPS)
	if mask is None:
		return num / den 
	else:
		return num[mask.nonzero()] / den[mask.nonzero()] 


def extract_mask_kfold(indices, N, fold=0, NFold=5):
	"""
		Extract a non-symmetric mask using KFold cross-validation. It contains pairs (i,j) but possibly not (j,i).
		KFold means no train/test sets intersect across the K folds.

		Parameters
		----------
		indices : ndarray
				  Indices of the adjacency tensor in a shuffled order.
		N : int
			Number of nodes.
		fold : int
			   Current fold.
		NFold : int
				Number of total folds.

		Returns
		-------
		mask : ndarray
			   Mask for selecting the held out set in the adjacency tensor.
	"""

	L = len(indices)
	mask = np.zeros((L, N, N), dtype=bool)
	for l in range(L):
		n_samples = len(indices[l])
		test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
		mask0 = np.zeros(n_samples, dtype=bool)
		mask0[test] = 1
		mask[l] = mask0.reshape((N, N))

	return mask


def fit_model(data, T, nodes, N, L, K, flag_anomaly,algo = 'ACD_Wdynamic', **conf):
	"""
		Model directed networks by using a probabilistic generative model that assume community parameters and
		reciprocity coefficient. The inference is performed via EM algorithm.

		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		B_T : None/sptensor
			  Graph adjacency tensor (transpose).
		data_T_vals : None/ndarray
					  Array with values of entries A[j, i] given non-zero entry (i, j).
		nodes : list
				List of nodes IDs.
		N : int
			Number of nodes.
		L : int
			Number of layers.
		algo : str
			   Configuration to use (CRep, CRepnc, CRep0).
		K : int
			Number of communities.

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
				 Maximum log-likelihood.
		mod : obj
			  The CRep object.
	"""

	# setting to run the algorithm 
	with open(conf['out_folder'] + 'setting_' + algo + '.yaml', 'w') as f:
		yaml.dump(conf, f)

	if algo == 'ACD_static':
		model = Dyn_ACD_static.Dyn_ACD_static(N=N, L=L, K=K,flag_anomaly=flag_anomaly, **conf)
		u, v, w, u0, v0, w0, beta, phi, ell, pi, mu, maxL = model.fit(data=data, T=T, nodes=nodes)
	elif algo == 'ACD_Wdynamic':
		model = Dyn_ACD_wtemp.Dyn_ACD_wtemp(N=N, L=L, K=K,flag_anomaly=flag_anomaly, **conf)
		u, v, w, beta, phi, ell, pi, mu, maxL = model.fit(data=data, T=T, nodes=nodes)
	else:
		raise ValueError('algo is invalid',algo)

	return u, v, w, beta, phi, ell, pi, mu, maxL,model


def CalculatePermuation(U_infer,U0):  
	"""
	Permuting the overlap matrix so that the groups from the two partitions correspond
	U0 has dimension NxK, reference memebership
	"""
	N,RANK=U0.shape
	M=np.dot(np.transpose(U_infer),U0)/float(N);   #  dim=RANKxRANK
	rows=np.zeros(RANK);
	columns=np.zeros(RANK);
	P=np.zeros((RANK,RANK));  # Permutation matrix
	for t in range(RANK):
	# Find the max element in the remaining submatrix,
	# the one with rows and columns removed from previous iterations
		max_entry=0.;c_index=1;r_index=1;
		for i in range(RANK):
			if columns[i]==0:
				for j in range(RANK):
					if rows[j]==0:
						if M[j,i]>max_entry:
							max_entry=M[j,i];
							c_index=i;
							r_index=j;
	 
		P[r_index,c_index]=1;
		columns[c_index]=1;
		rows[r_index]=1;

	return P


def cosine_similarity(U_infer,U0):
	"""
	I'm assuming row-normalized matrices 
	"""
	P=CalculatePermuation(U_infer,U0) 
	U_infer=np.dot(U_infer,P);      # Permute infered matrix
	N,K=U0.shape
	U_infer0=U_infer.copy()
	U0tmp=U0.copy()
	cosine_sim=0.
	norm_inf=np.linalg.norm(U_infer,axis=1)
	norm0=np.linalg.norm(U0,axis=1  )
	for i in range(N):
		if(norm_inf[i]>0.):U_infer[i,:]=U_infer[i,:]/norm_inf[i]
		if(norm0[i]>0.): U0[i,:]=U0[i,:]/norm0[i]
	   
	for k in range(K):
		cosine_sim+=np.dot(np.transpose(U_infer[:,k]),U0[:,k])
	U0=U0tmp.copy()
	return U_infer0,cosine_sim/float(N) 

def normalize_nonzero_membership(U):
	"""
		Given a matrix, it returns the same matrix normalized by row.

		Parameters
		----------
		U: ndarray
		   Numpy Matrix.

		Returns
		-------
		The matrix normalized by row.
	"""

	den1 = U.sum(axis=1, keepdims=True)
	nzz = den1 == 0.
	den1[nzz] = 1.

	return U / den1
	
def evalu(U_infer, U0, metric='f1', com=False):
	"""
		Compute an evaluation metric.

		Compare a set of ground-truth communities to a set of detected communities. It matches every detected
		community with its most similar ground-truth community and given this matching, it computes the performance;
		then every ground-truth community is matched with a detected community and again computed the performance.
		The final performance is the average of these two metrics.

		Parameters
		----------
		U_infer : ndarray
				  Inferred membership matrix (detected communities).
		U0 : ndarray
			 Ground-truth membership matrix (ground-truth communities).
		metric : str
				 Similarity measure between the true community and the detected one. If 'f1', it used the F1-score,
				 if 'jaccard', it uses the Jaccard similarity.
		com : bool
			  Flag to indicate if U_infer contains the communities (True) or if they have to be inferred from the
			  membership matrix (False).

		Returns
		-------
		Evaluation metric.
	"""

	if metric not in {'f1', 'jaccard'}:
		raise ValueError('The similarity measure can be either "f1" to use the F1-score, or "jaccard" to use the '
						 'Jaccard similarity!')

	K = U0.shape[1]

	gt = {}
	d = {}
	threshold = 1 / U0.shape[1]
	for i in range(K):
		gt[i] = list(np.argwhere(U0[:, i] > threshold).flatten())
		if com:
			try:
				d[i] = U_infer[i]
			except:
				pass
		else:
			d[i] = list(np.argwhere(U_infer[:, i] > threshold).flatten())
	# First term
	R = 0
	for i in np.arange(K):
		ground_truth = set(gt[i])
		_max = -1
		M = 0
		for j in d.keys():
			detected = set(d[j])
			if len(ground_truth & detected) != 0:
				precision = len(ground_truth & detected) / len(detected)
				recall = len(ground_truth & detected) / len(ground_truth)
				if metric == 'f1':
					M = 2 * (precision * recall) / (precision + recall)
				elif metric == 'jaccard':
					M = len(ground_truth & detected) / len(ground_truth.union(detected))
			if M > _max:
				_max = M
		R += _max
	# Second term
	S = 0
	for j in d.keys():
		detected = set(d[j])
		_max = -1
		M = 0
		for i in np.arange(K):
			ground_truth = set(gt[i])
			if len(ground_truth & detected) != 0:
				precision = len(ground_truth & detected) / len(detected)
				recall = len(ground_truth & detected) / len(ground_truth)
				if metric == 'f1':
					M = 2 * (precision * recall) / (precision + recall)
				elif metric == 'jaccard':
					M = len(ground_truth & detected) / len(ground_truth.union(detected))
			if M > _max:
				_max = M
		S += _max

	return np.round(R / (2 * len(gt)) + S / (2 * len(d)), 4)



def output_adjacency_G(G, outfile = None):
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
	edges = list(G.edges(data=True)) 
	try:
		data = [[u, v, d['weight']] for u, v, d in edges]
	except:
		data = [[u, v, 1] for u, v, d in edges]  
	df = pd.DataFrame(data, columns=['source', 'target', 'w'], index=None)
	df.to_csv(outfile, index=False, sep=' ')
	print(f'anomalous network saved in: {outfile}') 
	return  df
 
def build_multilayer_edgelist(nodes,A_tot,A,nodes_to_keep=None):
	A_coo = A_tot.tocoo()
	data_dict = {'source':A_coo.row,'target':A_coo.col}
	for t in range(len(A)):
		data_dict['weight_t'+str(t)] = np.squeeze(np.asarray(A[t][A_tot.nonzero()]))

	df_res = pd.DataFrame(data_dict)  
	# if nodes_to_keep is not None:
	# 	df_res = df_res[df_res.source.isin(nodes_to_keep) & df_res.target.isin(nodes_to_keep)]

	# nodes = list(set(df_res.source).union(set(df_res.target)))
	id2node = {}
	for i,n in enumerate(nodes):id2node[i] = n

	df_res['source'] = df_res.source.map(id2node)
	df_res['target'] = df_res.target.map(id2node)
	print(df_res)

	return df_res

def output_adjacency(nodes,A_tot,A,nodes_to_keep=None, outfile = None):
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

	df = build_multilayer_edgelist(nodes,A_tot,A,nodes_to_keep=nodes_to_keep)
	df.to_csv( outfile, index=False, sep=' ') 

def transpose_tensor(A):
	'''
	Assuming the first index is for the layer, it transposes the second and third
	'''
	return np.einsum('ij->ji',A)


def flt(x,d=3):
	return round(x, d)

def unpack_filename(
		filename: str,
		sep: str = '_'
	) -> pd.Series:
		list_str = filename.split(sep)
		columns = ['N','K','avg_degree','T','beta','pi','phi','rho_anomaly','rseed']
		assert len(columns) == len(list_str)
		return pd.Series(list_str, index = columns)
	
def pack_filename(
		N: int = 100,
		K: int = 5,
		avg_degree: float = 5.0,
		T: int = 20,
		beta: float = 0.2,
		pi: int = 2,
		phi: int = 2,
		rho_anomaly: float = 0.2,
		rseed: int = 0,
		sep: str = '_'
	) -> pd.Series:
		list_str = [N,K,avg_degree,T,beta,pi,phi,rho_anomaly,rseed]
		filename = (sep).join(list_str)
		return filename



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