"""
	Main function to implement cross-validation given a number of communities.

	- Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
	- Infer parameters on the training set;
	- Calculate performance measures in the test set (AUC).
"""

import csv
import os
import pickle
from argparse import ArgumentParser 
import numpy as np 
import yaml
import sktensor as skt
import time
import sys
sys.path.append('../')
sys.path.append('../Dyn_ACD') 
from Dyn_ACD  import cv_functions
from Dyn_ACD  import tools


def main():
	p = ArgumentParser()
	p.add_argument('-a', '--algorithm', type=str, choices=['ACD_static'], default='ACD_static')  # configuration
	p.add_argument('-K', '--K', type=int, default=2)  # number of communities
	p.add_argument('-T', '--T', type=int, default=6)  # number of time snapshots
	p.add_argument('-A', '--adj', type=str, default='US-Air-Transportation_11_16_1000.csv')  # name of the network	email_Eu_core_tem
	p.add_argument('-f', '--in_folder', type=str, default='../data/input/real_data/')  # path of the input network
	p.add_argument('-o', '--out_folder', type=str, default='../data/output/5-fold_cv/real_data/')  # path to store outputs
	p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
	p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge
	p.add_argument('-r', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
	p.add_argument('-i', '--out_inference', type=bool, default=True)  # flag to output the inferred parameters  
	p.add_argument('-s', '--sep', type=str, default='\s+')  # flag to output the results in a csv file 
	p.add_argument('-E', '--flag_anomaly', type=int, default=0)  # flag to output the results in a csv file 

	args = p.parse_args()

	'''
	Cross validation parameters and set up output directory
	'''
	K = args.K
	adjacency = args.adj.split('.')[0]  # name of the network without extension  
	network = args.in_folder + args.adj.split('_')[0]+'/' +  args.adj  # network complete path
	out_results = bool(args.out_results)
	out_folder = args.out_folder + args.adj.split('_')[0] +'/'
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	'''
	Model parameters
	''' 
	
	algorithm = args.algorithm  # algorithm to use to generate the samples
	with open('../settings/setting_inference.yaml') as f:
		conf = yaml.load(f, Loader=yaml.FullLoader)
	conf['out_folder'] = out_folder
	conf['out_inference'] = bool(args.out_inference) 
	if args.flag_anomaly:
		conf['N_real'] = 10
	else:
		conf['N_real'] = 50
 
 

	'''
	Import data
	'''
	A, B, B_T, data_T_vals = tools.import_data(network, ego=args.ego, alter=args.alter, force_dense=True, header=0,sep=args.sep,binary=True)  
	nodes = A[0].nodes() 
	valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
	assert any(isinstance(B, vt) for vt in valid_types) 

	print(B.shape)
	T = max(0, min(args.T, B.shape[0]-1))
	print("T:",T)

	print('\n### CV procedure ###', T) 
	'''
	output results
	'''

	cols = ['algo','constrained','K','beta0', 'phi0','mu0','pi0','T'] #0-7
	cols.extend(['beta','beta_aggr', 'auc', 'auc_aggr', 'phi','phi_aggr', 'ell', 'pi', 'mu', 'opt_func', 'opt_func_aggr', 'csu', 'csu_aggr'])#20
	comparison = [0 for _ in range(len(cols))]
	comparison[0] = True if args.flag_anomaly== True else False
	comparison[1] = conf['constrained']
	comparison[2] = K
	if out_results:
		out_file = out_folder + adjacency + '_cv.csv'
		if not os.path.isfile(out_file):  # write header
			with open(out_file, 'w') as outfile:
				wrtr = csv.writer(outfile, delimiter=',', quotechar='"') 
				wrtr.writerow(cols)
		outfile = open(out_file, 'a')
		wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
		print(f'Results will be saved in: {out_file}') 

	comparison[3] = 0
	comparison[4] = 0
	comparison[5] = 0
	comparison[6] = 0

	for t in range(1,T+1):  # skip first and last time step (last is hidden)
		print('='*60)
		print(t)

		comparison[7] = t
		
		B_train = B[:t] # use data up to time t-1 for training
		print(B_train.shape) 

		time_start = time.time()
		N = B_train.shape[-1]

		conf['end_file'] = adjacency+'_'+ str(t) +'_'+ str(K) + '_ACD_static' # needed to plot inferred theta

		'''
		Run CRep on the training 
		'''							
		s = time.process_time()	
		u, v, w, u0, v0, w0, beta, phi, ell, pi, mu, maxL, algo_obj = cv_functions.fit_model(B_train, t, nodes=nodes, N=N,L=1,  K=K,algo =algorithm, flag_anomaly= args.flag_anomaly, **conf)
		if conf['initialization']:
			theta = np.load(conf['in_parameters']+'.npz',allow_pickle=True) 
			unr = cv_functions.normalize_nonzero_membership(u)
			u, cs_u = cv_functions.cosine_similarity(unr,theta['u'])
			v, cs_v = cv_functions.cosine_similarity(v,theta['v'])
			comparison[19] = cs_u
			print('cs_u,cs_v:', cs_u, cs_v)
		

		print('u.shape:', u.shape)
		pi = ell
		
		e = time.process_time() 
		print('inference:',e-s, 'secs')
		'''
		Calculate performance results
		'''
		comparison[8] = beta
		comparison[12] = phi
		comparison[14] = ell
		comparison[15] = pi
		comparison[16] = mu



		M0 = cv_functions._lambda0_full(u, v, w)
		if args.flag_anomaly == False: 
			Q = np.zeros_like(M0)
			M = cv_functions.calculate_conditional_expectation(M0, Q,t, beta=beta, phi=0, ell=0) # use data at time t-1 to predict t 
			M[B[t-1].nonzero()] = 1 - beta # to calculate AUC, expected number of edges if A(t-1)==1
		else:#args.flag_anomaly == True
			Q = cv_functions.QIJ_dense(B[:t+1],B[0],M0,T= t, beta=beta, phi=phi, ell=ell, mu= mu) 
			M = cv_functions.calculate_conditional_expectation(M0, Q,t, beta=beta, phi=phi, ell=ell) # use data at time t-1 to predict t 
			subs_nzp =  B[t].nonzero()
			M[subs_nzp] = (1-Q)[subs_nzp] * (1 - beta) + Q[subs_nzp] * (1 - phi)# to calculate AUC
			maskQ_train = B_train>0
			maskQ_test =  B[t]>0 # count for existing edges-only 
			# comparison[8] = cv_functions.calculate_AUC(Q[:t], Z, mask = maskQ_train)
			# comparison[9] = cv_functions.calculate_AUC(Q[t], Z, mask = maskQ_test)


		s = time.process_time()	
		if args.flag_anomaly == False:
			loglik_test = cv_functions.Likelihood_conditional(M,B[t],B[t-1], beta=beta, phi=phi, ell=ell)
			# M[B[t-1].nonzero()] = (1 - beta)# to calculate AUC, expected number of edges if A(t-1)==1
		else:#args.flag_anomaly == True
			loglik_test = cv_functions.Likelihood_conditional_Q(M,Q,t, B[t],B[t-1], beta=beta, phi=phi, ell=ell, pi= pi, mu= mu)
		e = time.process_time()
		print('loglik:',e-s, 'secs')
		
		comparison[10] = cv_functions.calculate_AUC(M, B[t])
		comparison[17] = loglik_test

		'''
		Inference using aggregated data
		'''
		B_aggr = B_train.sum(axis=0)
		# B_aggr[B_aggr>1] = 1 # binarize	

		u, v, w, u0, v0, w0, beta, phi, ell, pi, mu, maxL, algo_obj = cv_functions.fit_model(B_aggr[np.newaxis,:,:], 0, nodes=nodes, N=N,L=1, K=K,algo =algorithm, flag_anomaly= args.flag_anomaly, **conf)
		if conf['initialization']:
			theta = np.load(conf['in_parameters']+'.npz',allow_pickle=True) 
			unr = cv_functions.normalize_nonzero_membership(u)
			u, cs_u = cv_functions.cosine_similarity(unr,theta['u'])
			v, cs_v = cv_functions.cosine_similarity(v,theta['v'])
			comparison[20] = cs_u
			print('cs_u,cs_v:', cs_u, cs_v)
		pi = ell
		comparison[9] = beta
		comparison[13] = phi


		M0 = cv_functions._lambda0_full(u, v, w)
		if args.flag_anomaly == False: 
			Q = np.zeros_like(M0)
			M = cv_functions.calculate_conditional_expectation(M0, Q,t, beta=beta, phi=0, ell=0) # use data at time t-1 to predict t
			M[B[t-1].nonzero()] = 1 - beta # to calculate AUC
		else:   
			Q = cv_functions.QIJ_dense_aggre(B_aggr,B[0],M0,ell=ell, mu= mu)
			M = cv_functions.calculate_conditional_expectation(M0, Q, t, beta=beta, phi=phi, ell=ell) # use data at time t-1 to predict t

		# M = cv_functions.calculate_conditional_expectation(B_aggr,B_aggr, u, v, w, beta=beta, phi=phi, ell=ell, pi=pi, mu=mu)

		if args.flag_anomaly == True:
			# loglik_test = cv_functions.Likelihood_conditional_Q(M0,Q,t, B[t],B_aggr, beta=beta, phi=phi, ell=ell, pi= pi, mu= mu)
			loglik_test = 0
		else: 
			# loglik_test = cv_functions.Likelihood_conditional(M0,B[t],B_aggr, beta=beta, phi=phi, ell=ell)
			loglik_test = 0
			# M[B[t-1].nonzero()] = 1 - beta # to calculate AUC
		
		comparison[11] = cv_functions.calculate_AUC(M, B[t])
		comparison[18] = loglik_test

		print(t,comparison)

		if out_results:
			wrtr.writerow(comparison)
			outfile.flush()

	if out_results:
		outfile.close()
		

if __name__ == '__main__':
	main()
