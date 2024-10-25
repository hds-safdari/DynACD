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
from Dyn_ACD  import tools_temp


def main():
	p = ArgumentParser()
	p.add_argument('-a', '--algorithm', type=str, choices=['ACD_Wdynamic', 'MT'], default='ACD_Wdynamic')  # configuration
	p.add_argument('-K', '--K', type=int, default=4)  # number of communities
	p.add_argument('-T', '--T', type=int, default=12)  # number of time snapshots
	p.add_argument('-A', '--adj', type=str, default='US-Air-Transportation_11_16_1000.csv')  # name of the network	email_Eu_core_tem
	p.add_argument('-f', '--in_folder', type=str, default='../data/input/real_data/opsahl-ucsocial/')  # path of the input network
	p.add_argument('-o', '--out_folder', type=str, default='../data/output/5-fold_cv/real_data/opsahl-ucsocial/injected/')  # path to store outputs
	p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
	p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge
	p.add_argument('-r', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
	p.add_argument('-i', '--out_inference', type=bool, default=True)  # flag to output the inferred parameters  
	p.add_argument('-s', '--sep', type=str, default='\s+')  # flag to output the results in a csv file   \s+
	p.add_argument('-ef', '--end_file', type=str, default='.csv')  # flag to output the results in a csv file 
	p.add_argument('-E', '--flag_anomaly', type=int, default=0)  # flag to output the results in a csv file 

	args = p.parse_args()

	'''
	Cross validation parameters and set up output directory
	'''
	K = args.K
	adjacency = args.adj.split(args.end_file)[0]  # name of the network without extension  
	in_folder = args.in_folder   # network complete path
	network  = in_folder + args.adj
	out_results = bool(args.out_results)
	out_folder = args.out_folder  
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	'''
	Model parameters
	''' 
	
	algorithm = args.algorithm  # algorithm to use to generate the samples
	with open('../settings/setting_' + algorithm + '_K'+str(K)+'.yaml') as f:
		conf = yaml.load(f, Loader=yaml.FullLoader)
	conf['out_folder'] = out_folder
	conf['out_inference'] = bool(args.out_inference)

	'''
	Import data
	'''
	A, B, B_T, data_T_vals = tools_temp.import_data(network, ego=args.ego, alter=args.alter, force_dense=True, header=0,sep=args.sep,binary=False)  
	nodes = A[0].nodes() 
	valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
	assert any(isinstance(B, vt) for vt in valid_types) 

	print(B.shape)
	T = max(0, min(args.T, B.shape[0]-1))
	# T = 2
	print("T:",T)
	'''
	output results
	'''

	cols = ['algo','K','T', 'csu'] #0-2
	cols.extend(['beta', 'phi', 'ell', 'pi', 'mu'])#8
	comparison = [0 for _ in range(len(cols))]
	comparison[0] = True if args.flag_anomaly== True else False 
	comparison[1] = K

	t = np.copy(T)
	comparison[2] = t

	if out_results:
		out_file = out_folder + adjacency + '.csv'
		if not os.path.isfile(out_file):  # write header
			with open(out_file, 'w') as outfile:
				wrtr = csv.writer(outfile, delimiter=',', quotechar='"') 
				wrtr.writerow(cols)
		outfile = open(out_file, 'a')
		wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
		print(f'Results will be saved in: {out_file}') 

	
	B[B>1] = 1 # binarize 
	N = B.shape[-1] 
	conf['end_file'] = adjacency+'_'+ str(K)+'_'+ str(bool(args.flag_anomaly))+ '_ACD_Wdynamic' # needed to plot inferred theta

	'''
	Run CRep on the training 
	'''							
	s = time.process_time()	
	u, v, w, beta, phi, ell, pi, mu, maxL, algo_obj = cv_functions.fit_model(B, t, nodes=nodes, N=N,L=t,  K=K,algo = 'ACD_Wdynamic',flag_anomaly= args.flag_anomaly, **conf)
	comparison[4] = beta
	comparison[5] = phi
	comparison[6] = ell
	comparison[7] = pi
	comparison[8] = mu
	e = time.process_time() 
	print('inference:',e-s, 'secs')
	if conf['initialization']:
		theta = np.load(conf['in_parameters']+'.npz',allow_pickle=True) 
		unr = cv_functions.normalize_nonzero_membership(u)
		u, cs_u = cv_functions.cosine_similarity(unr,theta['u'])
		v, cs_v = cv_functions.cosine_similarity(v,theta['v'])
		comparison[3] = cs_u
		print('cs_u,cs_v:', cs_u, cs_v)

	# '''
	# Inference using aggregated data
	# '''
	# B_aggr = B.sum(axis=0)
	# B_aggr[B_aggr>1] = 1 # binarize	

	# u, v, w, u0, v0, w0, beta, phi, ell, pi, mu, maxL, algo_obj = cv_functions.fit_model(B_aggr[np.newaxis,:,:], 0, nodes=nodes, N=N,L=1, K=K,algo = 'ACD_Wdynamic',flag_anomaly= args.flag_anomaly, **conf)
	# if conf['initialization']:
	# 	theta = np.load(conf['in_parameters']+'.npz',allow_pickle=True) 
	# 	unr = cv_functions.normalize_nonzero_membership(u)
	# 	u, cs_u = cv_functions.cosine_similarity(unr,theta['u'])
	# 	v, cs_v = cv_functions.cosine_similarity(v,theta['v'])
	# 	comparison[4] = cs_u
	# 	print('cs_u,cs_v:', cs_u, cs_v)

	if out_results:
		wrtr.writerow(comparison)
		outfile.flush()

	if out_results:
		outfile.close()
		

if __name__ == '__main__':
	main()
