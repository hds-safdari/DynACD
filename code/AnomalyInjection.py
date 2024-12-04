"""
	
"""

import csv
import os
import pickle
from argparse import ArgumentParser
import scipy as sp
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../Dyn_ACD') 
from Dyn_ACD  import cv_functions 
import numpy as np
import tools as tl
import yaml
import sktensor as skt
import random
import networkx as nx
import time


def main():
	p = ArgumentParser()
	p.add_argument('-A', '--adj', type=str, default='adjacency_2008_23_len')  # name of the network  ['adjacency_2008_23_len.dat', 'US-Air-Transportation_11_16_1000.csv']
	p.add_argument('-f', '--in_folder', type=str, default='../data/input/real_data/transfermarket/')  # path of the input network ['transfermarket', 'US-Air-Transportation']
	p.add_argument('-o', '--out_folder', type=str, default='../data/input/real_data/transfermarket/')  # path to store outputs 
	p.add_argument('-ef', '--end_file', type=str, default='.dat')  # flag to output the results in a csv file 
	p.add_argument('-TT',  '--timestep', type=int, default=5)  # time 
	p.add_argument('-sep', '--sep', type=str, default='\s+')  # flag to output the results in a csv file  

	p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
	p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge   
	p.add_argument('-b', '--binary', type=str, default=False) 
	p.add_argument('-ud', '--undirected', type=str, default=False) 
	p.add_argument('-r', '--out_results', type=bool, default=False)  # flag to output the results in a csv file
	p.add_argument('-g', '--gt', type=str, default=False)  
	p.add_argument('-im', '--inj_min', type=float, default=0.05)  # min of injection level
	p.add_argument('-ij', '--inj_step', type=float, default=0.05)  # injection steps
	p.add_argument('-it',  '--inj_it'    , type=int, default=1)  # number of inection 
	p.add_argument('-s',  '--rseed'    , type=int, default=3000)  # number of inection  : 0,2,5,42,100
		# # p.add_argument('-K', '--K', type=int, default=10)  # number of communities 

	args = p.parse_args()
	overlap = False


	out_results = args.out_results
	undirected=args.undirected
	binary=args.binary
	gt = args.gt

	injection_it = args.inj_it 
	TT = args.timestep


	random.seed(args.rseed)

	'''
	Model parameters
	'''
	# algorithm = args.algorithm  # algorithm to use to generate the samples 
	adjacency = args.adj.split(args.end_file)[0]  # name of the network without extension 
	print('='*65)
	print('adjacency matrix:', adjacency)

	out_folder = args.out_folder +'injected/'
	if not os.path.exists(out_folder):
		os.makedirs(out_folder) 
	'''
	Import data
	'''
	network = args.in_folder + adjacency + args.end_file  # network complete path 
	A, B, _, _=tl.import_data(network,undirected=undirected,ego=args.ego,alter=args.alter,
											force_dense=True,binary=binary, sep = args.sep)
	nodes = A[0].nodes()
	valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
	assert any(isinstance(B, vt) for vt in valid_types) 
	B_aggr = B.sum(axis=0)
 
	if args.out_results:
		out_file = out_folder + args.adj +  '_' + str(args.rseed) + '.dat'
		if not os.path.isfile(out_file):  # write header
			with open(out_file, 'w') as outfile:
				wrtr = csv.writer(outfile, delimiter=',', quotechar='"') 
		outfile = open(out_file, 'a')
		wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
		print(f'Results will be saved in: {out_file}')

	time_start = time.time()
	L = B.shape[0]
	N = B.shape[-1] 
	
	# rng = np.random.RandomState(10)

	
	nd_list = list(nodes)  
	idx_d = {}
	for idx, n in enumerate(nodes): 
		idx_d[idx] = n  


	G = [nx.DiGraph() for t in range(TT)]
	all_edges = [] 
	for t in range(TT): 
		print('t:', t)
		G[t].add_nodes_from(nd_list) 
		G[t].add_edges_from(list(A[t].edges()))
		all_edges.extend(list(A[t].edges())) 
	all_edges = list(set(all_edges))  

	anomalous_edges = []
	thrsh = int(thrsh0 * len(all_edges))
	print('='*65)
	print('thrsh:', thrsh, len(all_edges))   
	theta_gt_z = np.zeros_like(B_aggr)  
	zero_indices = np.where(B_aggr == 0)  
	selected_indices = np.random.choice(len(zero_indices[0]), thrsh, replace=False)  
	theta_gt_z[zero_indices[0][selected_indices], zero_indices[1][selected_indices]] = 1 

	injection_steps = 2
	t1 = TT - injection_steps - 1
	non_zero_indices = np.nonzero(theta_gt_z)
	num_non_zero = len(non_zero_indices[0])
	part_size = num_non_zero // injection_steps

	edges_all = []

	for step in range(injection_steps):
		part_indices = (
			non_zero_indices[0][step * part_size : (step + 1) * part_size],
			non_zero_indices[1][step * part_size : (step + 1) * part_size],
		)

		edges = [(idx_d[i], idx_d[j]) for i, j in zip(part_indices[0], part_indices[1])]
		G[t1 + step].add_edges_from(edges, weight=1)
		edges_all.extend(edges)
		anomalous_edges.extend(edges)  

	print('len(anomalous_edges):', len(anomalous_edges)) 
			
	injected_graph = adjacency+ '_' + str(np.round(thrsh0,3)) + '_' + str(args.rseed)
	
	np.savez(out_folder + 'theta_gt_z_'+injected_graph, z=theta_gt_z) 
		
	np.savetxt(out_folder+injected_graph+'_anomalous_edges.dat', anomalous_edges, fmt='%s', delimiter=" ")
	outfile = out_folder+injected_graph+'_injected.dat' 


	### Network post-processing 
	assert len(nodes) == N 
	A1 = [nx.to_scipy_sparse_array(G[t], nodelist=nodes, weight='weight') for t in range(len(G))]


	# Keep largest connected component
	A_sum = A1[0].copy()
	for t in range(1,len(A1)): A_sum += A1[t] 
	cv_functions.output_adjacency(nodes,A_sum,A1,outfile=outfile) 


	print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')
	# print(f'anomalous network saved in: {out_folder}')

if __name__ == '__main__':
	main()
