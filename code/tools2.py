"""Functions for handling the data."""
import networkx as nx
import numpy as np
import pandas as pd
import sktensor as skt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import random as rdm

sns.set_style('white')


def import_data(dataset, L=1, undirected=False, ego='source', alter='target', force_dense=True, noselfloop=True, verbose=True, header=0,sep=',',
				binary=False):
	"""
		Import data, i.e. the adjacency tensor, from a given folder.
		Return the NetworkX graph and its numpy adjacency tensor.
		Parameters
		----------
		dataset : str
				  Path of the input file.
		undirected : bool
					 If set to True, the algorithm considers an undirected graph.
		ego : str
			  Name of the column to consider as source of the edge.
		alter : str
				Name of the column to consider as target of the edge.
		force_dense : bool
					  If set to True, the algorithm is forced to consider a dense adjacency tensor.
		noselfloop : bool
					 If set to True, the algorithm removes the self-loops.
		verbose : bool
				  Flag to print details.
		binary : bool
				 Flag to force the matrix to be binary.
		Returns
		-------
		A : list
			List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
		B : ndarray
			Graph adjacency tensor.
		nodes : list
				List of nodes IDs.
	"""
	# read adjacency file
	df_adj = pd.read_csv(dataset, sep=sep, header = header)
	print('{0} shape: {1}'.format(dataset, df_adj.shape))
	# create the graph adding nodes and edges
	A = read_graph(df_adj=df_adj, L =L, ego=ego, alter=alter, undirected=undirected, noselfloop=noselfloop, verbose=verbose,
				   binary=binary)
	nodes = list(A[0].nodes)
	print('\nNumber of nodes =', len(nodes))
	print('Number of layers =', len(A))
	if verbose:
		print_graph_stat(A)
	# save the multilayer network in a tensor with all layers
	if force_dense:
		B, rw = build_B_from_A(A, nodes=nodes)
		B_T, data_T_vals = None, None
	else:
		B, B_T, data_T_vals, rw = build_sparse_B_from_A(A)
	return A, B, B_T, data_T_vals


def read_graph(df_adj, L=1, ego='source', alter='target', undirected=False, noselfloop=True, verbose=True, binary=False):
	"""
		Create the graph by adding edges and nodes.
		Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
		Parameters
		----------
		df_adj : DataFrame
				 Pandas DataFrame object containing the edges of the graph.
		ego : str
			  Name of the column to consider as source of the edge.
		alter : str
				Name of the column to consider as target of the edge.
		undirected : bool
					 If set to True, the algorithm considers an undirected graph.
		noselfloop : bool
					 If set to True, the algorithm removes the self-loops.
		verbose : bool
				  Flag to print details.
		binary : bool
				 If set to True, read the graph with binary edges.
		Returns
		-------
		A : list
			List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
	"""
	# build nodes
	egoID = df_adj[ego].unique()
	alterID = df_adj[alter].unique()
	nodes = list(set(egoID).union(set(alterID)))
	nodes.sort()
	# build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers
	if undirected:
		A = [nx.MultiGraph() for _ in range(L)]
	else:
		A = [nx.MultiDiGraph() for _ in range(L)]

	if verbose:
		print('Creating the network ...', end=' ')
	# set the same set of nodes and order over all layers
	for l in range(L):
		A[l].add_nodes_from(nodes)  
	
	# ONLY APPLICABLE FOR DYNAMIC TRANSFERMARKT DATASET
	for index, row in df_adj.iterrows():
		v1 = row[ego]
		v2 = row[alter]
		for l in range(1):   
			if row[l + 3] > 0:
				if binary:
					if A[l].has_edge(v1, v2):
						A[l][v1][v2][0]['weight'] = 1
					else:
						A[l].add_edge(v1, v2, weight=1)
				else:
					if A[l].has_edge(v1, v2):
						A[l][v1][v2][0]['weight'] += int(row[l + 3])  # the edge already exists, no parallel edge created
					else:
						A[l].add_edge(v1, v2, weight=int(row[l + 3]))
	if verbose:
		print('done!')

	if verbose:
		print('done!')
	# remove self-loops
	if noselfloop:
		if verbose:
			print('Removing self loops')
		for l in range(L):
			A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))
	return A



def print_graph_stat(G):
	"""
		Print the statistics of the graph A.
		Parameters
		----------
		G : list
			List of MultiDiGraph NetworkX objects.
	"""
	L = len(G)
	N = G[0].number_of_nodes()
	print('Number of edges and average degree in each layer:')
	for l in range(L):
		E = G[l].number_of_edges()
		k = 2 * float(E) / float(N)
		print(f'E[{l}] = {E} - <k> = {np.round(k, 3)}')
		weights = [d['weight'] for u, v, d in list(G[l].edges(data=True))]
		if not np.array_equal(weights, np.ones_like(weights)):
			M = np.sum([d['weight'] for u, v, d in list(G[l].edges(data=True))])
			kW = 2 * float(M) / float(N)
			print(f'M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}')
		print(f'Sparsity [{l}] = {np.round(E / (N * N), 3)}')
		print(f'Reciprocity (networkX) = {np.round(nx.reciprocity(G[l]), 3)}')
		print(f'Reciprocity (intended as the proportion of bi-directional edges over the unordered pairs) = '
			  f'{np.round(reciprocal_edges(G[l]), 3)}\n')


def build_B_from_A(A, nodes=None):
	"""
		Create the numpy adjacency tensor of a networkX graph.
		Parameters
		----------
		A : list
			List of MultiDiGraph NetworkX objects.
		nodes : list
				List of nodes IDs.
		Returns
		-------
		B : ndarray
			Graph adjacency tensor.
		rw : list
			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
	"""
	N = A[0].number_of_nodes()
	if nodes is None:
		nodes = list(A[0].nodes())
	B = np.empty(shape=[len(A), N, N])
	rw = []
	for l in range(len(A)):
		B[l, :, :] = nx.to_numpy_matrix(A[l], weight='weight', dtype=int, nodelist=nodes)
		rw.append(np.multiply(B[l], B[l].T).sum() / B[l].sum())
	return B, rw


def build_sparse_B_from_A(A):
	"""
		Create the sptensor adjacency tensor of a networkX graph.
		Parameters
		----------
		A : list
			List of MultiDiGraph NetworkX objects.
		Returns
		-------
		data : sptensor
			   Graph adjacency tensor.
		data_T : sptensor
				 Graph adjacency tensor (transpose).
		v_T : ndarray
			  Array with values of entries A[j, i] given non-zero entry (i, j).
		rw : list
			 List whose elements are reciprocity (considering the weights of the edges) values, one per each layer.
	"""
	N = A[0].number_of_nodes()
	L = len(A)
	rw = []
	d1 = np.array((), dtype='int64')
	d2, d2_T = np.array((), dtype='int64'), np.array((), dtype='int64')
	d3, d3_T = np.array((), dtype='int64'), np.array((), dtype='int64')
	v, vT, v_T = np.array(()), np.array(()), np.array(())
	for l in range(L):
		b = nx.to_scipy_sparse_matrix(A[l])
		b_T = nx.to_scipy_sparse_matrix(A[l]).transpose()
		rw.append(np.sum(b.multiply(b_T)) / np.sum(b))
		nz = b.nonzero()
		nz_T = b_T.nonzero()
		d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
		d2 = np.hstack((d2, nz[0]))
		d2_T = np.hstack((d2_T, nz_T[0]))
		d3 = np.hstack((d3, nz[1]))
		d3_T = np.hstack((d3_T, nz_T[1]))
		v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
		vT = np.hstack((vT, np.array([b_T[i, j] for i, j in zip(*nz_T)])))
		v_T = np.hstack((v_T, np.array([b[j, i] for i, j in zip(*nz)])))
	subs_ = (d1, d2, d3)
	subs_T_ = (d1, d2_T, d3_T)
	data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)
	data_T = skt.sptensor(subs_T_, vT, shape=(L, N, N), dtype=vT.dtype)
	return data, data_T, v_T, rw


def reciprocal_edges(G):
	"""
		Compute the proportion of bi-directional edges, by considering the unordered pairs.
		Parameters
		----------
		G: MultiDigraph
		   MultiDiGraph NetworkX object.
		Returns
		-------
		reciprocity: float
					 Reciprocity value, intended as the proportion of bi-directional edges over the unordered pairs.
	"""
	n_all_edge = G.number_of_edges()
	n_undirected = G.to_undirected().number_of_edges()  # unique pairs of edges, i.e. edges in the undirected graph
	n_overlap_edge = (n_all_edge - n_undirected)  # number of undirected edges reciprocated in the directed network
	if n_all_edge == 0:
		raise nx.NetworkXError("Not defined for empty graphs.")
	reciprocity = float(n_overlap_edge) / float(n_undirected)
	return reciprocity


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


def transpose_tensor(M):
	"""
		Given M tensor, it returns its transpose: for each dimension a, compute the transpose ij->ji.
		Parameters
		----------
		M : ndarray
			Tensor with the mean lambda for all entries.
		Returns
		-------
		Transpose version of M_aij, i.e. M_aji.
	"""
	return np.einsum('aij->aji', M)


def expected_computation(B, U, V, W, eta):
	"""
		Return the marginal and conditional expected value.
		Parameters
		----------
		B : ndarray
			Graph adjacency tensor.
		U : ndarray
			Out-going membership matrix.
		V : ndarray
			In-coming membership matrix.
		W : ndarray
			Affinity tensor.
		eta : float
			  Pair interaction coefficient.
		Returns
		-------
		M_marginal : ndarray
					 Marginal expected values.
		M_conditional : ndarray
						Conditional expected values.
	"""
	lambda0_aij = lambda0_full(U, V, W)
	L = lambda0_aij.shape[0]
	Z = calculate_Z(lambda0_aij, eta)
	M_marginal = (lambda0_aij + eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z
	for l in np.arange(L):
		np.fill_diagonal(M_marginal[l], 0.)
	M_conditional = (eta ** transpose_tensor(B) * lambda0_aij) / (eta ** transpose_tensor(B) * lambda0_aij + 1)
	for l in np.arange(L):
		np.fill_diagonal(M_conditional[l], 0.)
	return M_marginal, M_conditional


def lambda0_full(u, v, w):
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


def calculate_Z(lambda0_aij, eta):
	"""
		Compute the normalization constant of the Bivariate Bernoulli distribution.
		Returns
		-------
		Z : ndarray
			Normalization constant Z of the Bivariate Bernoulli distribution.
	"""
	Z = lambda0_aij + transpose_tensor(lambda0_aij) + eta * np.einsum('aij,aji->aij', lambda0_aij, lambda0_aij) + 1
	for l in range(len(Z)):
		assert check_symmetric(Z[l])
	return Z


def check_symmetric(a, rtol=1e-05, atol=1e-08):
	"""
		Check if a matrix a is symmetric.
	"""
	return np.allclose(a, a.T, rtol=rtol, atol=atol)


def compute_M_joint(U, V, W, eta):
	"""
		Return the vectors of joint probabilities of every pair of edges.
		Parameters
		----------
		U : ndarray
			Out-going membership matrix.
		V : ndarray
			In-coming membership matrix.
		W : ndarray
			Affinity tensor.
		eta : float
			  Pair interaction coefficient.
		Returns
		-------
		[p00, p01, p10, p11] : list
							   List of ndarray with joint probabilities of having no edges, only one edge in one
							   direction and both edges for every pair of edges.
	"""
	lambda0_aij = lambda0_full(U, V, W)
	Z = calculate_Z(lambda0_aij, eta)
	p00 = 1 / Z
	p10 = lambda0_aij / Z
	p01 = transpose_tensor(p10)
	p11 = (eta * lambda0_aij * transpose_tensor(lambda0_aij)) / Z
	return [p00, p01, p10, p11]


# Utils to visualize the data
def plot_hard_membership(graph, communities, pos, colors, edge_color, N_th, E_th, labels=False, weighted=False):
	"""
		Plot a graph with nodes colored by their hard memberships.
		weighted: True only if there is a defined weight value in the network
		N_th: number of nodes that should be displayed
		E_th: number of edges that should be displayed
		labels: True only if node_labels should be displayed
	"""

	# Defining the level of details for nodes plot
	if N_th is None:
		N_th = graph.number_of_nodes()  # number of nodes to plot
	else:
		N_th = min(N_th, graph.number_of_nodes())

	# Getting the N_th nodes with the highest degree and creating a sorter that will be
	# used to sort the list of nodes and communities.
	sort_n = sorted(enumerate(list(graph.degree)), key=lambda x: x[1][1], reverse=True)
	sort_n = np.array(list(zip(*sort_n))[0])
	nodes_sub = np.array(graph.nodes())[sort_n]
	nodes_sub = nodes_sub[:N_th]
	# Asserting if sort process is correct

	# Generating new subgraph containing only the N_th more relevant nodes
	graph_Nth = graph.subgraph(nodes_sub)

	# Defining the level of details for edges plot
	if E_th is None:
		E_th = graph_Nth.number_of_edges()  # number of edges to plot
	else:
		E_th = min(E_th, graph_Nth.number_of_edges())

	# Sorting communities to match the new node list
	communities_sorted = {'u': communities['u'][sort_n][:N_th], 'v': communities['v'][sort_n][:N_th]}
	# Asserting if sort process is correct

	# Adjusting node_size parameter
	node_size_new = [graph.degree[i] * 15 for i in nodes_sub]

	if weighted:
		# Getting E_th edges with biggest weight
		edges_sorted_weights = dict(
			sorted(nx.get_edge_attributes(graph_Nth, 'weight').items(), key=lambda x: x[1], reverse=True))
		edges_sub = list(edges_sorted_weights.keys())
		edges_sub = edges_sub[:E_th]

		# Adjusting edge_size parameter
		edge_size_new = list(edges_sorted_weights.values())[:E_th]
	else:
		# Getting a random sample of E_th edges to plot
		rdm.seed(1)
		edges_sub = rdm.sample(list(graph_Nth.edges()), E_th)
		edge_size_new = 0.4

	# Generate the plot for u and v
	plt.figure(figsize=(20, 10))
	for i, k in enumerate(communities_sorted):
		fig = plt.subplot(1, 2, i + 1)
		nx.draw_networkx_edges(graph_Nth, pos, edgelist=edges_sub, width=edge_size_new, edge_color=edge_color,
							   connectionstyle="arc3,rad=0.2", arrows=True, arrowsize=5)
		nx.draw_networkx(graph_Nth, pos, node_size=node_size_new, edgelist=[],
						 node_color=[colors[node] for node in communities_sorted[k]],
						 with_labels=labels)
		plt.title(k, fontsize=20)
		plt.axis('off')
	plt.tight_layout()
	plt.show()


def extract_bridge_properties(i, color, U, threshold=0.):
	groups = np.where(U[i] > threshold)[0]
	wedge_sizes = U[i][groups]
	wedge_colors = [color[c] for c in groups]
	return wedge_sizes, wedge_colors


def plot_soft_membership(graph, thetas, pos, node_size, colors, edge_color, N_th, E_th,
						 labels=False, weighted=False,mult_size_factor=0.0002,label= None):
	"""
		Plot a graph with nodes colored by their mixed (soft) memberships.
	"""
	# Defining the level of details for nodes plot
	if N_th is None:
		N_th = graph.number_of_nodes()  # number of nodes to plot
	else:
		N_th = min(N_th, graph.number_of_nodes())

	# Getting the N_th nodes with the highest degree and creating a sorter that will be
	# used to sort the list of nodes and communities.
	sort = sorted(enumerate(list(graph.degree)), key=lambda x: x[1][1], reverse=True)
	sort = np.array(list(zip(*sort))[0])
	nodes_sub = np.array(graph.nodes())[sort]
	nodes_sub = nodes_sub[:N_th]

	# Generating new subgraph containing only the N_th more relevant nodes
	graph_Nth = graph.subgraph(nodes_sub)

	# Defining the level of details for edges plot
	if E_th is None:
		E_th = graph_Nth.number_of_edges()  # number of edges to plot
	else:
		E_th = min(E_th, graph_Nth.number_of_edges())

	if weighted:
		# Getting E_th edges with biggest weight
		edges_sorted_weights = dict(sorted(nx.get_edge_attributes(graph_Nth, 'weight').items(),
										   key=lambda x: x[1], reverse=True))
		edges_sub = list(edges_sorted_weights.keys())
		edges_sub = edges_sub[:E_th]

		# Adjusting edge_size parameter
		edge_size_new = list(edges_sorted_weights.values())[:E_th]
	else:
		# Getting a random sample of E_th edges to plot
		rdm.seed(1)
		edges_sub = rdm.sample(list(graph_Nth.edges()), E_th)
		edge_size_new = 0.1

	# Sorting mixed communities to match the new node list
	thetas_sorted = {'u': thetas['u'][sort][:N_th]}

	# Adjusting node_size parameter
	node_size_new = [graph.degree[i] * 6 for i in nodes_sub]

	fig = plt.figure(figsize=(10, 10))

	for j,k in enumerate(thetas_sorted):
		nx.draw_networkx_edges(graph_Nth, pos, edgelist=edges_sub, width=edge_size_new,
							edge_color=edge_color, arrows=True, arrowsize=5,
							connectionstyle="arc3,rad=0.2")
		nx.draw_networkx(graph_Nth, pos, nodelist=nodes_sub, node_size=0, with_labels=labels,
						edgelist=[])
		for i, n in enumerate(graph_Nth.nodes()):
			wedge_sizes, wedge_colors = extract_bridge_properties(i, colors, thetas_sorted[k])
			if len(wedge_sizes) > 0:
				_ = plt.pie(wedge_sizes, center=pos[n], colors=wedge_colors,
							radius=(node_size_new[i]) * mult_size_factor,
							normalize=True)
		# plt.title(k, fontsize=17)
	
	# plt.tight_layout() 
	plt.axis('off')
	# plt.show()
	plt.savefig(f'../figures/real_data/transfermarket/{label}', facecolor='white', edgecolor='none', dpi=300)


def plot_adjacency(Bd, M_marginal, M_conditional, nodes, cm='Purples'):
	"""
			Plot the adjacency matrix and its reconstruction given by the marginal and the conditional expected values.
	"""
	# For a cleaner visualization when the number of nodes is too big
	if len(nodes()) <= 60:
		labels = nodes()
		font_size = 10
	elif 60 < len(nodes()) <= 200:
		labels = nodes()
		font_size = 4
	else:
		labels = []
		font_size = 0

	sns.set_style('ticks')
	fig = plt.figure(figsize=(24, 48))
	gs = gridspec.GridSpec(3, 2, width_ratios=(1.0, 0.01))

	plt.subplot(gs[0, 0])
	im = plt.imshow(Bd[0], vmin=0, vmax=1, cmap=cm)
	plt.xticks(ticks=np.arange(len(nodes)), labels=labels, fontsize=font_size, rotation=55, ha='right')
	plt.yticks(ticks=np.arange(len(nodes)), labels=labels, fontsize=font_size)
	plt.title('Data', fontsize=17)
	axes = plt.subplot(gs[0, 1])
	cbar = plt.colorbar(im, cax=axes)
	cbar.ax.tick_params(labelsize=15)

	plt.subplot(gs[1, 0])
	plt.imshow(M_marginal[0], vmin=0, vmax=1, cmap=cm)
	plt.xticks(ticks=np.arange(len(nodes)), labels=labels, fontsize=font_size, rotation=55, ha='right')
	plt.yticks(ticks=np.arange(len(nodes)), labels=labels, fontsize=font_size)
	plt.title(r'$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$', fontsize=17)
	axes = plt.subplot(gs[1, 1])
	cbar = plt.colorbar(im, cax=axes)
	cbar.ax.tick_params(labelsize=15)

	plt.subplot(gs[2, 0])
	plt.imshow(M_conditional[0], vmin=0, vmax=1, cmap=cm)
	plt.xticks(ticks=np.arange(len(nodes)), labels=labels, fontsize=font_size, rotation=55, ha='right')
	plt.yticks(ticks=np.arange(len(nodes)), labels=labels, fontsize=font_size)
	plt.title(r'$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$', fontsize=17)
	axes = plt.subplot(gs[2, 1])
	cbar = plt.colorbar(im, cax=axes)
	cbar.ax.tick_params(labelsize=15)

	plt.tight_layout()
	plt.show()


def mapping(G, A):
	old = list(G.nodes)
	new = list(A.nodes)
	mapping = {}
	for x in old:
		mapping[x] = new[x]
	return nx.relabel_nodes(G, mapping)


def plot_graph(graph, M_marginal, M_conditional, pos, node_size, node_color, edge_color, threshold=0.2):
	"""
		Plot a graph and its reconstruction given by the marginal and the conditional expected values.
	"""
	plt.figure(figsize=(15, 5))
	gs = gridspec.GridSpec(1, 3)
	plt.subplot(gs[0, 0])
	edgewidth = [d['weight'] for (u, v, d) in graph.edges(data=True)]
	nx.draw_networkx(graph, pos, node_size=node_size, node_color=node_color, connectionstyle="arc3,rad=0.2",
					 with_labels=False, width=edgewidth, edge_color=edge_color, arrows=True, arrowsize=5,
					 font_size=15, font_color="black")
	plt.axis('off')
	plt.title('Data', fontsize=17)
	mask = M_marginal[0] < threshold
	M = M_marginal[0].copy()
	M[mask] = 0.
	G = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
	G = mapping(G, graph)
	edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
	plt.subplot(gs[0, 1])
	nx.draw_networkx(G, pos, node_size=node_size, node_color=node_color, connectionstyle="arc3,rad=0.2",
					 with_labels=False, width=edgewidth, edge_color=edgewidth,
					 edge_cmap=plt.cm.Greys, edge_vmin=0, edge_vmax=1, arrows=True, arrowsize=5)
	plt.axis('off')
	plt.title(r'$\mathbb{E}_{P(A_{ij} | \Theta)}[A_{ij}]$', fontsize=17)
	mask = M_conditional[0] < threshold
	M = M_conditional[0].copy()
	M[mask] = 0.
	G = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
	G = mapping(G, graph)
	edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
	plt.subplot(gs[0, 2])
	nx.draw_networkx(G, pos, node_size=node_size, node_color=node_color, connectionstyle="arc3,rad=0.2",
					 with_labels=False, width=edgewidth, edge_color=edgewidth,
					 edge_cmap=plt.cm.Greys, edge_vmin=0, edge_vmax=1, arrows=True, arrowsize=5)
	plt.axis('off')
	plt.title(r'$\mathbb{E}_{P(A_{ij} | A_{ij}, \Theta)}[A_{ij}]$', fontsize=17)
	plt.tight_layout()
	plt.show()


def plot_precision_recall(conf_matrix, cm='Purples'):
	"""
		Plot precision and recall of a given confusion matrix.
	"""
	plt.figure(figsize=(10, 5))
	# normalized by row
	gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
	plt.subplot(gs[0, 0])
	im = plt.imshow(conf_matrix / np.sum(conf_matrix, axis=1)[:, np.newaxis], cmap=cm, vmin=0, vmax=1)
	plt.xticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
	plt.yticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
	plt.ylabel('True', fontsize=15)
	plt.xlabel('Predicted', fontsize=15)
	plt.title('Precision', fontsize=17)
	# normalized by column
	plt.subplot(gs[0, 1])
	plt.imshow(conf_matrix / np.sum(conf_matrix, axis=0)[np.newaxis, :], cmap=cm, vmin=0, vmax=1)
	plt.xticks([0, 1, 2, 3], labels=[(0, 0), (0, 1), (1, 0), (1, 1)], fontsize=13)
	plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
					labelleft=False)
	plt.xlabel('Predicted', fontsize=15)
	plt.title('Recall', fontsize=17)
	axes = plt.subplot(gs[0, 2])
	plt.colorbar(im, cax=axes)
	plt.tight_layout()
	plt.show()


def reduce_to_most_relevant_nodes(graph, communities, N_th):
	if N_th is None:
		N_th = graph.number_of_nodes()  # number of nodes to plot
	else:
		N_th = min(N_th, graph.number_of_nodes())

	# Getting the N_th nodes with the highest degree and creating a sorter that will be
	# used to sort the list of nodes and communities.
	sort_n = sorted(enumerate(list(graph.degree)), key=lambda x: x[1][1], reverse=True)
	sort_n = np.array(list(zip(*sort_n))[0])
	nodes_sub = np.array(graph.nodes())[sort_n]
	nodes_sub = nodes_sub[:N_th] 

	# Generating new subgraph containing only the N_th more relevant nodes
	graph_Nth = graph.subgraph(nodes_sub)
	communities_sorted = {'u': communities['u'][sort_n][:N_th], 'v': communities['v'][sort_n][:N_th]} 

	return graph_Nth, communities_sorted
def plot_uv_aggregated(A, communities, K, type_community):
	nodesA = np.array(A.nodes())
	sort = np.argsort(communities) 
	aux_uv = [[0 for col in range(K)] for row in range(len(nodesA))]
	x_ticks = []
	for i in range(K):
		x_ticks.append(i)
	for index, community in enumerate(communities[sort]):
		aux_uv[index][community] = 1
		aux_uv_dense = np.array(aux_uv)
	sns.set_style('ticks')
	plt.figure(figsize=(15, 20))
	plt.imshow(aux_uv_dense, vmin=0, vmax=1, cmap='Purples')
	plt.yticks(ticks=np.arange(len(nodesA[sort])), labels=nodesA[sort], fontsize=10)
	plt.xticks(ticks=x_ticks, fontsize=9)
	plt.title(type_community + ' communities', fontsize=17)


def CalculatePermutation(U_infer, U0):
	"""
	Permuting the overlap matrix so that the groups from the two partitions correspond
	U0 has dimension NxK, reference memebership
	"""
	N, RANK = U0.shape
	M = np.dot(np.transpose(U_infer), U0) / float(N);  # dim=RANKxRANK
	rows = np.zeros(RANK);
	columns = np.zeros(RANK);
	P = np.zeros((RANK, RANK));  # Permutation matrix
	for t in range(RANK):
		# Find the max element in the remaining submatrix,
		# the one with rows and columns removed from previous iterations
		max_entry = 0.;
		c_index = 0;
		r_index = 0;
		for i in range(RANK):
			if columns[i] == 0:
				for j in range(RANK):
					if rows[j] == 0:
						if M[j, i] > max_entry:
							max_entry = M[j, i];
							c_index = i;
							r_index = j;

		P[r_index, c_index] = 1;
		columns[c_index] = 1;
		rows[r_index] = 1;

	return P
def cosine_similarity(U_infer, U0):
	"""
		Compute the cosine similarity between ground-truth communities and detected communities.

		Parameters
		----------
		U_infer : ndarray
				  Inferred membership matrix (detected communities), row-normalized.
		U0 : ndarray
			 Ground-truth membership matrix (ground-truth communities), row-normalized.

		Returns
		-------
		RES : float
			  Cosine similarity value.
	"""
	K1 = U_infer.shape[1]
	K2 = U0.shape[1]

	def cosine_similarity_sameK(U_infer, U0):
		P = CalculatePermutation(U_infer, U0)
		U_infer_tmp = np.dot(U_infer, P)  # permute inferred matrix
		U0_tmp = U0.copy()
		N, K = U0.shape
		cosine_sim = 0.
		norm_inf = np.linalg.norm(U_infer_tmp, axis=1)
		norm0 = np.linalg.norm(U0_tmp, axis=1)
		for i in range(N):
			if norm_inf[i] > 0.:
				U_infer_tmp[i, :] = U_infer_tmp[i, :] / norm_inf[i]
			if norm0[i] > 0.:
				U0_tmp[i, :] = U0_tmp[i, :] / norm0[i]

			cosine_sim += np.dot(U_infer_tmp[i], U0_tmp[i])
		# for k in range(K):
		# cosine_sim += np.dot(np.transpose(U_infer_tmp[:, k]), U0_tmp[:, k])
		return cosine_sim / float(N)

	if K1 == K2:
		# print(f"same K={K1}")
		return cosine_similarity_sameK(U_infer, U0)
	else:
		# print(f"K1={K1}, K2={K2}")
		if K1 < K2:
			U1 = np.zeros_like(U0)
			U1[:, :K1] = U_infer.copy()
			U2 = U0.copy()
		else:
			U1 = np.zeros_like(U_infer)
			U1[:, :K2] = U0.copy()
			U2 = U_infer.copy()
		assert U1.shape == U2.shape
		return cosine_similarity_sameK(U1, U2)

def extract_node_order(u):
	node_order = []
	N, K = u.shape
	u_max = np.argmax(u, axis=1)
	for k in range(K):
		nodes_k = [i for i in range(N) if u_max[i] == k]
		node_order.extend(nodes_k)
	assert len(node_order) == N

	return node_order
