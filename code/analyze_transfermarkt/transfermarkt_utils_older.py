'''
Utils to process results of anomaly detection on transfermarkt data
'''

from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple, Union
import osmnx as ox
from os.path import isfile
from collections import defaultdict
import json

import matplotlib.pyplot as plt
import contextily as ctx

import networkx as nx

import geopandas as gpd

map_colors = {0: '#C998D7', 1: '#85C98D', 2: '#F8C38A', 3: '#846C5B', 4:'#95CBC7', 5:'#F5D058',6:'#E74E4E', 7:'#CCA5A5', 8:'#6359BE', 9:'#E868B5'}

# import warnings
# warnings.simplefilter(action='ignore', category=DeprecationWarning)
def import_network(
    end_file: str = '.dat',
    pref_ad: str = 'adjacency_20',
    suff_ad: str = '_binary',
    time_span: str = '08_23',
    infolder: str = '../data/input/real_data/transfermarket/'
)-> pd.DataFrame:

    dataset = f"{pref_ad}{time_span}{suff_ad}"
    infile = f"{infolder}{dataset}{end_file}"
    return pd.read_csv(infile, index_col=None, header=0, sep='\s+', comment='#',
                     usecols=['source', 'target', 'weight_t0', 'weight_t1', 'weight_t2', 'weight_t3',
                              'weight_t4'])  # \s+

def import_inferred_params(
    pref_ad: str = 'adjacency_20',
    suff_ad: str = '_binary',
    time_span: str = '08_23',
    K: int = 8,
    infolder: str = '../data/input/real_data/transfermarket/'
)-> dict:

    # infile_params = f"{infolder}theta_{pref_ad}{time_span}{suff_ad}_K{K}_cv.npz"
    infile_params = f"{infolder}theta_{pref_ad}{time_span}{suff_ad}_MT.npz"
    return np.load(infile_params, allow_pickle=True)

def import_inferred_params_cd(
    flag_anomaly: bool = False,
    pref_ad: str = 'adjacency_20',
    suff_ad: str = '_binary',
    time_span: str = '08_23',
    K: int = 8,
    infolder: str = '../data/output/5-fold_cv/real_data/transfermarket/'
)-> Tuple[dict,str]:
    '''
    `Dyn_ACD` with `flag_anomaly == False`
    ''' 
    dataset = f"{pref_ad}{time_span}{suff_ad}"
    infile_params = f"{infolder}theta_{dataset}_{K}_{flag_anomaly}_ACD_Wdynamic.npz"
    return (np.load(infile_params, allow_pickle=True),infile_params)

def import_metadata(
    infolder: str = '../data/input/real_data/transfermarket/'
    )-> pd.DataFrame:

    input_meta_in_data = pd.read_csv(f"{infolder}income.csv", index_col=None, header=0, sep=',')
    input_meta_out_data = pd.read_csv(f"{infolder}outcome.csv", index_col=None, header=0, sep=',')

    df_meta = input_meta_in_data.merge(input_meta_out_data, left_on='Alter', right_on='Ego',
                                       suffixes=['_fee_spent', '_fee_received'], validate='one_to_one')
    df_meta = df_meta.drop(columns=['Ego']).rename(
        columns={'Alter': 'node', 'weight_fee_spent': 'fee_spent', 'weight_fee_received': 'fee_received'})
    df_meta.loc[:,'spent_over_received'] = df_meta['fee_spent'] / df_meta['fee_received']

    df_meta = df_meta.sort_values(by=['fee_spent','spent_over_received'], ascending=[False,False])

    '''
    Merge with GPS coordinates for plotting
    '''
    with open(f'{infolder}clubs_and_cities.json', 'r') as file:
        clubs_and_cities = json.load(file)

    with open(f'{infolder}clubs_and_cities_coordinates.json', 'r') as file:
        clubs_and_cities_coordinates = json.load(file)

    team_names = [x.replace(' ','_') for x in clubs_and_cities.keys()]
    df_coord = pd.DataFrame({'team_name': team_names, 'team_city': clubs_and_cities.values()})
    df_coord.loc[:, 'coordinates'] = df_coord['team_city'].map(clubs_and_cities_coordinates).fillna(0)
    df_coord.loc[:, 'lat'] = df_coord['coordinates'].map(lambda x: x[0])
    df_coord.loc[:, 'lon'] = df_coord['coordinates'].map(lambda x: x[1])

    df_meta = df_meta.merge(df_coord, left_on='node',right_on='team_name', how='outer').drop(columns=['team_name'])
    return df_meta

def get_transfer(df: pd.DataFrame, source: str, target: str = None, both_directions: bool = False,
                 source_label: str = 'source',target_label: str = 'target') -> pd.DataFrame:
    '''
    Get the transfers between two teams
    '''
    if target is None:
        cond1 = df[source_label] == source
        cond2 = df[target_label] == source
        # mask = cond1 | cond2
        df_out = df[cond1].sort_values(
            by=['weight_t0', 'weight_t1', 'weight_t2', 'weight_t3', 'weight_t4', source_label, target_label],
            ascending=[False, False, False, False, False, True, True])
        df_in = df[cond2].sort_values(
            by=['weight_t0', 'weight_t1', 'weight_t2', 'weight_t3', 'weight_t4', source_label, target_label],
            ascending=[False, False, False, False, False, True, True])
        return pd.concat([df_out,df_in])
        # return df[mask].sort_values(
        #     by=['weight_t0', 'weight_t1', 'weight_t2', 'weight_t3', 'weight_t4', source_label, target_label],
        #     ascending=[False, False, False, False, False, True, True])

    cond1 = df[source_label] == source
    cond2 = df[target_label] == target
    mask = cond1 & cond2

    if both_directions == False:
        return df[mask]

    else:
        cond3 = df[target_label] == source
        cond4 = df[source_label] == target
        mask = mask | (cond3 & cond4)
        return df[mask]

def get_df_anomalies(
    QIJ_dense: np.ndarray,
    index_to_name: dict,
    lambdaIJ: np.ndarray = None,
    mask: np.ndarray = None,
    df_meta: pd.DataFrame = None,
    threshold: float = 0.01
)-> pd.DataFrame:

    if mask is None:
        mask = np.ones_like(QIJ_dense).astype(bool)

    cols = ['ego_id', 'alter_id', 'Q_ij']
    Q_sub_nz = (QIJ_dense * mask).nonzero()
    data_Q = list(zip(Q_sub_nz[0], Q_sub_nz[1], (QIJ_dense * mask)[Q_sub_nz]))
    df_anomalies = pd.DataFrame(data_Q, columns=cols)
    df_anomalies.loc[:, 'ego_name'] = df_anomalies['ego_id'].map(index_to_name)
    df_anomalies.loc[:, 'alter_name'] = df_anomalies['alter_id'].map(index_to_name)

    if lambdaIJ is not None:
        df_anomalies.loc[:, 'lambda_ij'] = lambdaIJ[Q_sub_nz]

    '''
    Add metadata weight
    '''
    if df_meta is not None:
        df_anomalies = df_anomalies.merge(df_meta[['node','fee_received']], left_on=['ego_name'], right_on='node')
        df_anomalies.rename(columns={'fee_received': 'ego_fee_received'}, inplace=True)
        df_anomalies.drop(columns=['node'], inplace=True)
        df_anomalies = df_anomalies.merge(df_meta[['node','fee_received']], left_on=['alter_name'], right_on='node')
        df_anomalies.rename(columns={'fee_received': 'alter_fee_received'}, inplace=True)
        df_anomalies.drop(columns=['node'], inplace=True)
        df_anomalies = df_anomalies.merge(df_meta[['node','fee_spent']], left_on=['ego_name'], right_on='node')
        df_anomalies.rename(columns={'fee_spent': 'ego_fee_spent'}, inplace=True)
        df_anomalies.drop(columns=['node'], inplace=True)
        df_anomalies = df_anomalies.merge(df_meta[['node','fee_spent']], left_on=['alter_name'], right_on='node')
        df_anomalies.rename(columns={'fee_spent': 'alter_fee_spent'}, inplace=True)
        df_anomalies.drop(columns=['node'], inplace=True)

    df_anomalies = df_anomalies.sort_values(by=['Q_ij', 'alter_fee_spent', 'ego_fee_spent'],
                                            ascending=[False, False, False]).reset_index(drop=True)

    # mask = df_anomalies['Q_ij'] >= threshold

    return df_anomalies

def get_unique_anomalous_edges(df_anomalies: pd.DataFrame,threshold: float = 0.01,
                               ego_label: str = 'ego_id',alter_label: str = 'alter_id')-> list:

    mask = df_anomalies['Q_ij'] > threshold
    unique_edges = list(set([tuple(sorted(edge)) for edge in set(df_anomalies[mask].groupby(by=[ego_label,alter_label]).groups.keys())]))
    return unique_edges

def is_regular(source_node,target_node, anomalous_edges):
    if (source_node, target_node) in anomalous_edges:
        return False
    else:
        return True

def extract_bridge_properties(i, color, U, threshold=0.):
    groups = np.where(U[i] > threshold)[0]
    wedge_sizes = U[i][groups]
    wedge_colors = [color[c] for c in groups]
    return wedge_sizes, wedge_colors


def get_Qij_from_df_anomalies(df: pd.DataFrame, i: str, j: str):
    cond1 = df['ego_name'] == i
    cond2 = df['alter_name'] == j
    mask = cond1 & cond2

    assert np.sum(mask) <= 1

    return df[mask]['Q_ij'].iloc[0]

def assign_pos_based_on_membership(nodes:list, membership: np.ndarray, nodeName2Id:dict,delta=0.5)-> dict:
    groups = np.argmax(membership,axis=1)
    C = max(np.unique(groups)) +1
    pos_groups = nx.circular_layout(nx.cycle_graph(C))
    pos = {}
    for c, i in enumerate(nodes):
        r = np.random.rand() * 2 * np.math.pi
        radius = np.random.rand()
        pos[i] = pos_groups[groups[nodeName2Id[i]]] + delta * radius * np.array(
            [np.math.cos(r), np.math.sin(r)])
    return pos


def plotting_map(
        graph: nx.Graph,
        pos: dict,
        map_colors: dict = {0: '#C998D7', 1: '#85C98D', 2: '#F8C38A', 3: '#846C5B', 4: '#95CBC7', 5: '#F5D058',
                            6: '#E74E4E', 7: '#CCA5A5', 8: '#6359BE', 9: '#E868B5'},
        U: np.ndarray = None,
        node_size: np.ndarray = None,
        alpha: float = 0.8,
        edge_width: Union[float, list] = 0.75,
        nodelist: list = None,
        edgelist: list = None,
        edge_color: Union[str, list] = 'grey',
        node_size_factor: float = 0.001,
        edge_width_factor: float = None,
        figsize: tuple = (15, 10),
        ax: plt.Axes = None,
        with_labels: bool = True,
        plot_map: bool = False,
        map_boundaries: tuple = (-15, 35, 37, 55),
        zoom: int = 5,
        **kw_args
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if nodelist is None:
        nodelist = list(graph.nodes())

    if edgelist is None:
        edgelist = list(graph.edges())

    if node_size is None:
        node_size = np.array([graph.out_degree[i] for i in nodelist]) + 0.1

    if edge_width_factor is not None:
        edge_width = [w * edge_width_factor for w in edge_width]

    nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=edge_width, alpha=alpha, node_size=0, ax=ax,
                           edge_color=edge_color, arrows=True, arrowsize=1, style='--', connectionstyle="arc3,rad=0.2")
    nx.draw_networkx(graph, pos, nodelist=nodelist, node_size=0, edgelist=[], with_labels=False, ax=ax)

    if with_labels:
        nx.draw_networkx_labels(graph, pos, font_size=10, alpha=0.8, horizontalalignment='right', ax=ax)

    if U is not None:
        for i, n in enumerate(graph.nodes()):
            if n in nodelist:
                wedge_sizes, wedge_colors = extract_bridge_properties(i, map_colors, U)
                if len(wedge_sizes) > 0:
                    _ = ax.pie(wedge_sizes, center=pos[n], colors=wedge_colors,
                               radius=(node_size[i]) * node_size_factor, normalize=True)
                ax.axis("equal")
    if plot_map is True:
        # extent = (-15, 35, 37, 55)
        #     extent = (-25, 40, 15, 55)
        #     extent = (-10, 50, 30, 55)  # Adjusted extent for Western Europe
        #     extent = (-15, 35, 37, 55)  # Adjusted extent for Western Europe
        ax.axis(map_boundaries)
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, zoom=zoom)

        # plt.axis('off')
    #     plt.savefig(f'../figures/real_data/transfermarket/map_europe_2014-2023', facecolor='white', edgecolor='none', dpi=300)
    # plt.show()
    plt.tight_layout()
