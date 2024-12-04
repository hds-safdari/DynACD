'''
Utils to perfrom kmeans clustering on (samples,features)
Necessary to process POI on the nodes for NYC taxi data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from kneed import KneeLocator

import seaborn as sns

from .nyc_taxi_utils import get_observed_categories


import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

DEFAULT_CONFIG = {'init': 'random','n_init': 10, 'max_iter': 300, 'random_state': 10}
DEFAULT_ENCODER_NODES = 'encoder_nodes'
DEFAULT_ENCODER_ATTRIBUTES = 'encoder_attributes'

def process_features(node2attribute: dict, test: bool=True)-> Dict[np.ndarray,LabelEncoder]:
    '''
    Process raw data where features are stored in a dictionary, with categorical feature names into encoded int
    '''
    le_nodes = LabelEncoder()
    le_nodes.fit(np.array(list(node2attribute.keys())))

    le_cat = LabelEncoder()
    categories = get_observed_categories(node2attribute)
    le_cat.fit(categories)

    N = len(node2attribute)
    K = len(categories)
    X = np.zeros((N,K))
    for k,v in node2attribute.items():
        i = le_nodes.transform([k])[0]
        features_ids = np.array(le_cat.transform(list(v.index)))
        X[i,features_ids] = v.values
    feat = {'X':X, DEFAULT_ENCODER_NODES:le_nodes, DEFAULT_ENCODER_ATTRIBUTES:le_cat}
    if test:
        test_processed_features(node2attribute,feat)
    return feat

def test_processed_features(node2attribute: dict, feat: dict):
    '''
    Test to make sure we encoded correctly from dict to array
    '''
    for n in node2attribute:
        cats = feat[DEFAULT_ENCODER_ATTRIBUTES].transform(list(node2attribute[n].index)) # transform categories into int
        idx = feat[DEFAULT_ENCODER_NODES].transform([n])[0] # transform node label into int
        sorted_id = np.argsort(cats)
        if np.allclose(np.where(feat['X'][idx] > 0)[0], sorted(cats)) == False: #check that the indices of non-zero entries correspond to right categories
            raise ValueError(f"{np.where(feat['X'][idx] > 0)[0]}, {sorted(cats)}")
        if np.allclose(feat['X'][idx][feat['X'][idx].nonzero()[0]], node2attribute[n].values[sorted_id]) == False: #check that the values of features are correct
            raise ValueError(f"{feat['X'][idx][feat['X'][idx].nonzero()[0]]}, {node2attribute[n].values[sorted_id]}")

def get_kmeans_clustering(
    X: np.ndarray, # should have dimension (n_samples,n_features)
    n_clusters: int = None,
    scaling_features: bool = True,
    pca_n_components: int = None,
    ref_range: np.ndarray = np.arange(4,21),
    plot_results: bool=False,
    method: str = 'elbow',
    **extra_args
):

    # if scaling_features:
    #     scaler = StandardScaler()
    #     features = scaler.fit_transform(X)
    # else:
    #     features = np.copy(X)

    kmeans_kwargs = DEFAULT_CONFIG
    available_extra_args = {'init','n_init','max_iter','random_state'}
    for k in set(extra_args.keys()).intersection(available_extra_args):
        kmeans_kwargs[k] = extra_args[k]

    if scaling_features:
        if pca_n_components is not None:
            preprocessor = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_n_components, random_state=kmeans_kwargs['random_state']))
                    ])
        else:
            preprocessor = Pipeline([("scaler", StandardScaler())])
    else:
        preprocessor = Pipeline([])

    clusterer = Pipeline(
        [
            ('kmeans',KMeans(n_clusters = n_clusters,**kmeans_kwargs))
        ]
    )

    # if n_clusters is None:
    #     n_clusters = get_n_clusters(features,ref_range=ref_range,plot_results=plot_results,method=method, **kmeans_kwargs)
    #
    # kmeans = KMeans(
    #     n_clusters = n_clusters,
    #     **kmeans_kwargs
    # )
    # kmeans.fit(features)

    # return kmeans
    pipe = Pipeline([("preprocessor",preprocessor),("clusterer",clusterer)])
    if n_clusters is None:
        n_clusters = get_n_clusters(X,pipeline=pipe,ref_range=ref_range,plot_results=plot_results,method=method)
        pipe["clusterer"]["kmeans"].n_clusters = n_clusters

    pipe.fit(X)

    return pipe


def get_n_clusters(data: np.ndarray, ref_range: np.ndarray = np.arange(1,21),
                   method: str = 'elbow',
                   pipeline: Pipeline = None,
                   plot_results: bool=False,**kmeans_kwargs) -> int:
    '''
    Get number of cluster with elbow method
    '''
    sse = []
    if method == 'elbow':
        for k in ref_range:
            # kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            # kmeans.fit(features)
            # sse.append(kmeans.inertia_)
            pipeline["clusterer"]["kmeans"].n_clusters = k
            pipeline.fit(data)
            sse.append(pipeline["clusterer"]["kmeans"].inertia_)
        kl = KneeLocator(ref_range, sse, curve="convex", direction="decreasing")

        if plot_results:
            plt.figure()
            plt.style.use("fivethirtyeight")
            plt.plot(ref_range, sse)
            plt.xticks(ref_range)
            plt.xlabel("Number of Clusters")
            plt.ylabel("SSE")
            plt.title(f"Best k = {kl.elbow}")
            plt.show()

        return kl.elbow

    if method == 'silhouette':
        silhouette_coefficients = []
        ref_range[0] = max(ref_range[0],2)
        for k in ref_range:
            # kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            # kmeans.fit(features)
            # score = silhouette_score(features, kmeans.labels_)
            pipeline["clusterer"]["kmeans"].n_clusters = k
            pipeline.fit(data)
            score = silhouette_score(pipeline["preprocessor"].transform(data),pipeline["clusterer"]["kmeans"].labels_)
            silhouette_coefficients.append(score)

        idx = np.argmax(np.array(silhouette_coefficients))

        if plot_results:
            plt.figure()
            plt.style.use("fivethirtyeight")
            plt.plot(ref_range, silhouette_coefficients)
            plt.xticks(ref_range)
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.title(f"Best k = {ref_range[idx]}")
            plt.show()

        return ref_range[idx]

    else:
        print(f"No valid method entered. Entered {method}, available ones are:\n'elbow' and 'silhouette'\n")

def get_silhouette_score(data: np.ndarray,pipeline: Pipeline)-> float:
    return silhouette_score(pipeline["preprocessor"].transform(data), pipeline["clusterer"]["kmeans"].labels_)

def get_centroids_original_space(pipeline: Pipeline) -> np.ndarray:
    '''
    Get centroid in the original space, i.e. reverse PCA and scaler transformations
    '''
    if 'pca' in [s[0] for s in pipeline['preprocessor'].steps]:
        centroids = pipeline['preprocessor']['pca'].inverse_transform(pipeline['clusterer']['kmeans'].cluster_centers_)
    else:
        centroids = pipeline['clusterer']['kmeans'].cluster_centers_

    return pipeline['preprocessor']['scaler'].inverse_transform(centroids)

def get_centroids_dataframe(
        pipeline: Pipeline,
        encoder: LabelEncoder,
        cmap: ListedColormap = plt.cm.Set2
    ) -> pd.DataFrame:
    '''
    Get a dataframe with centroids
    '''

    centroids = get_centroids_original_space(pipeline)
    df_centroid = pd.DataFrame(data=centroids, columns=encoder.classes_)
    df_centroid = pd.concat([df_centroid.T,
                             pd.Series([cmap(x) for x in np.arange(pipeline['clusterer']['kmeans'].n_clusters)],
                                       name='color').to_frame().T], axis=0)
    return df_centroid

def get_pca_loadings(pipeline: Pipeline,feature_names: list = None, encoder: LabelEncoder = None, return_loading_matrix:bool=True) -> pd.DataFrame:
    '''
    Get loadings with names of the categories they represent
    '''
    if encoder is not None:
        feature_names = encoder.classes_

    cols = [f"PC{i}" for i in np.arange(1, pipeline['preprocessor']['pca'].components_.shape[0] + 1)]

    if return_loading_matrix == True:
        loadings = pipeline['preprocessor']['pca'].components_.T * np.sqrt(
            pipeline['preprocessor']['pca'].explained_variance_)

        return pd.DataFrame(loadings, columns=cols, index=feature_names)

    return pd.DataFrame(pipeline['preprocessor']['pca'].components_.T, columns=cols, index=feature_names)

def get_feature_contributions(
        data: np.ndarray,
        pipeline: Pipeline,
        encoder_nodes: LabelEncoder=None,
        encoder_categories: LabelEncoder=None,
        df_loadings: pd.DataFrame = None,
        label: str = 'location_id',
        normalize: bool =True,
        **kwargs
  )-> pd.DataFrame:
    '''
    Get a DataFrame with each node as row and columns the loading in each category
    For interpratation of the nodes
    '''
    if df_loadings is None:
        df_loadings = get_pca_loadings(pipeline=pipeline,encoder=encoder_categories)
    if encoder_nodes is None:
        encoder_nodes = LabelEncoder()
        encoder_nodes.fit(np.arange(data.shape[0]))
    if encoder_categories is None:
        encoder_categories = LabelEncoder()
        encoder_categories.fit(np.arange(data.shape[1]))

    Y = np.dot(pipeline["preprocessor"].transform(data), df_loadings.values.T) # has dim n_features x n_samples
    df = pd.DataFrame(Y,
                 columns=encoder_categories.classes_,
                 index=encoder_nodes.classes_).reset_index().rename(columns={'index': label}).T

    if normalize:
        return normalize_df_by_row(df)

    return df

def normalize_df_by_row(df: pd.DataFrame)-> pd.DataFrame:
    # return df.div(df.max(axis=1) - df.min(axis=1), axis=0)
    return df.div(df.sum(axis=1), axis=0)

def plot_pca(
        data: np.ndarray,
        pipeline: Pipeline,
        true_labels: np.ndarray = None,
        x_label: str = 'Component 1',
        y_label: str = 'Component 2',
        ax: plt.axis = None,
        figsize: tuple = (8, 8)
     ):

    pcadf = pd.DataFrame(
        pipeline["preprocessor"].transform(data)[:,:2],
        columns=[x_label,y_label]
    )

    pcadf["predicted_cluster"] = pipeline["clusterer"]["kmeans"].labels_
    if true_labels is not None:
        pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

    plt.style.use("fivethirtyeight")
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if true_labels is not None:
        scat = sns.scatterplot(
            data=pcadf,
            x=x_label,
            y=y_label,
            s=50,
            hue="predicted_cluster",
            style="true_label",
            palette="Set2",
            ax=ax
        )
    else:
        scat = sns.scatterplot(
            data=pcadf,
            x=x_label,
            y=y_label,
            s=50,
            hue="predicted_cluster",
            palette="Set2",
            ax = ax
        )

    scat.set_title(
        "Clustering results from kmeans"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.show()