import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
import plotly.graph_objects as go


class ClusterAnalysis:
    def __init__(self, dataframe, n_neighbors=15, min_cluster_size=5, min_dist=0.1, metric='euclidean', cluster_dims=None):
        self.dataframe = dataframe.copy()
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_dist = min_dist
        self.metric = metric
        self.cluster_dims = cluster_dims

    def perform_umap(self):
        reducer = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric, random_state=42)
        umap_data = reducer.fit_transform(self.dataframe[self.cluster_dims])
        self.dataframe['x'] = umap_data[:, 0]
        self.dataframe['y'] = umap_data[:, 1]

    def perform_hdbscan(self):
        np.random.seed(42)
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, metric=self.metric)
        self.dataframe['cluster'] = clusterer.fit_predict(self.dataframe[['x', 'y']])
    
    def plot_scatter(self):
        unique_clusters = sorted(self.dataframe['cluster'].unique())
        fig = go.Figure()
        
        weight_class = self.dataframe['weight_class'].drop_duplicates().values

        for cluster in unique_clusters:
            cluster_data = self.dataframe[self.dataframe['cluster'] == cluster]
            fig.add_trace(go.Scatter(x=cluster_data['x'], y=cluster_data['y'], mode='markers', name='Cluster ' + str(cluster),
                                     marker=dict(size=6, opacity=0.4), hovertext=cluster_data['Fighter_dims'],
                                     text=cluster_data['Fighter_dims'], textposition='top center', textfont=dict(size=10, color='black')))

        fig.update_layout(title=f'Fighting Style Clusters: {weight_class}', showlegend=True, width=750, height=750)
        fig.show()

    def run(self):
        self.perform_umap()
        self.perform_hdbscan()
        self.plot_scatter()