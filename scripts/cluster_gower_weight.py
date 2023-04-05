import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
import plotly.graph_objects as go
import gower



class ClusterAnalysis:
    def __init__(self, dataframe, n_neighbors=15, min_cluster_size=5, min_dist=0.1, cluster_dims=None, export_data=False, weight_factor=1):
        self.dataframe = dataframe.copy()
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_dist = min_dist
        self.cluster_dims = cluster_dims
        self.weight_class = self.dataframe['weight_class'].drop_duplicates().values[0]
        self.export_data = export_data
        self.weight_factor = weight_factor
        
    
    def get_binary_columns(self, dataframe):
        binary_columns = []
        for col in dataframe.columns:
            unique_values = dataframe[col].unique()
            if len(unique_values) == 2 and sorted(unique_values) == [0, 1]:
                binary_columns.append(col)
        return binary_columns
    
    def perform_umap(self):
        data = self.dataframe[self.cluster_dims]
        gower_distance_matrix = gower.gower_matrix(data)

        # Identify the binary columns
        binary_columns = self.get_binary_columns(data)

        # Get the indices of binary columns in 'data'
        binary_column_indices = [data.columns.get_loc(col) for col in binary_columns]
        
        for index in binary_column_indices:
            gower_distance_matrix[:, index] *= self.weight_factor
            gower_distance_matrix[index, :] *= self.weight_factor

        # Perform UMAP with the modified Gower distance matrix
        reducer = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric='precomputed', random_state=42)
        umap_data = reducer.fit_transform(gower_distance_matrix)
        self.dataframe['x'] = umap_data[:, 0]
        self.dataframe['y'] = umap_data[:, 1]


    def perform_hdbscan(self):
        np.random.seed(42)
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, metric='euclidean')
        self.dataframe['cluster'] = clusterer.fit_predict(self.dataframe[['x', 'y']])
        self.dataframe['specific_cluster'] = self.dataframe['cluster'].astype(str) + '_' + self.dataframe['weight_class']
        if self.export_data == True:
            self.dataframe.to_csv(f"C:\\Users\\johna\\anaconda3\\envs\\ufc-env\\ufc_styles\\data\\02_intermediate\\fighter_cluster{ self.weight_class}.csv", index=False)
        
    
    def plot_scatter(self):
        unique_clusters = sorted(self.dataframe['cluster'].unique())
        fig = go.Figure()
        
        for cluster in unique_clusters:
            cluster_data = self.dataframe[self.dataframe['cluster'] == cluster]
            fig.add_trace(go.Scatter(x=cluster_data['x'], y=cluster_data['y'], mode='markers', name='Cluster ' + str(cluster),
                                     marker=dict(size=6, opacity=0.4), hovertext=cluster_data['Fighter_dims'],
                                     text=cluster_data['Fighter_dims'], textposition='top center', textfont=dict(size=10, color='black')))

        fig.update_layout(title=f'Fighting Style Clusters: {self.weight_class}', showlegend=True, width=750, height=750)
        fig.show()

    def run(self):
        self.perform_umap()
        self.perform_hdbscan()
        self.plot_scatter()