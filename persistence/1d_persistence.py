from gtda.diagrams import PersistenceLandscape
from scipy.sparse.csgraph import connected_components
import networkx as nx
from gtda.plotting import plot_diagram, plot_betti_curves
# data wrangling
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List
from PIL import Image
from hepml.core import download_dataset
from scipy import ndimage

# tda magic
from gtda.homology import VietorisRipsPersistence, CubicalPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_heatmap, plot_point_cloud, plot_diagram
from gtda.pipeline import Pipeline
from hepml.core import make_point_clouds, load_shapes
from gtda.graphs import GraphGeodesicDistance

# ml tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# dataviz
import matplotlib.pyplot as plt


class VietorisPersistenceModule:
  def __init__(self, max_edge_length = np.inf, homology_dim = (0, 1, 2, 3)):
    self.point_clouds_basic, self.labels_basic = make_point_clouds(n_samples_per_shape=100, n_points=40, noise=0.5)
    self.point_cloud = self.point_clouds_basic[0]
    self.max_edge_length = max_edge_length
    self.homology_dim = homology_dim
    self.persistence_diagram = None
    self.betti_numbers = None
    self.adj_graph = None
    self.rips_complex = None

  def vietoris_rips_complex(self):
    self.rips_complex = VietorisRipsPersistence(metric = 'euclidean',
                               max_edge_length = self.max_edge_length,
                               homology_dimensions = self.homology_dim)

    self.persistence_diagram = self.rips_complex.fit_transform([self.point_cloud])[0]


  def compute_betti(self):
        landscape = PersistenceLandscape(n_layers=1, n_bins=100, n_jobs=6)
        landscapes = landscape.fit_transform([self.persistence_diagram])

        # Compute Betti numbers
        self.betti_numbers = [
            np.sum(landscape[i]) for landscape, i in zip(landscapes, self.homology_dim)
        ]

        # Plot Betti curves
        fig = plot_betti_curves(
            landscapes[0],
            samplings=landscape.sampling_range_[0],
            homology_dimensions=self.homology_dim
        )
        fig.show()

  def plot_persistence_diagram(self):
        # Plot the persistence diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_diagram(self.persistence_diagram, ax=ax, show=False)
        plt.title("Persistence Diagram")
        plt.show()

  def create_persistent_graph(self):
      # Get the adjacency matrix from the persistence module (note: fit_transform returns the diagram)
      adjacency_matrix = self.rips_complex.fit_transform(self.point_cloud.reshape(1, *self.point_cloud.shape))

      # Convert the persistence diagram to a graph structure
      self.adj_graph = nx.Graph()

      # Here, we extract edges from the adjacency matrix that correspond to a certain filtration value
      for i in range(len(self.point_cloud)):
          for j in range(i+1, len(self.point_cloud)):
              if adjacency_matrix[0][i][j] < self.max_edge_length:
                  self.adj_graph.add_edge(i, j, weight=adjacency_matrix[0][i][j])

      # Optionally, visualize the graph
      pos = {i: self.point_cloud[i] for i in range(len(self.point_cloud))}
      nx.draw(self.adj_graph, pos, with_labels=True, node_size=50)
      plt.title("Graph from Vietoris-Rips Complex")
      plt.show()

  def preprocess(self):
    self.vietoris_rips_complex()
    self.compute_betti()
    self.plot_persistence_diagram()
    self.create_persistent_graph()