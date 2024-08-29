import numpy as np
import pandas as pd
import multipers as mp
import multipers.grids as mpg
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import awkward as ak
import dask.array as da
import dask.dataframe as dd
from multipers.ml.convolutions import KDE as KernelDensity

class PersistenceModule:
    def __init__(self, dataset):
        """
        Initialize the PersistenceModule with a point cloud dataset.

        Parameters:
        dataset : Awkward Array or Dask Array
            The point cloud dataset where each entry contains a variable number of particles
            with 4 coordinates (px, py, pz, energy).
        """
        self.dataset = dataset

    def compute_simplex_trees(self):
        """
        Compute the SimplexTreeMulti for each entry in the dataset.

        Returns:
        simplex_trees : list of mp.SimplexTreeMulti
            A list containing the SimplexTreeMulti for each entry in the dataset.
        """
        simplex_trees = []

        # Iterate over each entry in the dataset
        for entry in self._iterate_dataset(self.dataset):
            # Extract the (x, y, z, t) coordinates
            coordinates = np.array(entry)

            # Create a RipsComplex using the first 3 coordinates (x, y, z)
            rips_complex = gd.RipsComplex(points=coordinates[:, :3], sparse=0.2)
            simplex_tree = rips_complex.create_simplex_tree()

            # Create a SimplexTreeMulti with 2 parameters
            st_multi = mp.SimplexTreeMulti(simplex_tree, num_parameters=2)

            # Calculate the co-log-density for the second parameter
            codensity = - KernelDensity(bandwidth=0.2).fit(coordinates[:, :3]).score_samples(coordinates[:, :3])

            # Fill the second parameter with the co-log-density
            st_multi.fill_lowerstar(codensity, parameter=1)

            # Append the SimplexTreeMulti to the list
            simplex_trees.append(st_multi)

        return simplex_trees

    def _iterate_dataset(self, dataset):
        """
        A helper function to iterate over the dataset.

        Parameters:
        dataset : Awkward Array or Dask Array
            The dataset to iterate over.

        Yields:
        entry : np.ndarray
            Each entry in the dataset as a numpy array.
        """
        if isinstance(dataset, ak.Array):
            for entry in dataset:
                yield entry
        elif isinstance(dataset, da.Array):
            for entry in dataset.to_delayed():
                yield entry.compute()
        else:
            raise TypeError("Dataset must be an Awkward Array or Dask Array.")