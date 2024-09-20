import os
import pickle
import logging
import awkward as ak
import numpy as np
import dask
import gudhi as gd
from scnn.build_matrices import build_boundaries, build_laplacians
import dask
from dask import delayed, compute
from dask.distributed import Client
class EventProcessor:
    def __init__(self, pkl_file, columns, label_column=None):
        """
        Initialize the EventProcessor class.

        Parameters:
        - pkl_file (str): Path to the .pkl file containing event data.
        - columns (list of str): List of column names to extract from each event's data.
        - label_column (str, optional): The name of the column to extract as labels.
        """
        self.pkl_file = pkl_file
        self.columns = columns
        self.label_column = label_column
        self.event_dicts = self._load_pkl_file()

    def _load_pkl_file(self):
        """Load the event dictionary from the provided pickle file."""
        with open(self.pkl_file, 'rb') as f:
            return pickle.load(f)

    def _extract_coordinates(self):
        """
        Extracts specified columns from the event_dicts.

        Returns:
        - coordinates (dict): A dictionary with event IDs as keys and stacked arrays of specified columns as values.
        - labels (dict): A dictionary with event IDs as keys and label column values, if label_column is provided.
        """
        coordinates = {}
        labels = {}

        for event_id, data in self.event_dicts.items():
            extracted_columns = []
            for column in self.columns:
                if column in data:
                    extracted_columns.append(ak.to_numpy(data[column]))

            # Stack the extracted columns along the last axis
            if extracted_columns:
                coordinates[event_id] = np.stack(extracted_columns, axis=-1)

            # Extract the label column if provided
            if self.label_column and self.label_column in data:
                labels[event_id] = ak.to_numpy(data[self.label_column])

        return coordinates, labels

    def _compute_rips(self, event_id, coord, max_dimension, sparsity, filtration_val):
        """
        Helper function to compute Vietoris-Rips complex for a single event.

        Parameters:
        - event_id (str): The event ID.
        - coord (numpy.ndarray): The coordinates of the event.
        - max_dimension (int): The maximum dimension of the simplicial complex.

        Returns:
        - tuple: (laplacians, boundaries) for the event.
        """

        def extract_simplices(simplex_tree):
            simplices = [dict() for _ in range(simplex_tree.dimension() + 1)]
            for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
                k = len(simplex)
                simplices[k - 1][frozenset(simplex)] = len(simplices[k - 1])
            return simplices

        rips_complex = gd.RipsComplex(points=coord, sparse=sparsity)
        st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        _ = st.prune_above_filtration(filtration_val)
        assert st.num_vertices() == coord.shape[0]
        # assert st.dimension() == max_dimension

        simplices = extract_simplices(st)
        boundaries = build_boundaries(simplices)
        laplacians = build_laplacians(boundaries)

        return laplacians, boundaries, simplices

    def compute_lapl_and_bounds(self, step, limit, max_dimension=2, sparsity=0.3, filtration_val=1.0,
                                                    output_dir="scnn/bounds_and_laps", entity = "train"):
        """
        Compute laplacians and boundaries for all events and store them in separate .npy files.

        Parameters:
        - step (int): Number of events to process at a time.
        - max_dimension (int, optional): The maximum dimension of the simplicial complexes (default is 2).
        - sparsity (float, optional): Sparsity factor for Rips complex construction.
        - filtration_val (float, optional): Filtration value for pruning.
        - output_dir (str, optional): Directory to store the laplacians and boundaries (default is 'downloads/scnn').

        Returns:
        - None
        """

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Create the 'scnn' directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Extract coordinates
        coordinates, labels = self._extract_coordinates()
        total_events = len(coordinates)
        logging.info(f'Total events: {total_events}')

        if limit is not None:
            total_events = min(limit, total_events)
            logging.info(f'Limiting to the first {total_events} events')

        all_laplacians = []
        all_boundaries = []
        all_node_feats = []
        all_labels = []
        all_simpl = []
        for idx, (event_id, coord) in enumerate(coordinates.items()):
            logging.info(f'Processing event {event_id} ({idx + 1}/{total_events})')
            if idx >= total_events:
                break
            # Compute the laplacians and boundaries for the event
            laplacians, boundaries, simplices = self._compute_rips(event_id, coord, max_dimension, sparsity, filtration_val)

            # Store results in lists
            all_laplacians.append(laplacians)
            all_boundaries.append(boundaries)
            all_node_feats.append(coord)
            all_labels.append(labels[event_id])
            all_simpl.append(simplices)
            # Free memory for large variables
            del laplacians, boundaries, coord, simplices

        np.savez_compressed(os.path.join(output_dir, f'{entity}_laplacians.npz'), *all_laplacians)
        np.savez_compressed(os.path.join(output_dir, f'{entity}_boundaries.npz'), *all_boundaries)
        np.savez_compressed(os.path.join(output_dir, f'{entity}_node_feats.npz'), *all_node_feats)
        np.savez_compressed(os.path.join(output_dir, f'{entity}_labels.npz'), *all_labels)
        np.savez_compressed(os.path.join(output_dir, f'{entity}_simpl.npz'), *all_simpl)
        logging.info('Laplacians, boundaries, node feats, simplices and labels computed and saved successfully.')



if __name__ == "__main__":
    pkl_path = os.path.join(os.getcwd(), 'downloads/processed/')
    pkl_file = os.path.join(pkl_path, 'train_data.pkl')
    columns_to_extract = ['y', 'pt']  # or ['eta', 'phi'], etc.
    label_column = 'label'  # Specify the label column if needed

    # Initialize the processor
    processor = EventProcessor(pkl_file, columns_to_extract, label_column)

    processor.compute_lapl_and_bounds(step=100, limit = 6500, max_dimension=2, sparsity=0.8, filtration_val=np.inf, entity = "train")
    # Use filtration value around 10-15
    # TODO: For now, the model works only when all 3 channels - nodes, edges and faces are present in each simplicial complex with corresponding feature vectors
    # TODO: Add handling for dynamic simplices and compute Laplacians and Boundary maps for those variable simplices.

    del processor, pkl_file

    pkl_file = os.path.join(pkl_path, 'test_data.pkl')

    # Initialize the processor
    processor = EventProcessor(pkl_file, columns_to_extract, label_column)

    processor.compute_lapl_and_bounds(step=100, limit = 1000, max_dimension=2, sparsity=0.8, filtration_val=np.inf, entity = "test")

    del processor, pkl_file

    pkl_file = os.path.join(pkl_path, 'val_data.pkl')

    # Initialize the processor
    processor = EventProcessor(pkl_file, columns_to_extract, label_column)

    processor.compute_lapl_and_bounds(step = 100, limit = 2000, max_dimension=2, sparsity=0.8, filtration_val=np.inf, entity = "val")

    del processor, pkl_file