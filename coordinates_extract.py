import numpy as np
import awkward as ak


def extract_coordinates_cartesian(event_dicts):
    """
    Extracts x, y, z coordinates from the event_dicts.

    Parameters:
    - event_dicts (dict): A dictionary where keys are event IDs and values are dictionaries with 'x', 'y', 'z' arrays.

    Returns:
    - A dictionary with event IDs as keys and (x, y, z) coordinates as values.
    """
    coordinates = {}
    for event_id, data in event_dicts.items():
        x = ak.to_numpy(data['x'])
        y = ak.to_numpy(data['y'])
        z = ak.to_numpy(data['z'])
        x_y_z_stack = np.stack((x, y, z), axis = -1)
        # print(x_y_z_stack.shape)
        # input()
        coordinates[event_id] = x_y_z_stack
    return coordinates


def extract_coordinates_eta_phi(event_dicts):
    """
    Extracts eta, phi coordinates from the event_dicts.

    Parameters:
    - event_dicts (dict): A dictionary where keys are event IDs and values are dictionaries with 'eta', 'phi' arrays.

    Returns:
    - A dictionary with event IDs as keys and (eta, phi) coordinates as values.
    """
    coordinates = {}
    for event_id, data in event_dicts.items():
        eta = ak.to_numpy(data['eta'])
        # print(eta.shape)
        phi = ak.to_numpy(data['phi'])
        # print(eta.shape)
        eta_phi_stack = np.stack((eta, phi), axis=-1)
        # print(eta_phi_stack.shape)
        # print(type(eta_phi_stack))
        # input()
        coordinates[event_id] = eta_phi_stack

        del eta, phi, eta_phi_stack
    return coordinates


def extract_coords_pt_y_phi(event_dicts):
    """

    Extracts spatial coordinates from the event_dicts to apply Lorentz Invariance arguments.

    Paramters:
    - event_dicts (dict) : A dictionary where keys are event IDs and values are dictionaries with 'pt', 'y' and 'phi' arrays.

    Returns :
    - A dictionary with event IDs as keys and (pt, y, phi) coordinates as values.
    """

    coordinates = {}
    for event_id, data in event_dicts.items():
        pt = ak.to_numpy(data['pt'])
        y = ak.to_numpy(data['y'])
        phi = ak.to_numpy(data['phi'])
        pt_y_phi_stack = np.stack((pt, y, phi), axis=-1)

        coordinates[event_id] = pt_y_phi_stack

    del pt, y, phi, pt_y_phi_stack
    return coordinates



def extract_coords_y_phi(event_dicts):
    """
    Extracts spatial coordinates from the event_dicts to apply Lorentz Invariance arguments.

    Parameters:
    - event_dicts (dict) : A dictionary where keys are event IDs and values are dictionaries with 'y' and 'phi' arrays.

    Returns :
    - A dictionary with event IDs as keys and (y, phi) coordinates as values.
    """

    coordinates = {}
    for event_id, data in event_dicts.items():
        y = ak.to_numpy(data['y'])
        phi = ak.to_numpy(data['phi'])
        y_phi_stack = np.stack((y, phi), axis=-1)

        coordinates[event_id] = y_phi_stack

    del y, phi, y_phi_stack
    return coordinates