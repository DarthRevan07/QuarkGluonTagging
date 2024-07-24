from gtda.point_clouds import ConsistentRescaling, ConsecutiveRescaling
from gtda.graphs import TransitionGraph, KNeighborsGraph

def adjust_density(point_cloud, density_factor):
    """Adjust point cloud density."""
    return point_cloud * density_factor

def adjust_distance(point_cloud, *args):
    """Adjust point cloud distances."""
    cr = ConsistentRescaling()
    point_cloud = cr.fit_transform(point_cloud, *args)
    return point_cloud