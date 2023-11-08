import ott
import jax.numpy as jnp
import numpy as np
from tqdm.notebook import tqdm


def get_pitch_array(pitcher, data):
    pitcher_data = data[data["pitcher"] == pitcher]

    locations = pitcher_data[["plate_x", "plate_z"]]
    return jnp.array(locations)


def get_pitch_geometry(pitcher, data):
    pitcher_data = data[data["pitcher"] == pitcher]

    locations = pitcher_data[["plate_x", "plate_z"]]
    return ott.geometry.pointcloud.PointCloud(jnp.array(locations))


def get_sinkhorn_distance(array_1, array_2, epsilon):
    geom = ott.geometry.pointcloud.PointCloud(x=array_1, y=array_2, epsilon=epsilon)
    prob = ott.problems.linear.linear_problem.LinearProblem(geom=geom)
    solver = ott.solvers.linear.sinkhorn.Sinkhorn()
    soln = solver(prob)
    return soln


def get_dist_matrix(arrays, epsilon=1.0):
    n = len(arrays)
    mat = np.zeros([n, n])
    for i in tqdm(range(0, n)):
        for j in range(i + 1, n):
            soln = get_sinkhorn_distance(arrays[i], arrays[j], epsilon=epsilon)
            mat[i, j] = soln.primal_cost
            mat[j, i] = soln.primal_cost
    return mat
