import numpy as np

from typing import Callable as func
from itertools import product, starmap

DATASET = 'data/C1.txt'

def load_dataset(dataset: str) -> np.ndarray:
    data = np.loadtxt(dataset, usecols=(1,2))
    return data

def dist_mat(C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    return np.array(list(starmap(lambda p, q: np.linalg.norm(p - q), product(C1, C2))))

def s_link(C1: np.ndarray, C2: np.ndarray) -> float:
    return np.min(dist_mat(C1, C2))

def c_link(C1: np.ndarray, C2: np.ndarray) -> float:
    return np.max(dist_mat(C1, C2))

def hierarchical(points: np.ndarray, linkage: func, k: int = 3) -> list:
    clusters = [np.array([point]) for point in points]
    min_dist = np.inf
    while (num_clusters := len(clusters)) > k:
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                dist = linkage(clusters[i], clusters[j])
                if dist <= min_dist:
                    min_dist = dist
                    C1_idx = i
                    C2_idx = j
        clusters[C1_idx] = np.vstack((clusters[C1_idx], clusters[C2_idx]))
        clusters.pop(C2_idx)
        min_dist = np.inf
    return clusters

def main() -> int:
    dataset = load_dataset(DATASET)
    s_link_clusters = hierarchical(dataset, s_link)
    c_link_clusters = hierarchical(dataset, c_link)

    for index, cluster in enumerate(s_link_clusters, 1):
        np.savetxt(f'data/hierarchical/s-link-cluster-{index}.csv', cluster, delimiter=',')
    for index, cluster in enumerate(c_link_clusters, 1):
        np.savetxt(f'data/hierarchical/c-link-cluster-{index}.csv', cluster, delimiter=',')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
