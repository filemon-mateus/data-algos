import numpy as np

from kplusplus import kplusplus

DATASET = 'data/C3.txt'

np.random.seed(0)

def load_dataset(dataset: str) -> np.ndarray:
    data = np.loadtxt(dataset, usecols=(1,2))
    return data

def center_dists(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = np.ones(points.shape[0]) * np.inf
    for point_id, point in enumerate(points):
        for center_id, center in enumerate(centers):
            dists[point_id] = min(dists[point_id], np.linalg.norm(point - center))
    return dists

def kmedians_cost(points: np.ndarray, centers: np.ndarray) -> float:
    dists = center_dists(points, centers)
    return np.mean(dists)

def kmedians(points: np.ndarray, k: int = 4, init_centers: np.ndarray | None = None, num_iter: int = 100) -> np.ndarray:
    centers = init_centers if init_centers is not None else points[:k]
    for _ in range(num_iter):
        dists = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
        index = np.argmin(dists, axis=1)
        for center_id in range(k):
            centers[center_id] = np.median(points[index == center_id], axis=0)
    return centers

def main() -> int:
    dataset = load_dataset(DATASET)
    centers = kmedians(dataset, init_centers=kplusplus(dataset))
    kmecost = kmedians_cost(dataset, centers)
    indices = np.arange(centers.shape[0]) + 1

    np.savetxt('data/kmedians/best-centers.txt', np.column_stack((indices, centers)), delimiter='\t', fmt=['%d','%5f','%5f','%5f','%5f','%5f'])

    print(kmecost)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
