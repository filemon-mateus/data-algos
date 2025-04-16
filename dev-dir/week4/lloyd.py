import numpy as np

from gonzalez import gonzalez

DATASET = 'data/C2.txt'

def load_dataset(dataset: str) -> np.ndarray:
    data = np.loadtxt(dataset, usecols=(1,2))
    return data

def center_dists(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = np.ones(points.shape[0]) * np.inf
    for point_id, point in enumerate(points):
        for center_id, center in enumerate(centers):
            dists[point_id] = min(dists[point_id], np.linalg.norm(point - center))
    return dists

def kmeans_cost(points: np.ndarray, centers: np.ndarray) -> float:
    dists = center_dists(points, centers)
    return np.sqrt(np.mean(np.square(dists)))

def lloyd(points: np.ndarray, k: int = 4, init_centers: np.ndarray | None = None, num_iter: int = 100) -> np.ndarray:
    centers = init_centers if init_centers is not None else points[:k]
    for _ in range(num_iter):
        dists = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
        index = np.argmin(dists, axis=1)
        for center_id in range(k):
            centers[center_id] = np.mean(points[index == center_id], axis=0)
    return centers

def query_labels(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    labels, dists = np.zeros(points.shape[0]), np.ones(points.shape[0]) * np.inf
    for point_id, point in enumerate(points):
        for center_id, center in enumerate(centers):
            if (curr_dist := np.linalg.norm(point - center)) < dists[point_id]:
                dists[point_id], labels[point_id] = curr_dist, center_id
    return labels

def main() -> int:
    dataset = load_dataset(DATASET)
    all_centers = [
        ('lloyd', lloyd(dataset)),
        ('gonzalez', lloyd(dataset, init_centers=gonzalez(dataset)))
    ]

    for algorithm, centers in all_centers:
        clabels = query_labels(dataset, centers)
        np.savetxt(f'data/lloyd/centers-{algorithm}.csv', centers, delimiter=',')
        np.savetxt(f'data/lloyd/points-{algorithm}.csv', np.column_stack((dataset, clabels)), delimiter=',')
        print(algorithm, kmeans_cost(dataset, centers))

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
