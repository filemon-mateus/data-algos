import numpy as np

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

def kcenter_cost(points: np.ndarray, centers: np.ndarray) -> float:
    dists = center_dists(points, centers)
    return np.max(dists)

def kmeans_cost(points: np.ndarray, centers: np.ndarray) -> float:
    dists = center_dists(points, centers)
    return np.sqrt(np.mean(np.square(dists)))

def max_dist_point(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = center_dists(points, centers)
    return points[dists.argmax()]

def gonzalez(points: np.ndarray, k: int = 4) -> np.ndarray:
    centers = np.array([points[0]])
    while len(centers) < k:
        centers = np.vstack((centers, max_dist_point(points, centers)))
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
    centers = gonzalez(dataset)
    clabels = query_labels(dataset, centers)

    np.savetxt('data/gonzalez/centers.csv', centers, delimiter=',')
    np.savetxt('data/gonzalez/points.csv', np.column_stack((dataset, clabels)), delimiter=',')

    print(kcenter_cost(dataset, centers))
    print(kmeans_cost(dataset, centers))

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
