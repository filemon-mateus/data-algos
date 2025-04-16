import numpy as np
import sklearn.datasets

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(42)

def preprocess(size: int = 500) -> np.ndarray:
    data = sklearn.datasets.fetch_rcv1().data
    idxs = np.random.choice(data.shape[0], size)
    data = data[idxs, :]
    return data

def main() -> int:
    data = preprocess()
    eps_vals = []
    abs_diff = []

    for eps in np.arange(0.1, 0.999, 0.2):
        min_dimension = johnson_lindenstrauss_min_dim(data.shape[0], eps=eps)
        if data.shape[1] < min_dimension:
            continue
        gaussian_projection = GaussianRandomProjection(eps=eps)
        data_transform = gaussian_projection.fit_transform(data)
        original_distances = euclidean_distances(data)
        tranform_distances = euclidean_distances(data_transform)
        absolute_difference = np.abs(original_distances - tranform_distances)
        abs_diff.append(absolute_difference.mean())
        eps_vals.append(eps)

    np.savetxt('data/gauss-proj-abs-diff.csv', absolute_difference.flatten(), delimiter=',')
    np.savetxt('data/gauss-proj-eps-vs-mean-abs-diff.csv', np.column_stack((eps_vals, abs_diff)), delimiter=',')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
