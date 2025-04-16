import numpy as np

from lloyd import load_dataset, kmeans_cost
from sklearn.cluster import KMeans

TRIALS = 100
EPSILON = 0.01
KPLUSPLUS = 1.1120373276842588

def main() -> int:
    dataset = load_dataset('data/C2.txt')
    kmcosts = np.zeros(TRIALS)
    counter = 0
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=1)

    for trial in range(TRIALS):
        centers = kmeans.fit(dataset).cluster_centers_
        kmcosts[trial] = kmeans_cost(dataset, centers)
        if np.abs(kmcosts[trial] - KPLUSPLUS) < EPSILON:
            counter += 1

    np.savetxt('data/lloyd/lloyd-scikit-cdf.csv', kmcosts, delimiter=',')
    print(counter / TRIALS)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
