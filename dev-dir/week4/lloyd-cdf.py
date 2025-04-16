import numpy as np

from kplusplus import kplusplus
from lloyd import load_dataset, kmeans_cost, lloyd

TRIALS = 100
EPSILON = 0.01
KPLUSPLUS = 1.1120373276842588

def main() -> int:
    dataset = load_dataset('data/C2.txt')
    kmcosts = np.zeros(TRIALS)
    counter = 0

    for trial in range(TRIALS):
        centers = lloyd(dataset, init_centers=kplusplus(dataset))
        kmcosts[trial] = kmeans_cost(dataset, centers)
        if np.abs(kmcosts[trial] - KPLUSPLUS) < EPSILON:
            counter += 1

    np.savetxt('data/lloyd/lloyd-cdf.csv', kmcosts, delimiter=',')
    print(counter / TRIALS)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
