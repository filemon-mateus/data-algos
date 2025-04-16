import numpy as np

from kplusplus import load_dataset, kplusplus, kmeans_cost

TRIALS = 100
EPSILON = 0.01
GONZALEZ = 1.6005365908052087

np.random.seed(0)

def main() -> int:
    dataset = load_dataset('data/C2.txt')
    kmcosts = np.zeros(TRIALS)
    counter = 0

    for trial in range(TRIALS):
        centers = kplusplus(dataset)
        kmcosts[trial] = kmeans_cost(dataset, centers)
        if np.abs(kmcosts[trial] - GONZALEZ) < EPSILON:
            counter += 1

    np.savetxt('data/kplusplus/kplusplus-cdf.csv', kmcosts, delimiter=',')
    print(counter / TRIALS)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
