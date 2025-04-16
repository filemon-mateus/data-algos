import numpy as np

from itertools import combinations, starmap

TAU_VAL = .75
DATASET = 'data/R.txt'

def normalize(v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v)
    return v / v_norm

def angular_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return 1 - np.arccos(np.dot(u, v)) / np.pi

def load_dataset(dataset: str) -> np.ndarray:
    vectors = np.loadtxt(dataset)
    return vectors

def main() -> int:
    vectors = np.array(list(map(normalize, load_dataset(DATASET))))
    combins = combinations(vectors, 2)

    angular_sims = np.array(list(starmap(angular_similarity, combins)))
    np.savetxt('data/angular-sims-r-dataset.csv', angular_sims, delimiter=',')

    print((angular_sims > TAU_VAL).sum())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
