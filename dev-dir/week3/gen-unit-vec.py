import numpy as np

from itertools import combinations, starmap

TAU_VAL = .75
D_VALUE = 100
T_VALUE = 200

def random_uniform(lowerbound: float, upperbound: float, size: tuple) -> np.ndarray:
    return np.random.uniform(lowerbound, upperbound, size)

def random_normal(lowerbound: float, upperbound: float, size: tuple) -> np.ndarray:
    u_one = random_uniform(lowerbound, upperbound, size)
    u_two = random_uniform(lowerbound, upperbound, size)
    return np.sqrt(-2 * np.log(u_one)) * np.cos(2 * np.pi * u_two)

def normalize(v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v)
    return v / v_norm

def angular_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return 1 - np.arccos(np.dot(u, v)) / np.pi

def main() -> int:
    vectors = random_normal(0.0, 1.0, (T_VALUE, D_VALUE))
    vectors = np.array(list(map(normalize, vectors)))
    combins = list(combinations(vectors, 2))

    dot_products = np.array(list(starmap(np.dot, combins)))
    angular_sims = np.array(list(starmap(angular_similarity, combins)))

    np.savetxt('data/dot-products.csv', dot_products, delimiter=',')
    np.savetxt('data/angular-sims.csv', angular_sims, delimiter=',')

    print((angular_sims > TAU_VAL).sum())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
