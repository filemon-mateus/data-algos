import numpy as np

from numpy.linalg import norm

M_DATA = 'data/M.csv'
W_DATA = 'data/W.csv'

def load_dataset(dataset: str) -> np.ndarray:
    data = np.loadtxt(dataset, delimiter=',')
    return data

def matching_pursuit(X_mat: np.ndarray, y_vec: np.ndarray, s_val: float = 0.0, k_val: int = 69) -> np.ndarray:
    r_vec = y_vec
    alpha = np.zeros(X_mat.shape[1])

    for _ in range(k_val):
        r_cov = r_vec @ X_mat
        j_idx = np.argmax(np.abs(r_cov))

        alph1 = (r_vec @ X_mat[:, j_idx] + 0.5 * s_val) / np.square(norm(X_mat[:, j_idx]))
        alph2 = (r_vec @ X_mat[:, j_idx] - 0.5 * s_val) / np.square(norm(X_mat[:, j_idx]))
        norm1 = norm(r_vec - X_mat[:, j_idx] * alph1) + s_val * np.abs(alph1)
        norm2 = norm(r_vec - X_mat[:, j_idx] * alph2) + s_val * np.abs(alph2)

        alphj = alph1 if norm1 < norm2 else alph2
        r_vec = r_vec - X_mat[:, j_idx] * alphj
        alpha[j_idx] = alphj

        print(alphj, norm(r_vec))

    return alpha

def main() -> int:
    M = load_dataset(M_DATA)
    W = load_dataset(W_DATA)
    S = matching_pursuit(M, W)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
