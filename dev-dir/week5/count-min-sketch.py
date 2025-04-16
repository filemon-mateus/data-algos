import numpy as np

from numba import njit

K, T = 12, 6
DATASET_ONE = 'data/S1.txt'
DATASET_TWO = 'data/S2.txt'

np.random.seed(42)

def read_file(filename: str) -> np.ndarray:
    with open(filename) as file:
        corpus = file.readline()
    corpus_ = np.array(list(map(ord, corpus)))
    return corpus_

def rand_salts() -> np.ndarray:
    return np.random.randint(T, size=T)

@njit('int64(int64, int64)')
def hash_func(salt: int, item: int) -> int:
    return hash((salt, item)) % K

@njit('int64[:,:](int64[:], int64, int64, int64[:])')
def count_min(S: np.ndarray, K: int, T: int, salts: np.ndarray) -> np.ndarray:
    count = np.zeros((T,K), dtype=np.int64)
    for i in range(S.shape[0]):
        for j in range(T):
            l = hash_func(salts[j], S[i])
            count[j,l] += 1
    return count

@njit('int64(int64[:,:], int8, int64[:])')
def query_min(count: np.ndarray, char: int, salts: np.ndarray) -> int:
    char_count = np.zeros(count.shape[0], dtype=np.int64)
    for j in range(count.shape[0]):
        l = hash_func(salts[j], char)
        char_count[j] = count[j,l]
    return char_count.min()

def main() -> int:
    all_files = list(map(read_file, [DATASET_ONE, DATASET_TWO]))
    rnd_salts = rand_salts()
    for S in all_files:
        count = count_min(S, K, T, rnd_salts)
        query_vals = ['a','b','c']
        count_mins = list(map(lambda char: query_min(count, ord(char), rnd_salts), query_vals))
        for char, char_count in zip(query_vals, count_mins):
            print(f'{char} | {char_count:{7}} | {char_count / S.shape[0]:.5f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
