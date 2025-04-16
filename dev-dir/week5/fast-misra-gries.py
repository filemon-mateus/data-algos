import numpy as np

from numba import njit

K = 12
DATASET_ONE = 'data/S1.txt'
DATASET_TWO = 'data/S2.txt'

def read_file(filename: str) -> np.ndarray:
    with open(filename) as file:
        corpus = file.readline()
    corpus_ = np.array(list(map(ord, corpus)))
    return corpus_

@njit('int64(int64[:], int64)')
def find_item(label: np.ndarray, item: int) -> int:
    for i in range(label.shape[0]):
        if label[i] == item:
            return i
    return -1

@njit('UniTuple(int64[:], 2)(int64[:], int64)')
def misra_gries(S: np.ndarray, K: int) -> tuple:
    count = np.zeros(K-1, dtype=np.int64)
    label = np.zeros(K-1, dtype=np.int64)

    for i in range(S.shape[0]):
        j = find_item(label, S[i])
        l = find_item(count, 0)
        if j != -1:
            count[j] += 1
        elif l != -1:
            label[l] = S[i]
            count[l] = 1
        else:
            count -= 1

    return count, label

def main() -> int:
    all_files = list(map(read_file, [DATASET_ONE, DATASET_TWO]))
    for S in all_files:
        count, label = misra_gries(S, K)
        for i in range(K-1):
            print(f'{chr(label[i])} | {count[i]:{7}} | {count[i] / S.shape[0]:.5f}')
        print()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
