import numpy as np

from typing import Callable
from functools import reduce
from itertools import starmap

FILENAME = 'data/web-Google.txt'
EPSILON = 0.0001
BETA = 0.85

def generate_vector_stripes(vector: np.ndarray, stripe_size: int) -> np.ndarray:
    n_value = vector.shape[0]
    indices = np.arange(stripe_size, n_value, stripe_size)
    stripes = np.hsplit(vector, indices)
    return stripes

def mapper(matrix: dict, r_prev: np.ndarray) -> np.ndarray:
    r_size = r_prev.shape[0]
    r_next = np.repeat((1 - BETA) / n_size, n_size)
    for src, dst_list in matrix.items():
        for dst in dst_list:
            r_next[dst] += BETA * r_prev[src % r_size] / len(dst_list)
    return r_next

def reducer(accumulator: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return accumulator + vector

def map_reduce(mapper: Callable, reducer: Callable, args: tuple) -> np.ndarray:
    mapped = starmap(mapper, args)
    return reduce(reducer, mapped)

with open(FILENAME, 'r') as file:
    lines = [line.strip().split('\t') for line in file]
    edges = [[int(src), int(dst)] for (src, dst) in lines[4:]]
    nodes = set([node for edge in edges for node in edge])

    node_index = {node: index for index, node in enumerate(nodes)}
    index_node = {index: node for node, index in node_index.items()}

    n_size = len(nodes)
    vector = np.zeros(n_size)

    stripe_size = 30197
    stripe_nums = n_size // stripe_size
    matrix_stripes = [{} for _ in range(stripe_nums)]
    vector_stripes = generate_vector_stripes(vector, stripe_size)

    for (src, dst) in edges:
        src = node_index[src]
        dst = node_index[dst]
        matrix_stripe = matrix_stripes[src // stripe_size]
        matrix_stripe[src] = matrix_stripe.get(src, []) + [dst]

    stripes = zip(matrix_stripes, vector_stripes)
    reduced = map_reduce(mapper, reducer, stripes)

    while np.linalg.norm(reduced - vector, ord=1) >= EPSILON:
        vector = reduced
        vector_stripes = generate_vector_stripes(vector, stripe_size)
        stripes = zip(matrix_stripes, vector_stripes)
        reduced = map_reduce(mapper, reducer, stripes)

    max_node = index_node[reduced.argmax()]
    n_degree = sum(1 for _, dst in edges if dst == max_node)
    print(max_node, n_degree)
