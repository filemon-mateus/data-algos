import numpy as np

FILENAME = 'data/web-Google.txt'
EPSILON = 0.0001
BETA = 0.85

def update_ranks(matrix: dict, r_prev: np.ndarray) -> np.ndarray:
    r_size = r_prev.shape[0]
    r_next = np.repeat((1 - BETA) / r_size, r_size)
    for src, dst_list in matrix.items():
        r_next[dst_list] += BETA * r_prev[src] / len(dst_list)
    return r_next

with open(FILENAME) as file:
    lines = [line.strip().split('\t') for line in file]
    edges = [(int(src), int(dst)) for (src, dst) in lines[4:]]
    nodes = set([node for edge in edges for node in edge])

    node_index = {node: index for index, node in enumerate(nodes)}
    index_node = {index: node for node, index in node_index.items()}

    matrix = {}
    for (src, dst) in edges:
        src = node_index[src]
        dst = node_index[dst]
        matrix[src] = matrix.get(src, []) + [dst]

    n_size = len(nodes)
    r_prev = np.repeat(1 / n_size, n_size)
    r_next = np.zeros(n_size)

    while np.linalg.norm(r_next - r_prev, ord=1) >= EPSILON:
        r_prev = r_next
        r_next = update_ranks(matrix, r_prev)

    max_node = index_node[r_next.argmax()]
    n_degree = sum(1 for _, dst in edges if dst == max_node)
    print(max_node, n_degree)
