import numpy as np

FILENAME = 'data/web-Google-10k.txt'
EPSILON = 0.0001
BETA = 0.85

def build_trans_matrix(init_matrix: np.ndarray, beta: float = BETA) -> np.ndarray:
    num_nodes = init_matrix.shape[0]
    const_vec = np.repeat(1 / num_nodes, num_nodes)
    d_entries = init_matrix.sum(axis = 0)
    dead_ends = np.where(d_entries == 0)[0]

    for dead_end in dead_ends:
        init_matrix[:, dead_end] = 1 / num_nodes
        d_entries[dead_end] = 1

    init_matrix /= d_entries
    return beta * init_matrix + (1 - beta) * const_vec

def power_iteration(trans_matrix: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    n_size = trans_matrix.shape[0]
    r_prev = np.repeat(1 / n_size, n_size)
    r_next = trans_matrix @ r_prev

    while np.linalg.norm(r_next - r_prev, ord=1) >= epsilon:
        r_prev = r_next
        r_next = trans_matrix @ r_prev

    return r_next

with open(FILENAME) as file:
    lines = [line.strip().split('\t') for line in file]
    edges = [(int(src), int(dst)) for (src, dst) in lines[4:]]
    nodes = set([node for edge in edges for node in edge])

    node_index = {node: index for index, node in enumerate(nodes)}
    index_node = {index: node for node, index in node_index.items()}

    n_size = len(nodes)
    matrix = np.zeros((n_size, n_size))
    for (src, dst) in edges:
        src = node_index[src]
        dst = node_index[dst]
        matrix[dst][src] = 1

    M_matrix = build_trans_matrix(matrix)
    r_vector = power_iteration(M_matrix)
    max_node = index_node[r_vector.argmax()]
    n_degree = sum(1 for _, dst in edges if dst == max_node)
    print(max_node, n_degree)
