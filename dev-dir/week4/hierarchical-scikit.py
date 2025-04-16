import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

DATASET = 'data/C1.txt'

def load_dataset(dataset: str) -> np.ndarray:
    data = np.loadtxt(dataset, usecols=(1,2))
    return data

def hierarchical(points: np.ndarray, linkage: str, k: int = 3) -> np.ndarray:
    clusters = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(points)
    return clusters.labels_

def main() -> int:
    dataset = load_dataset(DATASET)
    s_link_labels = hierarchical(dataset, 'single')
    c_link_labels = hierarchical(dataset, 'complete')
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(dataset[:,0], dataset[:,1], c=s_link_labels, s=10)
    ax2.scatter(dataset[:,0], dataset[:,1], c=c_link_labels, s=10)
    plt.show()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
