import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams['font.size'] = 20
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{kpfonts}'

DATASET = 'data/C1.txt'

def load_dataset(dataset: str) -> np.ndarray:
    data = np.loadtxt(dataset, usecols=(1,2))
    return data

def save_dendrogram(points: np.ndarray, method: str) -> None:
    plt.cla()
    Z = linkage(points, method=method)
    _ = dendrogram(
        Z,
        leaf_label_func=lambda i: r'$%s$' % (i+1),
        leaf_font_size=plt.rcParams.get('font.size'),
        leaf_rotation=90,
        count_sort=True,
        color_threshold=(0.7 if method == 'single' else 2.1),
    )
    plt.tight_layout()
    plt.savefig(f'figs/dendrogram-{method}.pdf')

def main() -> int:
    dataset = load_dataset(DATASET)
    save_dendrogram(dataset, 'single')
    save_dendrogram(dataset, 'complete')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
