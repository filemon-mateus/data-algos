import numpy as np
import pandas as pd

MORTALITY_PATH = 'data/child-mortality.csv'
FERTILITY_PATH = 'data/woman-fertility.csv'

def preprocess() -> pd.DataFrame:
    mortality = pd.read_csv(MORTALITY_PATH).set_index('country')['2017'].to_frame() / 10
    fertility = pd.read_csv(FERTILITY_PATH).set_index('country')['2017'].to_frame()
    data = mortality.merge(fertility, left_index=True, right_index=True).dropna()
    data.columns = ['mortality', 'fertility']
    return data

def center_data(data: pd.DataFrame) -> pd.DataFrame:
    data['mortality'] -= data['mortality'].mean()
    data['fertility'] -= data['fertility'].mean()
    return data

def rank_1_approx(data: pd.DataFrame) -> pd.DataFrame:
    U, S, V = np.linalg.svd(data, full_matrices=False)
    U = U[:, 0].reshape(-1, 1)
    V = V[0, :].reshape(1, -1)
    return S[0] * np.dot(U, V)

def main() -> int:
    data = center_data(preprocess())
    _, _, V = np.linalg.svd(data, full_matrices=False)
    np.savetxt('data/mortality-vs-fertility.csv', data, delimiter=',')
    np.savetxt('data/mortality-vs-fertility-rank-1-approx.csv', rank_1_approx(data), delimiter=',')
    print(V)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
