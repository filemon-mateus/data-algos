import numpy as np

B_VALUE = 11
R_VALUE = 19

class F:
    def __init__(self, b: int, r: int) -> None:
        self.b = b
        self.r = r

    def apply(self, s: np.ndarray) -> np.ndarray:
        return 1 - np.power(1 - np.power(s, self.b), self.r)

def main() -> int:
    s_values = np.linspace(0.0, 1.0, 1000)
    f_values = F(B_VALUE, R_VALUE).apply(s_values)
    np.savetxt('data/lsh-values.csv', np.column_stack((s_values, f_values)), delimiter=',')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
