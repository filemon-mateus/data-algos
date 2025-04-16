import time
import argparse
import numpy as np

from numba import njit, prange

@njit('float64[:](int64, int64)', parallel=True)
def simulation(n: int, m: int) -> np.ndarray:
    results = np.zeros(m)
    for i in prange(m):
        seen = np.zeros(n)
        count = 0
        while count < np.inf:
            index = np.random.randint(n)
            count = count + 1
            if seen[index] == 1:
                break
            seen[index] = 1
        results[i] = count
    return results

def run_simulation(args: argparse.Namespace) -> None:
    trials = simulation(args.n, args.m)
    print(trials)

def compute_expectation(args: argparse.Namespace) -> None:
    trials = simulation(args.n, args.m)
    print(trials.mean())

def export_cdf_values(args: argparse.Namespace) -> None:
    trials = simulation(args.n, args.m)
    np.savetxt('data/bp-data/trials.csv', trials, delimiter=',', fmt='%d')

def measure_duration(args: argparse.Namespace) -> None:
    tic = time.perf_counter()
    _ = simulation(args.n, args.m)
    toc = time.perf_counter()
    print(toc - tic)

def run_benchmark(args: argparse.Namespace) -> None:
    ns = np.linspace(10_000, 1_000_000, args.n, dtype=int)
    ms = np.linspace(500, 10_000, args.m, dtype=int)
    results = np.zeros((args.n * args.m, 3))

    for i, m in enumerate(ms):
        for j, n in enumerate(ns):
            tic = time.perf_counter()
            _ = simulation(n, m)
            toc = time.perf_counter()
            results[i * args.n + j] = [n, m, toc - tic]

    np.savetxt('data/bp-data/bench.csv', results, delimiter=',', header='n,m,t', comments='', fmt=['%d','%d','%.4f'])

def add_common_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument('-n', type=int, required=True, help='specify domain [n]')
    subparser.add_argument('-m', type=int, required=True, help='specify number of trials')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='birthday-paradox', description='birthday paradox simulation')

    subparsers = parser.add_subparsers(dest='command')

    for cmd in ['simulate', 'expectation', 'get-cdf-vals', 'duration']:
        add_common_args(subparsers.add_parser(cmd, description=f'runs the {cmd} command'))

    benchmark_parser = subparsers.add_parser('benchmark', description='runs a benchmark')
    benchmark_parser.add_argument('-n', type=int, required=True, help='specify number of ns')
    benchmark_parser.add_argument('-m', type=int, required=True, help='specify number of ms')

    return parser.parse_args()

def main() -> int:
    args = parse_args()

    cmd_map = {
        'simulate': run_simulation,
        'expectation': compute_expectation,
        'get-cdf-vals': export_cdf_values,
        'duration': measure_duration,
        'benchmark': run_benchmark
    }

    cmd_map[args.command](args)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
