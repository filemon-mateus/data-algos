import io
import time
import random
import string
import argparse
import numpy as np

from itertools import starmap
from sklearn.feature_extraction.text import CountVectorizer

class Hash:
    def __init__(self, salt: str) -> None:
        self.salt = salt

    def hash_fn(self, item: str) -> int:
        return hash((self.salt, item))

def read_file(file: io.TextIOWrapper) -> list:
    corpus = file.read().splitlines()
    return corpus

def build_ngrams(corpus: list, ngram: int, token: str) -> set:
    vectorizer = CountVectorizer(analyzer=token, ngram_range=(ngram, ngram))
    _ = vectorizer.fit(corpus)
    features = set(vectorizer.get_feature_names_out())
    return features

def rand_salt(length: int) -> str:
    return ''.join(random.sample(string.ascii_letters + string.digits, length))

def min_hash_sign(s: set, hash_fns: list) -> list:
    sign_s = []
    for hash_fn in hash_fns:
        min_hash = min(hash_fn(item) for item in s)
        sign_s.append(min_hash)
    return sign_s

def jaccard_similarity(a: set, b: set) -> float:
    return len(a.intersection(b)) / len(a.union(b))

def minhash_similarity(a: set, b: set, t: int) -> float:
    hashes = [Hash(rand_salt(9)).hash_fn for _ in range(t)]
    sign_a = min_hash_sign(a, hashes)
    sign_b = min_hash_sign(b, hashes)
    return sum(1 for a, b in zip(sign_a, sign_b) if a == b) / t

def run_similarity(ngrams: list) -> None:
    for t in [20, 60, 150, 300, 600]:
        ms = minhash_similarity(ngrams[0], ngrams[1], t)
        js = jaccard_similarity(ngrams[0], ngrams[1])
        print(f'{t=:03}, {ms=:.3f}, {js=:.3f}')

def run_benchmark(ngrams: list, nsims: int) -> None:
    ts = np.linspace(1, 600, 200, dtype=int)
    js = jaccard_similarity(ngrams[0], ngrams[1])

    avg_times = np.zeros_like(ts, dtype=float)
    avg_error = np.zeros_like(ts, dtype=float)

    for i, t in enumerate(ts):
        error = np.zeros(nsims)
        times = np.zeros(nsims)

        for j in range(nsims):
            tic = time.perf_counter()
            ms = minhash_similarity(ngrams[0], ngrams[1], t)
            toc = time.perf_counter()
            error[j] = abs(ms - js)
            times[j] = toc - tic

        avg_error[i] = np.mean(error)
        avg_times[i] = np.mean(times)

    np.savetxt('data/times.csv', np.column_stack((ts, avg_times)), delimiter=',')
    np.savetxt('data/error.csv', np.column_stack((ts, avg_error)), delimiter=',')

def main() -> int:
    parser = argparse.ArgumentParser(prog='fast-min-hash', description='implements a fully fledge minhash algorithm')

    subparsers = parser.add_subparsers(dest='command')

    similarity_parser = subparsers.add_parser('similarity', description='approximates document similarity')
    similarity_parser.add_argument('-i', '--input', nargs=2, type=argparse.FileType('r'), required=True, help='path to input files')
    similarity_parser.add_argument('-n', '--ngram', type=int, required=True, help='ngram size')
    similarity_parser.add_argument('-t', '--token', type=str, required=True, choices=['char', 'word'], help='token to use')

    benchmark_parser = subparsers.add_parser('benchmark', description='runs a benchmark')
    benchmark_parser.add_argument('-i', '--input', nargs=2, type=argparse.FileType('r'), required=True, help='path to input files')
    benchmark_parser.add_argument('-n', '--ngram', type=int, required=True, help='ngram size')
    benchmark_parser.add_argument('-t', '--token', type=str, required=True, choices=['char', 'word'], help='token to use')
    benchmark_parser.add_argument('-s', '--nsims', type=int, required=True, help='number of times to repeat experiment')

    args = parser.parse_args()
    ngrams = list(
        starmap(
            build_ngrams,
            zip(map(read_file, args.input), [args.ngram] * len(args.input), [args.token] * len(args.input))
        )
    )

    match args.command:
        case 'similarity':
            run_similarity(ngrams)
        case 'benchmark':
            run_benchmark(ngrams, args.nsims)
        case _:
            parser.print_usage()

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
