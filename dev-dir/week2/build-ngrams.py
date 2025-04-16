import io
import argparse
import pandas as pd

from itertools import starmap, combinations
from sklearn.feature_extraction.text import CountVectorizer

def generalized_set_similarity_measure(
    A: set, B: set, C: set, D: set,
    x: int, y: int, z: int, z_prime: int
) -> float:
    a = len(A.intersection(B))
    b = len(A.union(B).union(C).union(D)) - len(A.union(B))
    c = len(A.symmetric_difference(B))
    d = x * a + y * b + z * c
    e = x * a + y * b + z_prime * c
    return d / e

def read_file(file: io.TextIOWrapper) -> list:
    data = pd.read_csv(file)
    corpus = list(data.text)
    corpus = corpus[:4]
    return corpus

def build_ngrams(corpus: str, ngram: int, token: str) -> set:
    vectorizer = CountVectorizer(analyzer=token, ngram_range=(ngram, ngram))
    _ = vectorizer.fit([corpus])
    features = set(vectorizer.get_feature_names_out())
    return features

def main() -> int:
    parser = argparse.ArgumentParser(prog='build-ngram', description='n-gram document builder')
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), required=True, help='path to input file')
    parser.add_argument('-n', '--ngram', type=int, required=True, help='n-gram size')
    parser.add_argument('-t', '--token', type=str, required=True, choices=['char', 'word'], help='token to use')

    args = parser.parse_args()
    corpus = read_file(args.input)
    corpus_size = len(corpus)

    ngrams = list(
        starmap(
            build_ngrams,
            zip(corpus, corpus_size * [args.ngram], corpus_size * [args.token])
        )
    )

    for fst_ngram, snd_ngram in combinations(ngrams, 2):
        thd_ngram, fth_ngram = (ngram for ngram in ngrams if ngram not in (fst_ngram, snd_ngram))
        similarity = generalized_set_similarity_measure(
            fst_ngram, snd_ngram,
            thd_ngram, fth_ngram,
            1, 0, 0, 1
        )
        print(similarity)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
