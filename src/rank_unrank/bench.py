#!/usr/bin/env python
import random
import pyperf


import rank_unrank
from rank_unrank.texts import random_word


def factory(w):
    def rank_f(w=w):
        rank_unrank.rank_fast.FasterRankerUnranker().rank(w, 2, True)

    def rank_ub(w=w):
        rank_unrank.rank_ub.UBRankerUnranker().rank(w, 2, True)

    return rank_f, rank_ub


def main(seed=42):
    random.seed(seed)

    w = random_word(500, 2)
    rank_f, rank_ub = factory(w)

    runner = pyperf.Runner()
    runner.bench_func("rank_f", rank_f)
    runner.bench_func("rank_ub", rank_ub)


if __name__ == "__main__":
    main()
