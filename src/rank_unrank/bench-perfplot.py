#!/usr/bin/env python
import sys
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import perfplot

#
import rank_unrank
from rank_unrank.rank_ub import UBRankerUnranker
from rank_unrank.rank_fast import FasterRankerUnranker


def rank_gabric(w, k):
    return UBRankerUnranker().rank(w, k, True)


def rank_faster(w, k):
    return FasterRankerUnranker().rank(w, k, True)


def unrank_gabric(r, n, k):
    return UBRankerUnranker().unrank(r, n, k, True)


def unrank_faster(r, n, k):
    return FasterRankerUnranker().unrank(r, n, k, True)


def gen_benchmarks(
    output_fn: str, title: str, setup_func: callable, rank: bool = True
) -> None:
    def equality_check(*res):
        return all([x == res[0] for x in res])

    if rank:
        kernels = [rank_gabric, rank_faster]
        xlabel = "word size"
        n_range = range(100, 601, 100)
    else:
        kernels = [unrank_gabric, unrank_faster]
        xlabel = "n"
        n_range = range(50, 201, 50)

    out = perfplot.bench(
        setup=setup_func,
        kernels=kernels,
        labels=["Gabric algorithm", "New algorithm"],
        n_range=n_range,
        xlabel=xlabel,
        equality_check=equality_check,
    )
    ax = plt.gca()
    ax.xaxis.set_ticks(n_range)
    ax.xaxis.set_major_formatter(mt.StrMethodFormatter("{x}"))
    ax.set_title(title)
    out.save(output_fn, transparent=True, bbox_inches="tight", logx=False, logy=True)


def main():
    sys.setrecursionlimit(10**5)

    def setup_unrank(n, k):
        r = k ** (n - 10)
        return (r, n, k)

    def setup_random(n, k):
        w = rank_unrank.texts.random_word(n, k, seed=n)
        return (w, k)

    print("k=2 unrank")
    gen_benchmarks(
        "/tmp/unrank-perf-k2.png", "unrank k=2", partial(setup_unrank, k=2), rank=False
    )
    print("k=2 rank")
    gen_benchmarks("/tmp/rank-perf-k2.png", "rank k=2", partial(setup_random, k=2))
    print("k=5 unrank")
    gen_benchmarks(
        "/tmp/unrank-perf-k5.png", "unrank k=5", partial(setup_unrank, k=5), rank=False
    )
    print("k=5 rank")
    gen_benchmarks("/tmp/rank-perf-k5.png", "rank k=5", partial(setup_random, k=5))


if __name__ == "__main__":
    main()
