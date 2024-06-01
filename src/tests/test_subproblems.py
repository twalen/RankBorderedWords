import itertools
import pytest
import random
from rank_unrank.texts import calc_a_b, calculate_all_periods_of_all_prefixes
from rank_unrank.rank_fast_subproblems import (
    SumQueries,
    NaiveSumQueries,
    implicit_problem,
    implicit_problem_linear,
)


@pytest.mark.parametrize("n,k", [(8, 2), (7, 3), (5, 4)])
def test_sum_queries(n, k):
    for seq in itertools.product(range(1, k + 1), repeat=n):
        a, b = calc_a_b(seq)
        borders = calculate_all_periods_of_all_prefixes(seq)[-1].to_borders(n)

        n_sq = NaiveSumQueries(a, b)
        sq = SumQueries(a, borders)
        for p, q in itertools.combinations(range(1, n + 1), r=2):
            for d in range(1, p):
                assert n_sq.sum(p, q, d) == sq.sum(p, q, d)


@pytest.mark.parametrize("k,values_count", [(80, 10), (100, 20), (500, 30)])
def test_implicit_problem(k, values_count):
    for i in range(100):
        random.seed(k * 10 + values_count * 17 + i)
        keys = random.sample(range(1, k + 1), values_count)
        values = {k: random.randint(1, 100) for k in keys}
        default_value = random.randint(1, 100)

        r = random.randint(1, sum(values.values()) + default_value * (k - values_count))
        kwargs = {"values": values, "default_value": default_value, "k": k, "r": r}
        assert implicit_problem_linear(**kwargs) == implicit_problem(**kwargs)
