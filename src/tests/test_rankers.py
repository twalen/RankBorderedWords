from dataclasses import dataclass
import itertools
import pytest
import random
from rank_unrank.texts import is_bordered, calc_B_naive, random_word
from rank_unrank.base import BaseRankerUnranker
import rank_unrank.rank_ub as ub
from rank_unrank.rank_ub import UBRankerUnranker
from rank_unrank.rank_fast import FasterRankerUnranker, WordTool


@dataclass
class SeqTest:
    seq: tuple[int]
    bordered: bool
    rank_bordered: int
    rank_unbordered: int


@pytest.mark.parametrize(
    "test_cls,n,k",
    [
        (BaseRankerUnranker, 8, 2),
        (UBRankerUnranker, 12, 2),
        (FasterRankerUnranker, 12, 2),
    ],
)
def test_rankers_on_all_seq(test_cls, n, k):
    r = test_cls()
    rank_bordered, rank_unbordered = 1, 1
    for seq in itertools.product(range(1, k + 1), repeat=n):
        bordered = is_bordered(seq)
        seq_test = SeqTest(seq, bordered, rank_bordered, rank_unbordered)
        if bordered:
            rank_bordered += 1
        else:
            rank_unbordered += 1

        assert r.rank(seq_test.seq, k=k, bordered=True) == seq_test.rank_bordered
        assert r.rank(seq_test.seq, k=k, bordered=False) == seq_test.rank_unbordered
        if seq_test.bordered:
            assert seq_test.seq == r.unrank(seq_test.rank_bordered, n, k, bordered=True)
        else:
            assert seq_test.seq == r.unrank(
                seq_test.rank_unbordered, n, k, bordered=False
            )


@pytest.mark.parametrize("n,k,count", [(50, 2, 10)])
def test_rankers_on_long_random_seq(n, k, count):
    r_ub = UBRankerUnranker()
    r_fast = FasterRankerUnranker()
    for i in range(count):
        random.seed(n * 100 + 10 * k + i)
        w = random_word(n, k)
        bordered = is_bordered(w)
        expected_bordered = r_ub.rank(w, k=k, bordered=True)
        expected_unbordered = r_ub.rank(w, k=k, bordered=False)
        assert expected_bordered == r_fast.rank(w, k=k, bordered=True)
        assert expected_unbordered == r_fast.rank(w, k=k, bordered=False)

        r = expected_bordered if bordered else expected_unbordered
        assert r_ub.unrank(r, n, k, bordered) == w
        assert r_fast.unrank(r, n, k, bordered) == w


@pytest.mark.parametrize("n,k", [(8, 2), (7, 2), (5, 3), (6, 4)])
def test_calc_b(n, k):
    for m in range(1, n + 1):
        for u in itertools.product(range(1, k + 1), repeat=m):
            expected = calc_B_naive(u, k, n)
            w = [0] + list(u)
            a, b = ub.populateBorderArrays(w, m)
            result_ub = ub.B(a, b, n, m, k)
            assert result_ub == expected

            r = WordTool(u[:-1], k)
            result_r = r.B(m, u[-1], n)
            assert result_r == expected
