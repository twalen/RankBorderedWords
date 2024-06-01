"""
Fast implementation of Ranking and Unranking.
"""

from functools import cache
from typing import Union, Optional

from rank_unrank.base import BaseRankerUnranker
from rank_unrank.texts import (
    calculate_all_periods_of_all_prefixes,
    calculate_all_periods_single_step,
    calc_a_b,
    PeriodsGroup,
)
from rank_unrank.rank_fast_subproblems import SumQueries, implicit_problem


class WordTool:
    def __init__(self, seq: Union[list[int], tuple[int]], k: int) -> None:
        self.k = k
        self.n = len(seq)
        self.seq = seq
        self.w = (-1,) + tuple(seq)
        self.periods = calculate_all_periods_of_all_prefixes(seq)

    def U(self, u_len: int, c: int, n: int) -> int:
        return self.k ** (n - u_len) - self.B(u_len, c, n)

    def B(self, u_len: int, c: int, n: int) -> int:
        assert 1 <= u_len <= n
        a, _ = self.calc_a_b(u_len - 1, c)
        if u_len > 0:
            periods = calculate_all_periods_single_step(
                self.seq, u_len - 1, c, self.periods[u_len - 1]
            )
        else:
            periods = PeriodsGroup([])
        borders = periods.to_borders(u_len)
        sum_queries = SumQueries(a, borders)

        def precalc_B() -> list[Optional[int]]:
            b1 = [None] * (2 * u_len + 1)
            b1[u_len] = 0
            for n in range(u_len + 1, 2 * u_len + 1):
                b1[n] = b1[n - 1] * self.k + a[n - u_len]

            b2 = [None] * (2 * u_len + 1)
            for n in range(u_len, 2 * u_len + 1):
                b2[n] = sum_queries.sum(n - u_len + 1, n // 2, n - u_len)
            result = [None] * (2 * u_len + 1)
            for n in range(u_len, 2 * u_len + 1):
                result[n] = b1[n] + b2[n]
            return result

        @cache
        def U(n: int) -> int:
            if n <= 2 * u_len:
                return self.k ** (n - u_len) - B(n)
            else:
                assert n > 2 * u_len
                if n % 2 == 1:
                    return self.k * U(n - 1)
                else:
                    return self.k * U(n - 1) - U(n // 2)

        def precalc_U(n):
            for i in range(u_len, n + 1):
                U(i)

        @cache
        def B(n: int) -> int:
            nonlocal B_cache
            if n <= 2 * u_len:
                assert B_cache[n] is not None
                return B_cache[n]
            else:
                return self.k ** (n - u_len) - U(n)

        B_cache = precalc_B()
        precalc_U(n)  # cache all U values
        return B(n)

    def X(self, i: int) -> list[int]:
        assert 0 <= i <= self.n
        if self.n == 0 or i == 0:
            return []
        result = [self.w[1]]
        for period_seq in self.periods[i]:
            result.append(self.w[i + 1 - period_seq.step])
        return sorted(set(result))

    def X_prim(self, i: int) -> tuple[int, Optional[int]]:
        assert 0 <= i <= self.n
        x_set = set(self.X(i))
        count = self.k - len(x_set)
        repr_c = None
        for c in range(1, self.k + 1):
            if c not in x_set:
                repr_c = c
                break
        return (count, repr_c)

    def Y(self, i: int) -> list[int]:
        assert 0 <= i < self.n
        result = []
        for c in self.X(i):
            if c < self.w[i + 1]:
                result.append(c)
        return result

    def Y_prim(self, i: int) -> tuple[int, Optional[int]]:
        assert 0 <= i < self.n
        y_set = set(self.Y(i))
        count = self.w[i + 1] - 1 - len(y_set)
        repr_c = None
        for c in range(1, self.w[i + 1]):
            if c not in y_set:
                repr_c = c
                break
        return (count, repr_c)

    def calc_a_b(self, i: int, c: int) -> tuple[list[int], list[int]]:
        seq = self.w[1 : i + 1] + (c,)
        return calc_a_b(seq)


class FasterRankerUnranker(BaseRankerUnranker):
    def _rank_b(self, w: list[int], n: int, k: int) -> int:
        result = 0
        wt = WordTool(list(w[1:]), k)
        for i in range(0, n):
            y_i = wt.Y(i)
            y_prim_len, y_prim_repr = wt.Y_prim(i)

            for c in y_i:
                result += wt.B(i + 1, c, n)
            if y_prim_len != 0:
                result += y_prim_len * wt.B(i + 1, y_prim_repr, n)

        return result + 1

    def _rank_u(self, w: list[int], n: int, k: int) -> int:
        result = 0
        for i in range(1, n + 1):
            result += (w[i] - 1) * (k ** (n - i))
        return 2 + result - self._rank_b(w, n, k)

    def rank(self, seq: list[int], k: int, bordered: bool) -> int:
        n = len(seq)
        w = [0] + list(seq)
        if bordered:
            return self._rank_b(w, n, k)
        else:
            return self._rank_u(w, n, k)

    def unrank(self, r: int, n: int, k: int, bordered: bool) -> tuple[int]:
        w = []
        for i in range(1, n + 1):
            wt = WordTool(w, k)
            X = wt.X(i - 1)

            B_values = {}
            for c in X:
                B_values[c] = wt.B(i, c, n) if bordered else wt.U(i, c, n)

            _, X_prim_repr = wt.X_prim(i - 1)
            B_prim_value = None
            if X_prim_repr is not None:
                B_prim_value = (
                    wt.B(i, X_prim_repr, n) if bordered else wt.U(i, X_prim_repr, n)
                )

            (r, c) = implicit_problem(B_values, B_prim_value, k, r)
            w.append(c)
        assert r == 1, f"invalid final rank: {r}"
        return tuple(w)
