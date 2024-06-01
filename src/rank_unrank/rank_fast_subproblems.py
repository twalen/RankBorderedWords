from typing import Optional
from rank_unrank.texts import ArithSequence


class AbstractSumQueries:
    def sum(self, p: int, q: int, d: int) -> int:
        raise NotImplementedError()


class NaiveSumQueries(AbstractSumQueries):
    """Just for tesing purposes"""

    def __init__(self, a: list[int], b: list[int]) -> None:
        assert len(a) == len(b)
        self.a = a
        self.b = b

    def sum(self, p: int, q: int, d: int) -> int:
        return sum([self.a[i] * self.b[i - d] for i in range(p, q + 1)], 0)


class SumQueries(AbstractSumQueries):
    """Actual implementation of Sum Queries"""

    def __init__(self, a: list[int], borders: list[ArithSequence]) -> None:
        self.a = a
        self.borders = borders
        self.cache_sum_a = dict()

    def calc_sum_a(self, i: int, delta: int) -> int:
        if i <= 0:
            return 0
        elif i <= delta:
            return self.a[i]
        else:
            if (i, delta) not in self.cache_sum_a:
                self.cache_sum_a[(i, delta)] = self.a[i] + self.calc_sum_a(
                    i - delta, delta
                )
            return self.cache_sum_a[(i, delta)]

    def sum(self, p: int, q: int, d: int) -> int:
        if p > q:
            return 0
        result = 0
        for seq in self.borders:
            curr = seq.intersection(p - d, q - d)
            if curr.count() > 0:
                result += self.calc_sum_a(curr.last + d, curr.step)
                result -= self.calc_sum_a(curr.first - curr.step + d, curr.step)
        return result


def implicit_problem_linear(
    values: dict[int, int], default_value: Optional[int], k: int, r: int
) -> tuple[int, int]:
    assert k >= 1
    assert all(1 <= key <= k for key in values.keys())
    assert default_value is not None or len(values) == k
    assert 1 <= r <= sum(values.get(c, default_value) for c in range(1, k + 1))
    new_r = r
    for c in range(1, k + 1):
        value = values.get(c, default_value)
        assert value is not None
        if value >= new_r:
            return (new_r, c)
        else:
            new_r -= value
    assert False  # unreachable


def implicit_problem(
    values: dict[int, int], default_value: Optional[int], k: int, r: int
) -> tuple[int, int]:
    assert k >= 1
    assert all(1 <= key <= k for key in values.keys()), f"k={k} values={values}"
    assert default_value is not None or len(values) == k

    new_r = r
    prev_c = 0
    for c in sorted(tuple(values.keys()) + (k + 1,)):
        skip_len = c - prev_c - 1
        if skip_len > 0:
            if skip_len * default_value >= new_r:
                d = (new_r + default_value - 1) // default_value
                assert 1 <= prev_c + d <= k
                return (new_r - (d - 1) * default_value, prev_c + d)
            else:
                new_r -= skip_len * default_value
        if values[c] >= new_r:
            return (new_r, c)
        else:
            new_r -= values[c]
        prev_c = c
    assert False  # unreachable
