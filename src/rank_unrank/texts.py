from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import Union, Optional
import random


def random_word(n: int, k: int, seed: Optional[int] = None) -> tuple[int]:
    if seed is not None:
        random.seed(seed)
    return tuple(random.randint(1, k) for _ in range(n))


def fib_word(n: int) -> tuple[int]:
    if n == 0:
        return (1,)
    elif n == 1:
        return (1, 2)
    else:
        return fib_word(n - 1) + fib_word(n - 2)


def is_bordered_naive(seq: list[int]) -> bool:
    """Returns true if a sequence has a border (naive implementation)

    Time complexity: O(n^2)"""
    for i in range(1, len(seq)):
        if seq[:i] == seq[-i:]:
            return True
    return False


def calc_pi(seq: list[int]) -> list[int]:
    """Calculates prefix function of a sequence
    pi[i] = max(l : 0 <= l <= i && seq[:l] == seq[-l-i:i])

    Time complexity: O(n)"""
    n = len(seq)
    pi = [0] * n
    for i, c in enumerate(seq[1:], start=1):
        j = pi[i - 1]
        while j > 0 and seq[j] != c:
            j = pi[j - 1]
        if seq[j] == c:
            j += 1
        pi[i] = j
    return pi


def is_bordered(seq: list[int]) -> bool:
    """Returns true if a sequence has a border (non empty prefix that is also an suffix)

    Time complexity: O(n)"""
    if len(seq) > 0:
        pi = calc_pi(seq)
        return pi[len(seq) - 1] != 0
    else:
        return False


def calculate_all_periods_naive(seq: list[int]) -> list[int]:
    """Calculate all periods of a sequence

    Time complexity: O(n^2)"""
    n = len(seq)
    res = []
    for p in range(1, n):
        if seq[:-p] == seq[p:]:
            res.append(p)
    return res


@dataclass
class ArithSequence:
    start: int
    end: int
    step: int

    def __iter__(self):
        i = self.start
        while i <= self.end:
            yield i
            i += self.step

    @property
    def first(self) -> Optional[int]:
        if self.start <= self.end:
            return self.start
        else:
            return None

    @property
    def last(self) -> Optional[int]:
        if self.start <= self.end:
            return self.start + ((self.end - self.start) // self.step) * self.step
        else:
            return None

    def normalize(self) -> ArithSequence:
        if self.start <= self.end:
            if self.end != self.last:
                return ArithSequence(self.start, self.last, self.step)
            else:
                return self
        else:
            return ArithSequence(1, 0, self.step)

    def count(self) -> int:
        assert self.step > 0
        return max(0, (self.end + self.step - self.start) // self.step)

    def intersection(self, a: int, b: int) -> ArithSequence:
        """intersection of the arith. sequence and [a..b]"""
        assert a <= b
        if self.start > self.end:
            return self
        diff_a = (a - self.start + self.step - 1) // self.step
        new_start = self.start + self.step * diff_a if a >= self.start else self.start
        new_end = b if b <= self.end else self.end
        return ArithSequence(new_start, new_end, self.step).normalize()

    def add_next_step(self):
        return ArithSequence(self.start, self.end + self.step, self.step)


@dataclass
class PeriodsGroup:
    periods: list[ArithSequence]

    def __iter__(self):
        return iter(self.periods)

    def to_borders(self, n):
        return [
            ArithSequence(n - seq.end, n - seq.start, seq.step) for seq in self.periods
        ]

    def all_periods(self):
        for period in self.periods:
            for value in period:
                yield value


def calculate_all_periods_single_step(
    seq: Union[list[int], tuple[int]],
    n: int,
    curr_elem: int,
    periods: PeriodsGroup,
) -> PeriodsGroup:
    """calculate periods of word: seq[:n] + (curr_elem,)"""
    assert len(seq) >= n and n >= 0
    if n == 0:
        return PeriodsGroup([])
    added = False
    curr_periods = []
    for period_seq in periods:
        assert (
            period_seq.start == period_seq.step
            and period_seq.end + period_seq.step >= n
        )
        if curr_elem == seq[n - period_seq.step]:  # it does extend
            if period_seq.end + period_seq.step == n:
                curr_periods.append(period_seq.add_next_step())
                added = True
            else:
                curr_periods.append(period_seq)
    if not added and seq[0] == curr_elem:
        curr_periods.append(ArithSequence(n, n, n))
    return PeriodsGroup(curr_periods)


def calculate_all_periods_of_all_prefixes(seq: list[int]) -> list[PeriodsGroup]:
    n = len(seq)
    res = [None for _ in range(n + 1)]
    res[0] = PeriodsGroup([])
    if n > 0:
        res[1] = PeriodsGroup([])
    for i in range(1, n):
        curr_elem = seq[i]
        res[i + 1] = calculate_all_periods_single_step(seq, i, curr_elem, res[i])
    return res


def calc_a_b_naive(seq: list[int]) -> tuple[list[int], list[int]]:
    """Naive implementation
    a[i]=1 iff seq[:i] is unbordered
    b[i]=1 iff seq has a border of length i

     Time complexity: O(n^2)"""
    n = len(seq)
    a = [0] * (n + 1)
    b = [0] * (n + 1)
    for i in range(1, n + 1):
        a[i] = int(not is_bordered(seq[:i]))  # seq[:i] is not bordered
    for i in range(1, n):
        b[i] = int(seq[:i] == seq[-i:])  # has a border of length i
    return a, b


def calc_a_b(seq: list[int]) -> tuple[list[int], list[int]]:
    """a[i]=1 iff seq[:i] is unbordered
    b[i]=1 iff seq has a border of length i

    Time complexity: O(n)"""
    n = len(seq)
    a = [0] * (n + 1)
    b = [0] * (n + 1)
    pi = calc_pi(seq)
    for i in range(1, n + 1):
        if pi[i - 1] == 0:
            a[i] = 1
    i = n
    while i > 0 and pi[i - 1] > 0:
        b[pi[i - 1]] = 1
        i = pi[i - 1]
    return a, b


def calc_B_naive(u: Union[list[int], tuple[int]], k: int, n: int) -> int:
    """count of bordered words w: {1..k}^n with a prefix u (naive implementation)"""
    if isinstance(u, list):
        u = tuple(u)
    result = 0
    assert len(u) <= n
    for uu in itertools.product(range(1, k + 1), repeat=n - len(u)):
        if is_bordered(u + uu):
            result += 1
    return result


def calc_U_naive(u: Union[list[int], tuple[int]], k: int, n: int) -> int:
    """count of unbordered words w: {1..k}^n with a prefix u (naive implementation)"""
    if isinstance(u, list):
        u = tuple(u)
    result = 0
    assert len(u) <= n
    for uu in itertools.product(range(1, k + 1), repeat=n - len(u)):
        if not is_bordered(u + uu):
            result += 1
    return result


def calc_X_naive(u: Union[list[int], tuple[int]], k: int) -> list[int]:
    """returns sorted list of charactes c, such that u+c is bordered"""
    if isinstance(u, tuple):
        u = list(u)
    res = []
    for c in range(1, k + 1):
        if is_bordered(u + [c]):
            res.append(c)
    return res
