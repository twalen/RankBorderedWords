import itertools
import pytest
from rank_unrank.texts import (
    is_bordered,
    is_bordered_naive,
    calculate_all_periods_naive,
    calculate_all_periods_of_all_prefixes,
    calc_X_naive,
    calc_a_b_naive,
    calc_a_b,
    ArithSequence,
)
from rank_unrank.rank_ub import populateBorderArrays
from rank_unrank.rank_fast import WordTool


@pytest.mark.parametrize(
    "seq,expected_answer",
    [
        ([1, 2, 2], False),
        ([1, 2, 1], True),
        ([1], False),
    ],
)
def test_is_bordered(seq, expected_answer):
    assert is_bordered_naive(seq) == expected_answer
    assert is_bordered(seq) == expected_answer


def test_is_bordered_binary():
    for seq in itertools.product([1, 2], repeat=7):
        assert is_bordered(seq) == is_bordered_naive(seq)


def test_arith_sequence():
    s1 = ArithSequence(3, 9, 2)
    assert list(s1) == [3, 5, 7, 9]
    assert s1.count() == 4
    assert s1.first == 3
    assert s1.last == 9
    s2 = ArithSequence(3, 10, 2)
    assert list(s2) == [3, 5, 7, 9]
    assert s2.count() == 4
    assert s2.first == 3
    assert s2.last == 9

    assert list(s1.intersection(-10, -5)) == []
    assert list(s1.intersection(-10, 20)) == [3, 5, 7, 9]
    assert list(s1.intersection(5, 7)) == [5, 7]
    assert list(s1.intersection(4, 7)) == [5, 7]
    assert list(s1.intersection(5, 8)) == [5, 7]


@pytest.mark.parametrize(
    "seq,expected_answer",
    [
        ([1, 2, 3], []),
        ([1, 2, 3, 1], [3]),
        ([1, 2, 1, 2, 1, 2, 1], [2, 4, 6]),
    ],
)
def test_all_periods(seq, expected_answer):
    assert calculate_all_periods_naive(seq) == expected_answer
    all_periods = calculate_all_periods_of_all_prefixes(seq)
    seq_periods = list(all_periods[len(seq)].all_periods())
    assert set(seq_periods) == set(expected_answer)
    assert len(seq_periods) == len(expected_answer)


def test_all_periods_of_prefixes_on_all_binary_seq():
    n, k = 10, 2
    for seq in itertools.product(range(1, k + 1), repeat=n):
        expected_answer = calculate_all_periods_naive(seq)
        all_periods = calculate_all_periods_of_all_prefixes(seq)
        seq_periods = list(all_periods[len(seq)].all_periods())
        assert set(seq_periods) == set(expected_answer)
        assert len(seq_periods) == len(expected_answer)


@pytest.mark.parametrize(
    "seq,k,expected_answer",
    [
        ([1, 2, 3], 3, [1]),
        ([1, 2, 1], 2, [1, 2]),
    ],
)
def test_calc_X(seq, k, expected_answer):
    assert calc_X_naive(seq, k) == expected_answer
    assert WordTool(seq, k).X(len(seq)) == expected_answer


@pytest.mark.parametrize("n,k", [(8, 2)])
def test_calc_X_all_seq(n, k):
    for seq in itertools.product(range(1, k + 1), repeat=n):
        expected_answer = calc_X_naive(seq, k)
        assert WordTool(seq, k).X(len(seq)) == expected_answer


@pytest.mark.parametrize("n,k", [(8, 2), (7, 3), (5, 4)])
def test_calc_a_b(n, k):
    for seq in itertools.product(range(1, k + 1), repeat=n):
        expected_a, expected_b = calc_a_b_naive(seq)

        a1, b1 = calc_a_b(seq)
        assert expected_a == a1, f"wrong a1, on seq: {seq}"
        assert expected_b == b1, f"wrong b1, on seq: {seq}"

        w = (0,) + seq
        a2, b2 = populateBorderArrays(w, n)
        assert expected_a == a2
        assert expected_b == b2
