from rank_unrank.base import AbstractRankerUnranker


###############
# Adapted code from https://github.com/DanielGabric/RankUnrank
# WARNING! arrays are 1-indexed (0-index is unused)


def power(k, n):
    return k**n


def B(a: list[int], b: list[int], n: int, p: int, k: int) -> int:
    saveB = [0] * (n + 1)
    if n <= 2 * p:
        total = 0
        for i in range(1, n - p + 1):
            total += a[i] * power(k, n - p - i)
        for i in range(n - p + 1, n // 2 + 1):
            total += a[i] * b[i - (n - p)]
        return total
    else:
        for j in range(p, 2 * p + 1):
            saveB[j] = B(a, b, j, p, k)
        for j in range(2 * p + 1, n + 1):
            saveB[j] = 0
            for i in range(1, p + 1):
                saveB[j] += a[i] * power(k, j - p - i)
            for i in range(p + 1, j // 2 + 1):
                saveB[j] += (power(k, i - p) - saveB[i]) * power(k, j - 2 * i)
        return saveB[n]


def populateBorderArrays(w: list[int], n: int):
    PBA = [0] * (n + 1)
    a = [0] * (n + 1)
    b = [0] * (n + 1)

    a[1] = 1
    i = 2
    length = 0
    while i <= n:
        if w[i] == w[length + 1]:
            length += 1
            PBA[i] = length
            i += 1
        else:
            if length != 0:
                length = PBA[length]
            else:
                PBA[i] = 0
                i += 1
        a[i - 1] = PBA[i - 1] == 0

    i = n
    while PBA[i] > 0:
        b[PBA[i]] = 1
        i = PBA[i]
    return a, b


def rankB(w: list[int], n: int, k: int):
    result = 0
    for i in range(1, n + 1):
        save = w[i]
        for c in range(1, save):
            w[i] = c
            a, b = populateBorderArrays(w, i)
            result += B(a, b, n, i, k)
        w[i] = save
    return result + 1


def rankU(w: list[int], n: int, k: int):
    result = 0
    for i in range(1, n + 1):
        result += (w[i] - 1) * power(k, n - i)
    return 2 + result - rankB(w, n, k)


def unrank(rank: int, n: int, k: int, isB: bool) -> tuple[int]:
    w = [1] * (n + 1)
    for i in range(1, n + 1):
        left = 1
        right = k
        while left < right:
            save = w[i]
            mid = (left + right + 1) // 2
            w[i] = mid
            if isB:
                currRank = rankB(w, n, k)
            else:
                currRank = rankU(w, n, k)
            if currRank <= rank:
                left = mid
            else:
                w[i] = save
                right = mid - 1
    return tuple(w[1:])


###############


class UBRankerUnranker(AbstractRankerUnranker):
    def rank(self, seq: list[int], k: int, bordered: bool) -> int:
        w = [0] + list(seq)
        n = len(seq)
        if bordered:
            return rankB(w, n, k)
        else:
            return rankU(w, n, k)

    def unrank(self, r: int, n: int, k: int, bordered: bool) -> list[int]:
        return unrank(r, n, k, bordered)
