import itertools
from rank_unrank.texts import is_bordered


class SeqGenerator:
    def __init__(self, n: int, alphabet_size: int, bordered: bool) -> None:
        self.n = n
        self.k = alphabet_size
        self.bordered = bordered

    def __iter__(self):
        self.it = itertools.product(range(1, self.k + 1), repeat=self.n)
        return self

    def __next__(self):
        while True:
            seq = next(self.it)
            if is_bordered(seq) == self.bordered:
                return seq


class AbstractRankerUnranker:
    def rank(self, seq: list[int], k: int, bordered: bool) -> int:
        raise NotImplementedError()

    def unrank(self, r: int, n: int, k: int, bordered: bool) -> list[int]:
        raise NotImplementedError()


class BaseRankerUnranker(AbstractRankerUnranker):
    def rank(self, seq: list[int], k: int, bordered: bool) -> int:
        g = SeqGenerator(len(seq), k, bordered)
        res = 1
        for s in g:
            if s < seq:
                res += 1
            else:
                break
        return res

    def unrank(self, r: int, n: int, k: int, bordered: bool) -> list[int]:
        g = SeqGenerator(n, k, bordered)
        for i, s in enumerate(g, start=1):
            if i == r:
                return s


if __name__ == "__main__":
    bordered = False
    k = 2
    n = 5
    g = SeqGenerator(n, k, bordered)
    r = BaseRankerUnranker()
    for i, seq in enumerate(g, start=1):
        print(i, seq, r.rank(seq, k, bordered), r.unrank(i, n, k, bordered))
