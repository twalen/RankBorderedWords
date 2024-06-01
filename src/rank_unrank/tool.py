#!/usr/bin/env python
import re
import click

from rank_unrank.base import BaseRankerUnranker
from rank_unrank.rank_ub import UBRankerUnranker
from rank_unrank.rank_fast import FasterRankerUnranker
from rank_unrank.texts import is_bordered


@click.group()
def cli():
    pass


def parse_seq_from_string(s: str) -> tuple[int]:
    res = tuple(map(int, re.split(r"[\s,]+", s)))
    assert len(res) == 0 or min(res) >= 1
    return res


def gen_ranker(alg):
    return {
        "base": BaseRankerUnranker,
        "ub": UBRankerUnranker,
        "fast": FasterRankerUnranker,
    }[alg]()


@cli.command()
@click.option("-k", "--alphabet", type=int)
@click.option("--alg", type=click.Choice(["base", "ub", "fast"]))
@click.argument("seq", type=str)
def rank(alphabet, alg, seq):
    s = parse_seq_from_string(seq)
    assert len(s) > 0 and min(s) >= 1 and max(s) <= alphabet
    bordered = is_bordered(s)
    length = len(s)
    click.echo(
        f"[DEBUG] rank n={length} k={alphabet} bordered={bordered} seq={s} alg={alg}"
    )
    ranker = gen_ranker(alg)
    res = ranker.rank(s, alphabet, bordered)
    print(res)


@cli.command()
@click.option("-n", "--length", type=int, default=10)
@click.option("-k", "--alphabet", type=int, default=2)
@click.option("--bordered/--not-bordered", is_flag=True, default=True)
@click.option("--alg", type=click.Choice(["base", "ub", "fast"]))
@click.argument("R", type=int, default=1)
def unrank(length, alphabet, bordered, alg, r):
    click.echo(f"[DEBUG] unrank n={length} k={alphabet} bordered={bordered} alg={alg}")
    ranker = gen_ranker(alg)
    res = ranker.unrank(r, length, alphabet, bordered)
    print(" ".join(map(str, res)))


if __name__ == "__main__":
    cli()
