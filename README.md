# Rank & Unrank


# Installation

normal installation
```
pip install -e "."
```

development installation
```
pip install -e ".[dev]"
```


# Example usage

unranking:
```bash
$ rank-unrank unrank --alg=base -n 10 -k 2 50
[DEBUG] unrank n=10 k=2 bordered=True alg=base
1 1 1 2 1 2 2 2 1 1
```

ranking:
```bash
$ rank-unrank rank --alg=ub -k 2 "1 1 1 2 1 2 2 2 1 1"
[DEBUG] rank n=10 k=2 bordered=True seq=(1, 1, 1, 2, 1, 2, 2, 2, 1, 1) alg=ub
50
```
