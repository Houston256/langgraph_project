# Eval cases

Local and gitignored. Each `*.py` file here is one set with a `CASES` list of `assistant.evals.case.Case`; `--set <name>` runs one set.

```python
from assistant.evals.case import Case

CASES = [
    Case("women's zip hoodie", ["Hledám dámskou mikinu na zip do 3000 Kč."], min_cards=2, max_price=3000),
]
```
