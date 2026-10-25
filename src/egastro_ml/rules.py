"""Пошук асоціативних правил (FP-Growth) і рекомендації за вмістом кошика."""

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def encode(transactions: list[list[str]]) -> pd.DataFrame:
    """Транзакції -> бінарна матриця «транзакція × товар»."""
    te = TransactionEncoder()
    matrix = te.fit(transactions).transform(transactions)
    return pd.DataFrame(matrix, columns=te.columns_)


def mine_rules(
    transactions: list[list[str]],
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_antecedent_len: int = 2,
) -> pd.DataFrame:
    itemsets = fpgrowth(encode(transactions), min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[
        (rules["lift"] >= min_lift)
        & (rules["consequents"].apply(len) == 1)
        & (rules["antecedents"].apply(len) <= max_antecedent_len)
    ]
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)


def recommend(cart: set[str], rules: pd.DataFrame, top_k: int = 5) -> list[str]:
    """Бал кандидата = max confidence серед правил, чий антецедент ⊆ кошика."""
    scores: dict[str, float] = {}
    for row in rules.itertuples(index=False):
        if row.antecedents <= cart:
            item = next(iter(row.consequents))
            if item not in cart:
                scores[item] = max(scores.get(item, 0.0), row.confidence)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]]
