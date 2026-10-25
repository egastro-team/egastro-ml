"""Перевірка наявності датасетів. Нічого не завантажує автоматично.

Джерела (кладемо вручну в data/raw):
  * Instacart Online Grocery Shopping Dataset 2017 —
    https://www.kaggle.com/c/instacart-market-basket-analysis
    -> data/raw/instacart/{orders,order_products__prior,products,aisles,departments}.csv
    Ліцензія: лише некомерційне використання.
  * Groceries (R-пакет arules, 9 835 чеків) —
    https://search.r-project.org/CRAN/refmans/arules/html/Groceries.html
    -> data/raw/groceries/groceries.csv

Запуск: uv run python -m egastro_ml.download_data
"""

import sys

from egastro_ml.config import load_config, project_path

EXPECTED = {
    "instacart": ["orders.csv", "order_products__prior.csv", "products.csv", "aisles.csv", "departments.csv"],
    "groceries": ["groceries.csv"],
}


def check() -> bool:
    cfg = load_config()
    ok = True
    for name, files in EXPECTED.items():
        folder = project_path(cfg["datasets"][name])
        missing = [f for f in files if not (folder / f).exists()]
        if missing:
            ok = False
            print(f"[{name}] немає у {folder}: {', '.join(missing)}")
        else:
            print(f"[{name}] OK")
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if not check():
        print("Датасети ще не додано — див. джерела в описі модуля.")
