"""Smoke-тест середовища на іграшкових чеках (без реального датасету)."""

from egastro_ml.config import load_config
from egastro_ml.rules import mine_rules, recommend

BASKETS = [
    ["паста", "томатний соус", "пармезан"],
    ["паста", "томатний соус"],
    ["паста", "томатний соус", "базилік"],
    ["стейк", "соус барбекю", "спеції"],
    ["стейк", "спеції"],
    ["стейк", "соус барбекю"],
    ["молоко", "хліб"],
    ["молоко", "пластівці"],
    ["хліб", "масло"],
    ["паста", "пармезан"],
]


def test_config_loads():
    cfg = load_config()
    assert cfg["rules"]["min_lift"] == 1.2


def test_rules_and_recommendation():
    rules = mine_rules(BASKETS, min_support=0.2, min_confidence=0.5, min_lift=1.2)
    assert not rules.empty
    assert (rules["lift"] >= 1.2).all()
    assert recommend({"паста"}, rules)[0] == "томатний соус"
    assert "спеції" in recommend({"стейк"}, rules)
    assert recommend({"паста", "томатний соус"}, rules).count("томатний соус") == 0
