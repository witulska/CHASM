from experiments.common import aggregate_decisions


def test_aggregate_decisions_uses_nested_decision_field():
    rows = [
        {"sig_mean": {"decision": 1, "imf": 2.0}},
        {"sig_mean": {"decision": 0, "imf": 0.0}},
    ]
    assert aggregate_decisions(rows) == {"sig_mean": 0.5}
