from agents.reporter import _validate_select


def test_allows_select_from_registered_gold_table():
    assert _validate_select(
        "SELECT product, SUM(revenue_sum) FROM gold_1 GROUP BY product",
        {"gold_1"},
    ) is None


def test_rejects_unregistered_table():
    assert _validate_select("SELECT * FROM gold_2", {"gold_1"})


def test_rejects_file_reader_function():
    assert _validate_select("SELECT * FROM read_parquet('secrets.parquet')", {"gold_1"})


def test_rejects_multiple_statements():
    assert _validate_select("SELECT * FROM gold_1; SELECT * FROM gold_1", {"gold_1"})
