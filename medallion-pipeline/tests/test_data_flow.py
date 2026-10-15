import pandas as pd

from agents.bronze_agent import ingest_bronze
from agents.gold_agent import materialize_gold
from agents.silver_agent import cleanse_silver


def test_approved_rules_flow_from_csv_through_gold(tmp_path, monkeypatch):
    source_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "product": ["Widget", "Gadget", "Widget"],
            "quantity": [2, 1, 3],
            "revenue": [20.0, 10.0, 30.0],
        }
    ).to_csv(source_path, index=False)

    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", bronze_dir)
    monkeypatch.setattr("agents.silver_agent.SILVER_DIR", silver_dir)
    monkeypatch.setattr("agents.gold_agent.GOLD_DIR", gold_dir)

    bronze_rules_path = tmp_path / "bronze_rules.csv"
    pd.DataFrame(
        [
            {
                "source_file": str(source_path.resolve()),
                "source_table": "sales",
                "source_col": column,
                "target_col": column,
            }
            for column in ("product", "quantity", "revenue")
        ]
    ).to_csv(bronze_rules_path, index=False)
    bronze_paths = ingest_bronze([str(source_path)], str(bronze_rules_path), "run-12345678")
    assert len(bronze_paths) == 1
    assert list(pd.read_parquet(bronze_paths[0]).columns) == ["product", "quantity", "revenue"]

    silver_rules_path = tmp_path / "silver_rules.csv"
    pd.DataFrame(
        [
            {"source_table": "sales", "source_col": "product", "target_col": "product", "transformation_logic": "strip_text", "null_handling": "fill_mode"},
            {"source_table": "sales", "source_col": "quantity", "target_col": "quantity", "transformation_logic": "cast_integer", "null_handling": "fill_median"},
            {"source_table": "sales", "source_col": "revenue", "target_col": "revenue", "transformation_logic": "cast_numeric", "null_handling": "fill_mean"},
        ]
    ).to_csv(silver_rules_path, index=False)
    silver_paths = cleanse_silver(bronze_paths, str(silver_rules_path), "run-12345678")
    silver = pd.read_parquet(silver_paths[0])
    assert silver.columns[0] == "pk_sales_silver_id"
    assert len(silver) == 3

    gold_rules_path = tmp_path / "gold_rules.csv"
    pd.DataFrame(
        [
            {
                "source_table": "sales",
                "source_col": "product",
                "target_table": "sales_summary",
                "target_col": "product",
                "aggregation": "",
                "group_by": True,
            },
            {
                "source_table": "sales",
                "source_col": "revenue",
                "target_table": "sales_summary",
                "target_col": "revenue_sum",
                "aggregation": "sum",
                "group_by": False,
            },
            {
                "source_table": "sales",
                "source_col": "",
                "target_table": "sales_summary",
                "target_col": "record_count",
                "aggregation": "count",
                "group_by": False,
            },
        ]
    ).to_csv(gold_rules_path, index=False)
    gold_paths = materialize_gold(silver_paths, str(gold_rules_path), "run-12345678")
    gold = pd.read_parquet(gold_paths[0]).set_index("product")

    assert gold.loc["Widget", "revenue_sum"] == 50.0
    assert gold.loc["Gadget", "revenue_sum"] == 10.0
    assert gold.loc["Widget", "record_count"] == 2
    assert gold.loc["Gadget", "record_count"] == 1


def test_gold_outer_joins_silver_tables_on_approved_key(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold"
    monkeypatch.setattr("agents.gold_agent.GOLD_DIR", gold_dir)
    orders_path = tmp_path / "orders__silver__run12345.parquet"
    products_path = tmp_path / "products__silver__run12345.parquet"
    pd.DataFrame({"product_id": [10, 20], "amount": [100.0, 200.0]}).to_parquet(orders_path, index=False)
    pd.DataFrame({"product_id": [10, 30], "product_name": ["Widget", "Toaster"]}).to_parquet(products_path, index=False)

    mappings_path = tmp_path / "gold_rules.csv"
    pd.DataFrame(
        [
            {"source_table": "orders", "source_col": "product_id", "target_table": "gold_analysis", "target_col": "product_id", "join_key": True, "group_by": False, "aggregation": ""},
            {"source_table": "orders", "source_col": "amount", "target_table": "gold_analysis", "target_col": "orders_amount_sum", "join_key": False, "group_by": False, "aggregation": "sum"},
            {"source_table": "products", "source_col": "product_id", "target_table": "gold_analysis", "target_col": "product_id", "join_key": True, "group_by": False, "aggregation": ""},
            {"source_table": "products", "source_col": "product_name", "target_table": "gold_analysis", "target_col": "product_name", "join_key": False, "group_by": True, "aggregation": ""},
        ]
    ).to_csv(mappings_path, index=False)

    output_paths = materialize_gold(
        [str(orders_path), str(products_path)],
        str(mappings_path),
        "run-12345678",
    )
    result = pd.read_parquet(output_paths[0])

    assert len(result) == 3
    by_name = result.set_index("product_name")["orders_amount_sum"]
    assert by_name["Widget"] == 100.0
    assert by_name["Toaster"] == 0.0
