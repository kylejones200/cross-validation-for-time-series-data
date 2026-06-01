"""Time series CV metrics via DuckDB (replaces sklearn metrics in hot path)."""

import duckdb
import polars as pl


def calculate_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
    pl.DataFrame({"actual": actual, "predicted": predicted})
    row = duckdb.sql("""
        SELECT
            AVG(POWER(actual - predicted, 2)) AS mse,
            AVG(ABS(actual - predicted)) AS mae,
            SQRT(AVG(POWER(actual - predicted, 2))) AS rmse
        FROM df
    """).pl().row(0, named=True)
    return {k: float(v) for k, v in row.items()}
