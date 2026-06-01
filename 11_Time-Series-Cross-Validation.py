import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate

logger = logging.getLogger(__name__)

# Extracted code from '11_Time-Series-Cross-Validation.md'
# Blocks appear in the same order as in the markdown article.



# ── Data ──────────────────────────────────────────────────────────────────────

data_path = Path("timeseries/2025-11-12_us_voter_turnout.csv")
df = pd.read_csv(data_path)
df["Year"] = pd.to_datetime(df["Year"], format="%Y")
df = df.sort_values("Year")
df = df[df["Turnout Rate"].notna()]
ts = df.set_index("Year")["Turnout Rate"]

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")

X = ts.index.year.values.reshape(-1, 1)
y = ts.values

# ── CV configuration (set n_splits in config.yaml) ───────────────────────────

n_splits = config.get("cv", {}).get("n_splits", 5)
gap = config.get("cv", {}).get("gap", 0)

model = RandomForestRegressor(n_estimators=50, random_state=42)

SCORING = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    ),
}


def _report(label: str, cv_results: dict) -> float:
    mae = -cv_results["test_mae"]
    rmse = -cv_results["test_rmse"]
    logger.info(f"=== {label} ({n_splits} folds) ===")
    logger.info(f"  MAE:  {mae.mean():.4f} ± {mae.std():.4f}")
    logger.info(f"  RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}")
    return float(mae.mean())


# ── 1. TimeSeriesSplit ────────────────────────────────────────────────────────

tscv = TimeSeriesSplit(n_splits=n_splits)
cv_tscv = cross_validate(model, X, y, cv=tscv, scoring=SCORING)
mae_tscv = _report("TimeSeriesSplit", cv_tscv)

# ── 2. Purged CV ──────────────────────────────────────────────────────────────


class _PurgedTimeSeriesSplit:
    """sklearn-compatible purged CV — drops `gap` samples at the train/test
    boundary to prevent temporal leakage. Use gap ≥ 1 when features have
    look-ahead (e.g. rolling means that peek into the test window)."""

    def __init__(self, n_splits: int = 5, gap: int = 1):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size - self.gap
            test_start = (i + 1) * fold_size
            test_end = min((i + 2) * fold_size, n)
            if train_end > 0 and test_start < n:
                yield np.arange(train_end), np.arange(test_start, test_end)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


purged_cv = _PurgedTimeSeriesSplit(n_splits=n_splits, gap=max(gap, 2))
cv_purged = cross_validate(model, X, y, cv=purged_cv, scoring=SCORING)
mae_purged = _report("Purged CV", cv_purged)

# ── 3. Blocked CV ─────────────────────────────────────────────────────────────


class _BlockedTimeSeriesSplit:
    """Contiguous non-overlapping blocks — each fold gets a fresh slice of the
    series so there is no expanding-window bias in the error estimate."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        block_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = (i + 1) * block_size
            test_start = train_end
            test_end = min((i + 2) * block_size, n)
            if train_end > 0 and test_start < n:
                yield np.arange(train_end), np.arange(test_start, test_end)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


blocked_cv = _BlockedTimeSeriesSplit(n_splits=n_splits)
cv_blocked = cross_validate(model, X, y, cv=blocked_cv, scoring=SCORING)
mae_blocked = _report("Blocked CV", cv_blocked)

# ── 4. Walk-Forward (expanding window) ───────────────────────────────────────


class _WalkForwardCV:
    """Expanding or rolling walk-forward CV — the most production-realistic
    strategy because it mirrors how a deployed model re-trains over time."""

    def __init__(
        self, initial_train_size: int = 50, test_size: int = 10, expanding: bool = True
    ):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.expanding = expanding

    def split(self, X, y=None, groups=None):
        n = len(X)
        t_start, t_end = 0, self.initial_train_size
        while t_end + self.test_size <= n:
            yield np.arange(t_start, t_end), np.arange(t_end, t_end + self.test_size)
            t_end += self.test_size
            if not self.expanding:
                t_start += self.test_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return sum(1 for _ in self.split(np.empty(len(X))))


wf_cv = _WalkForwardCV(initial_train_size=50, test_size=10, expanding=True)
cv_wf = cross_validate(model, X, y, cv=wf_cv, scoring=SCORING)
mae_wf = _report("Walk-Forward (expanding)", cv_wf)

# ── Comparison ────────────────────────────────────────────────────────────────

results = {
    "TimeSeriesSplit": mae_tscv,
    "Purged CV": mae_purged,
    "Blocked CV": mae_blocked,
    "Walk-Forward": mae_wf,
}

logger.info("=== CV METHOD COMPARISON ===")
for method, mae in results.items():
    logger.info(f"  {method:<25} MAE = {mae:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    results.keys(),
    results.values(),
    color=["#2b2b2b", "#d62728", "#2980b9", "#27ae60"],
    alpha=0.85,
)
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Cross-Validation Method Comparison")
for bar, v in zip(bars, results.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
signalplot.tidy_axes(ax)
plt.tight_layout()
signalplot.save("cv_comparison.png")
