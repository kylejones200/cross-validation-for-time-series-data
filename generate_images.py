#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""
import logging

logger = logging.getLogger(__name__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    tf = None
except Exception:
    tf = None

# Tufte-style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
original_savefig = plt.savefig

def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith('/') and not str(filename).startswith('images/'):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")

plt.savefig = savefig_tufte

# Code blocks from article

# Code block 1

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load voter turnout data
data_path = Path("../../timeseries/2025-11-12_us_voter_turnout.csv")
df = pd.read_csv(data_path)

# Clean and prepare
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df = df.sort_values('Year')
df = df[df['Turnout Rate'].notna()]

ts = df.set_index('Year')['Turnout Rate']

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")



# Code block 2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = ts.index.year.values.reshape(-1, 1)
y = ts.values

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

scores_tscv = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Simple model for demonstration
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    scores_tscv.append(mae)
    logger.info(f"Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, MAE={mae:.2f}")

logger.info(f"\nTimeSeriesSplit average MAE: {np.mean(scores_tscv):.2f}")



# Code block 3
def purged_cv(data, n_splits=5, purge_gap=2):
    """
    Purged cross-validation with gap between train and test.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    n_splits : int
        Number of CV folds
    purge_gap : int
        Number of periods to purge between train and test
    """
    n = len(data)
    fold_size = n // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        test_start = (i + 1) * fold_size
        test_end = min((i + 2) * fold_size, n)
        train_end = test_start - purge_gap
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits

# Apply purged CV
purged_splits = purged_cv(ts.values, n_splits=5, purge_gap=2)

scores_purged = []
for fold, (train_idx, test_idx) in enumerate(purged_splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    scores_purged.append(mae)
    logger.info(f"Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, MAE={mae:.2f}")

logger.info(f"\nPurged CV average MAE: {np.mean(scores_purged):.2f}")



# Code block 4
def blocked_cv(data, n_splits=5):
    """
    Blocked cross-validation with contiguous blocks.
    
    Prevents leakage by using non-overlapping blocks.
    """
    n = len(data)
    block_size = n // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        test_start = (i + 1) * block_size
        test_end = min((i + 2) * block_size, n)
        train_end = test_start
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits

# Apply blocked CV
blocked_splits = blocked_cv(ts.values, n_splits=5)

scores_blocked = []
for fold, (train_idx, test_idx) in enumerate(blocked_splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    scores_blocked.append(mae)
    logger.info(f"Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, MAE={mae:.2f}")

logger.info(f"\nBlocked CV average MAE: {np.mean(scores_blocked):.2f}")



# Code block 5
def walk_forward_validation(data, initial_train_size, test_size, expanding=True):
    """
    Walk-forward validation with expanding or rolling windows.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    initial_train_size : int
        Initial training set size
    test_size : int
        Size of each test set
    expanding : bool
        If True, use expanding window; if False, use rolling window
    """
    n = len(data)
    splits = []
    
    train_start = 0
    train_end = initial_train_size
    
    while train_end + test_size <= n:
        test_start = train_end
        test_end = test_start + test_size
        
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        
        splits.append((train_idx, test_idx))
        
        # Update for next fold
        if expanding:
            train_end += test_size  # Expanding window
        else:
            train_start += test_size  # Rolling window
            train_end += test_size
    
    return splits

# Expanding window (most realistic for production)
expanding_splits = walk_forward_validation(ts.values, initial_train_size=50, test_size=10, expanding=True)

scores_expanding = []
for fold, (train_idx, test_idx) in enumerate(expanding_splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    scores_expanding.append(mae)
    logger.info(f"Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, MAE={mae:.2f}")

logger.info(f"\nWalk-forward (expanding) average MAE: {np.mean(scores_expanding):.2f}")



# Code block 6
# Compile results
results = {
    'TimeSeriesSplit': np.mean(scores_tscv),
    'Purged CV': np.mean(scores_purged),
    'Blocked CV': np.mean(scores_blocked),
    'Walk-Forward': np.mean(scores_expanding)
}

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
methods = list(results.keys())
mae_values = list(results.values())

bars = ax.bar(methods, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
ax.set_ylabel('Mean Absolute Error', fontsize=11)
ax.set_title('Cross-Validation Method Comparison', fontsize=13, fontweight='bold')
# Add value labels
for bar, value in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

logger.info("=== CROSS-VALIDATION METHOD COMPARISON ===")
logger.info(f"{'Method':<20} {'Average MAE':<15}")
for method, mae in results.items():
    logger.info(f"{method:<20} {mae:<15.2f}")



# Code block 7
# Complete code for reproducibility
# See individual code blocks above for full implementation



logger.info("All images generated successfully!")
