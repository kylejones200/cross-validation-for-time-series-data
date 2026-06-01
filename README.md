# Cross-Validation for Time Series Data

This project demonstrates various cross-validation strategies for time series data, including time series splits, rolling windows, nested CV, and blocking methods.

## Business context

Time series cross-validation differs fundamentally from standard cross-validation techniques because it must respect temporal ordering. Simple random splitting can lead to data leakage and over-optimistic performance estimates.

The `TimeSeriesSplit` method is a common way to perform cross-validation in time series. It ensures that the training data always precedes the validation data.

Rolling window cross-validation trains the model on a fixed-size window of data, which shifts forward in time for each iteration.

## Article

Medium article: [Cross-Validation for Time Series Data](https://medium.com/@kylejones_47003/cross-validation-for-time-series-data-51fd11c38e2b)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Cross-validation classes
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and column names
- Cross-validation parameters (n_splits, window_size, block_size)
- Which CV methods to run
- Analysis options

## Available CV Methods

- **TimeSeriesCV**: Standard time series cross-validation splits
- **RollingWindowCV**: Rolling window approach
- **NestedTimeSeriesCV**: Nested CV for hyperparameter tuning
- **BlockingTimeSeriesCV**: Block-based CV
- **TimeSeriesEvaluation**: Custom metrics evaluation

## Caveats

- Time series CV preserves temporal ordering (no shuffling).
- Data leakage checks verify that training data comes before test data.
- Rolling window CV can be computationally expensive for large datasets.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).