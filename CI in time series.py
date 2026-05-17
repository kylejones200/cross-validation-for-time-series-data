"""Generated from Jupyter notebook: CI in time series

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


def bootstrap_forecast_ci(
    model_order, data, steps=48, n_bootstraps=100, confidence=0.95
):
    forecasts = []
    for i in range(n_bootstraps):
        try:
            bootstrap_sample = data.sample(n=len(data), replace=True).sort_index()
            model = ARIMA(bootstrap_sample, order=model_order)
            fitted_model = model.fit()
            forecasts.append(fitted_model.forecast(steps=steps).values)
        except Exception as e:
            print(f"Bootstrap iteration {i} failed: {e}")
    if not forecasts:
        raise RuntimeError("All bootstrap iterations failed.")
    forecasts = np.array(forecasts)
    lower_ci = np.percentile(forecasts, (1 - confidence) / 2 * 100, axis=0)
    upper_ci = np.percentile(forecasts, (1 + confidence) / 2 * 100, axis=0)
    mean_forecast = np.mean(forecasts, axis=0)
    return (mean_forecast, lower_ci, upper_ci)


def forecast_with_confidence(data, order, steps=48, confidence=0.95):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    forecast_result = fitted_model.get_forecast(steps=steps)
    forecasts = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=1 - confidence)
    return (forecasts, conf_int.iloc[:, 0], conf_int.iloc[:, 1])


def inverse_transform_and_flatten(scaler, data):
    return scaler.inverse_transform(np.array(data).reshape(-1, 1)).flatten()


def load_and_preprocess_data(url):
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.resample("h").mean().asfreq("h")
    df["values"] = df["values"].interpolate()
    scaler = StandardScaler()
    df["scaled_values"] = scaler.fit_transform(df[["values"]])
    return (df, scaler)


def plot_forecast_with_ci(
    historical_data,
    test_data,
    forecasts,
    lower_ci,
    upper_ci,
    title="Forecast with Confidence Intervals",
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        historical_data.index,
        historical_data.values,
        label="Historical Data",
        color="blue",
    )
    plt.plot(test_data.index, test_data, label="Actual Test Data", color="green")
    forecast_index = test_data.index
    plt.plot(forecast_index, forecasts, "r-", label="Forecast")
    plt.fill_between(
        forecast_index, lower_ci, upper_ci, color="r", alpha=0.2, label="95% CI"
    )
    plt.axvline(
        x=test_data.index[0], color="black", linestyle="--", label="Test Data Start"
    )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()


def main() -> None:
    url = "https://raw.githubusercontent.com/kylejones200/time_series/refs/heads/main/ercot_load_data.csv"

    df, scaler = load_and_preprocess_data(url)

    train_data = df["scaled_values"].iloc[:-48]

    test_data = df["scaled_values"].iloc[-48:]

    auto_model = auto_arima(
        train_data, seasonal=False, trace=True, suppress_warnings=True, stepwise=True
    )

    best_order = auto_model.order

    print(f"Using ARIMA order: {best_order}")

    forecasts, lower_ci, upper_ci = forecast_with_confidence(
        train_data, best_order, steps=48
    )

    boot_forecasts, boot_lower_ci, boot_upper_ci = bootstrap_forecast_ci(
        best_order, train_data, steps=48, n_bootstraps=50
    )

    forecasts, lower_ci, upper_ci = map(
        lambda x: inverse_transform_and_flatten(scaler, x),
        [forecasts, lower_ci, upper_ci],
    )

    boot_forecasts, boot_lower_ci, boot_upper_ci = map(
        lambda x: inverse_transform_and_flatten(scaler, x),
        [boot_forecasts, boot_lower_ci, boot_upper_ci],
    )

    test_data_original = inverse_transform_and_flatten(scaler, test_data)

    test_data_original_series = pd.Series(test_data_original, index=test_data.index)

    plot_forecast_with_ci(
        df["values"],
        test_data_original_series,
        forecasts,
        lower_ci,
        upper_ci,
        title="ARIMA Forecast with Confidence Intervals",
    )

    plot_forecast_with_ci(
        df["values"],
        test_data_original_series,
        boot_forecasts,
        boot_lower_ci,
        boot_upper_ci,
        title="Bootstrapped Forecast with Confidence Intervals",
    )


if __name__ == "__main__":
    main()
