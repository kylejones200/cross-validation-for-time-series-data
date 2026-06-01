use cross_validation_for_time_series_data_core::rolling_window_cv_bounds;

fn main() {
    for _ in 0..50_000 {
        let _ = rolling_window_cv_bounds(5000, 500, 10);
    }
}
