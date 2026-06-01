//! Rolling-window CV split bounds: (train_start, train_end, test_start, test_end) per split.

pub fn rolling_window_cv_bounds(
    n_samples: usize,
    window_size: usize,
    step_size: usize,
) -> Vec<[usize; 4]> {
    if window_size == 0 || step_size == 0 || n_samples <= window_size {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start_idx = 0usize;
    while start_idx + window_size < n_samples {
        let end_idx = start_idx + window_size;
        if end_idx + step_size <= n_samples {
            out.push([start_idx, end_idx, end_idx, end_idx + step_size]);
            start_idx += step_size;
        } else {
            break;
        }
    }
    out
}
