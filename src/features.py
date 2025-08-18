import numpy as np

def extract_rr_features(r_idx: np.ndarray, fs: float) -> dict[str, float]:
    """Extract basic HRV features from R-peak indices."""
    rr_intervals = np.diff(r_idx) / fs  # in seconds
    if len(rr_intervals) < 2:
        return {}

    features = {
        "mean_rr": np.mean(rr_intervals),
        "std_rr": np.std(rr_intervals),
        "min_rr": np.min(rr_intervals),
        "max_rr": np.max(rr_intervals),
        "rmssd": np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
        "heart_rate": 60.0 / np.mean(rr_intervals),
    }
    return features


def extract_window_features(x: np.ndarray, r_idx: np.ndarray, fs: float, window: float = 10.0) -> list[dict[str, float]]:
    """Extract features per window of ECG."""
    n = len(x)
    step = int(window * fs)
    feats = []
    for start in range(0, n, step):
        end = min(start + step, n)
        r_in_window = r_idx[(r_idx >= start) & (r_idx < end)]
        if len(r_in_window) > 2:
            f = extract_rr_features(r_in_window, fs)
            f["window_start"] = start / fs
            feats.append(f)
    return feats

