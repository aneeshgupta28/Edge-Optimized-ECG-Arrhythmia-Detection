from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import wfdb
from pathlib import Path


def bandpass_ecg(x: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0,
                 order: int = 4, notch: int | None = None) -> np.ndarray:
    """Zero-phase ECG bandpass with optional mains notch.
    - low/high in Hz, order is Butterworth order (effective 2*order due to filtfilt)
    - notch: 50 or 60 to remove mains hum (optional)
    """
    x = np.asarray(x, dtype=float)

    if notch in (50, 60):
        w0 = notch / (fs / 2.0)
        Q = 30.0
        b_notch, a_notch = iirnotch(w0, Q)
        x = filtfilt(b_notch, a_notch, x)

    nyq = 0.5 * fs
    b, a = butter(N=order, Wn=[low / nyq, high / nyq], btype='band')
    y = filtfilt(b, a, x)
    return y


def load_mitbih_record(record_id: str | int, data_dir: str | Path) -> tuple[np.ndarray, float]:
    """Load a single MIT-BIH record using wfdb. Returns (signal, fs).
    - record_id: e.g., '100'
    - data_dir: folder containing .dat/.hea/.atr files
    """
    data_dir = Path(data_dir)
    rec = wfdb.rdrecord(str(data_dir / str(record_id)))
    fs = float(rec.fs)
    # choose first channel; many records are 2-lead ECG
    x = rec.p_signal[:, 0].astype(float)
    return x, fs


def time_axis(n: int, fs: float) -> np.ndarray:
    return np.arange(n) / fs