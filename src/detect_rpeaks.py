from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from wfdb import processing
from pathlib import Path
from preprocess import bandpass_ecg, load_mitbih_record, time_axis


def detect_rpeaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks in ECG using wfdb's XQRS detector.
    Returns indices of detected peaks.
    """
    x = np.asarray(x, float)
    xqrs = processing.XQRS(sig=x, fs=fs)
    xqrs.detect()
    return np.array(xqrs.qrs_inds, dtype=int)


def demo_record(record_id: str | int, data_dir: str | Path,
                seconds: int = 10, start: int = 0, notch: int | None = None):
    """Demo: load a record, filter, detect R-peaks, and plot."""
    x, fs = load_mitbih_record(record_id, data_dir)

    # crop
    s0 = int(start * fs)
    s1 = int((start + seconds) * fs)
    x = x[s0:s1]

    # filter
    xf = bandpass_ecg(x, fs, notch=notch)

    # detect R-peaks
    rpeaks = detect_rpeaks(xf, fs)

    # plot
    t = time_axis(len(xf), fs)
    plt.figure(figsize=(12, 4))
    plt.plot(t, xf, label="Filtered ECG")
    plt.scatter(t[rpeaks], xf[rpeaks], color="red", marker="x", label="R-peaks")
    plt.title(f"Record {record_id} R-peak detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()

    return xf, rpeaks, fs



