"""
Per-Beat Feature Extraction
============================
Extracts ~30 features per beat window as per the project plan:

RR-interval features (5):
  pre_rr, post_rr, local_avg_rr, pre_rr_ratio, post_rr_ratio

Morphological features (14):
  qrs_duration, qrs_amplitude, r_amplitude, t_wave_amplitude, st_level
  + 4 wavelet energy features (db4, levels 1-4)
  + 6 DCT shape coefficients

Frequency-domain HRV features (4):
  vlf_power, lf_power, hf_power, lf_hf_ratio
  (computed over a context window of surrounding RR intervals)

Total: ~23-25 features (some may be NaN-filled for edge beats).

Reference: de Chazal et al., IEEE TBME 2004.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.signal import welch

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# RR-Interval Features

def extract_rr_features(
    beat_idx: int,
    r_peaks: np.ndarray,
    fs: float,
    context: int = 10,
) -> dict[str, float]:
    """Compute per-beat RR features relative to surrounding beats.

    Parameters
    ----------
    beat_idx : index of the current beat in r_peaks array
    r_peaks  : array of all R-peak sample indices in the record
    fs       : sampling frequency (Hz)
    context  : number of surrounding beats for local average RR

    Returns dict with keys: pre_rr, post_rr, local_avg_rr,
                             pre_rr_ratio, post_rr_ratio
    """
    n = len(r_peaks)
    feats: dict[str, float] = {}

    # pre-RR: interval before this beat
    if beat_idx > 0:
        feats["pre_rr"] = (r_peaks[beat_idx] - r_peaks[beat_idx - 1]) / fs
    else:
        feats["pre_rr"] = float("nan")

    # post-RR: interval after this beat
    if beat_idx < n - 1:
        feats["post_rr"] = (r_peaks[beat_idx + 1] - r_peaks[beat_idx]) / fs
    else:
        feats["post_rr"] = float("nan")

    # local average RR: uses up to `context` surrounding beats on each side
    half = context // 2
    lo = max(0, beat_idx - half)
    hi = min(n - 1, beat_idx + half)
    local_rr = np.diff(r_peaks[lo : hi + 1]) / fs
    if len(local_rr) > 0:
        feats["local_avg_rr"] = float(np.mean(local_rr))
    else:
        feats["local_avg_rr"] = feats.get("pre_rr", float("nan"))

    avg = feats["local_avg_rr"]
    feats["pre_rr_ratio"] = feats["pre_rr"] / avg if (avg and not np.isnan(avg)) else float("nan")
    feats["post_rr_ratio"] = feats["post_rr"] / avg if (avg and not np.isnan(avg)) else float("nan")

    return feats


# Morphological Features

def extract_morphological_features(
    beat: np.ndarray,
    fs: float,
    cfg: dict | None = None,
) -> dict[str, float]:
    """Extract shape/morphology features from a single beat window.

    Parameters
    ----------
    beat : 1-D array of signal samples centred on R-peak
           (length = pre_samples + post_samples, e.g. 200 at 360 Hz)
    fs   : sampling frequency
    cfg  : optional config dict (keys: qrs_window, st_window, t_window,
                                       wavelet, wavelet_levels, dct_n_coeffs)

    Returns dict with morphological feature keys.
    """
    if cfg is None:
        cfg = {}

    qrs_win = cfg.get("qrs_window", [35, 65])
    st_win  = cfg.get("st_window",  [80, 110])
    t_win   = cfg.get("t_window",   [65, 110])
    wavelet = cfg.get("wavelet", "db4")
    levels  = cfg.get("wavelet_levels", 4)
    dct_n   = cfg.get("dct_n_coeffs", 6)

    feats: dict[str, float] = {}

    n = len(beat)
    # clip indices to valid range
    def clip(start, end):
        return max(0, min(start, n)), max(0, min(end, n))

    # QRS region
    qs, qe = clip(*qrs_win)
    qrs_segment = beat[qs:qe]
    feats["qrs_duration"] = len(qrs_segment) / fs * 1000        # ms
    feats["qrs_amplitude"] = float(np.ptp(qrs_segment)) if len(qrs_segment) else float("nan")

    # R-peak amplitude (sample at pre_samples index, default 90)
    r_idx = min(90, n - 1)
    feats["r_amplitude"] = float(beat[r_idx])

    # T-wave
    ts, te = clip(*t_win)
    t_segment = beat[ts:te]
    feats["t_wave_amplitude"] = float(np.max(np.abs(t_segment))) if len(t_segment) else float("nan")

    # ST level (mean amplitude in ST window)
    ss, se = clip(*st_win)
    st_segment = beat[ss:se]
    feats["st_level"] = float(np.mean(st_segment)) if len(st_segment) else float("nan")

    # Wavelet energy (db4, levels 1-4)
    if _HAS_PYWT:
        try:
            coeffs = pywt.wavedec(beat, wavelet, level=levels)
            # coeffs[0] = approx, coeffs[1..] = details (level 1 = finest)
            for lvl in range(1, levels + 1):
                detail = coeffs[lvl] if lvl < len(coeffs) else np.array([0.0])
                feats[f"wavelet_energy_l{lvl}"] = float(np.sum(detail ** 2))
        except Exception:
            for lvl in range(1, levels + 1):
                feats[f"wavelet_energy_l{lvl}"] = float("nan")
    else:
        for lvl in range(1, levels + 1):
            feats[f"wavelet_energy_l{lvl}"] = float("nan")

    # DCT shape descriptor (first dct_n coefficients)
    try:
        dct_coeffs = dct(beat.astype(float), norm="ortho")
        for i in range(dct_n):
            feats[f"dct_{i}"] = float(dct_coeffs[i]) if i < len(dct_coeffs) else 0.0
    except Exception:
        for i in range(dct_n):
            feats[f"dct_{i}"] = float("nan")

    return feats


# Frequency-Domain HRV Features

def extract_hrv_frequency_features(
    beat_idx: int,
    r_peaks: np.ndarray,
    fs: float,
    cfg: dict | None = None,
) -> dict[str, float]:
    """Compute Welch PSD on local RR-interval series for one beat.

    Uses the context window of surrounding beats to build an RR time series,
    then estimates power in VLF / LF / HF bands.

    Parameters
    ----------
    beat_idx : index of current beat in r_peaks
    r_peaks  : all R-peak indices for this record
    fs       : ECG sampling frequency

    Returns dict: vlf_power, lf_power, hf_power, lf_hf_ratio
    """
    if cfg is None:
        cfg = {}

    vlf_band = cfg.get("vlf_band", [0.0, 0.04])
    lf_band  = cfg.get("lf_band",  [0.04, 0.15])
    hf_band  = cfg.get("hf_band",  [0.15, 0.40])

    nan_result = {
        "vlf_power": float("nan"),
        "lf_power":  float("nan"),
        "hf_power":  float("nan"),
        "lf_hf_ratio": float("nan"),
    }

    # need at least a few beats around this one
    context = 30
    lo = max(0, beat_idx - context)
    hi = min(len(r_peaks), beat_idx + context)
    rr = np.diff(r_peaks[lo:hi]) / fs  # RR in seconds

    if len(rr) < 8:
        return nan_result

    # RR series is irregularly sampled → resample at 4 Hz for Welch
    try:
        rr_fs = 4.0
        t_rr   = np.cumsum(rr)
        t_rr  -= t_rr[0]
        t_uni  = np.arange(0, t_rr[-1], 1.0 / rr_fs)
        rr_uni = np.interp(t_uni, t_rr, rr)

        nperseg = min(len(rr_uni), 64)
        freqs, psd = welch(rr_uni, fs=rr_fs, nperseg=nperseg)

        def band_power(f_low, f_high):
            idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
            return float(np.trapz(psd[idx], freqs[idx])) if len(idx) > 0 else 0.0

        vlf = band_power(*vlf_band)
        lf  = band_power(*lf_band)
        hf  = band_power(*hf_band)
        ratio = lf / hf if hf > 1e-10 else float("nan")

        return {
            "vlf_power":   vlf,
            "lf_power":    lf,
            "hf_power":    hf,
            "lf_hf_ratio": ratio,
        }
    except Exception:
        return nan_result


# Combined Per-Beat Feature Extraction

def extract_beat_features(
    beat_idx: int,
    beat: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    feat_cfg: dict | None = None,
) -> dict[str, float]:
    """Extract all features for a single beat.

    Combines RR-interval, morphological, and frequency-domain features.

    Parameters
    ----------
    beat_idx : position of this beat in r_peaks
    beat     : signal segment centred on R-peak
    r_peaks  : all R-peak indices in the record
    fs       : sampling frequency
    feat_cfg : feature sub-dict from config (cfg['features'])

    Returns a single flat dict of all features.
    """
    if feat_cfg is None:
        feat_cfg = {}

    context = feat_cfg.get("rr_context", 10)

    feats: dict[str, float] = {}
    feats.update(extract_rr_features(beat_idx, r_peaks, fs, context=context))
    feats.update(extract_morphological_features(beat, fs, cfg=feat_cfg))
    feats.update(extract_hrv_frequency_features(beat_idx, r_peaks, fs, cfg=feat_cfg))

    return feats
