import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from preprocess import load_mitbih_record, bandpass_ecg, time_axis


def main():
    parser = argparse.ArgumentParser(description="ECG bandpass filter demo (MIT-BIH)")
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to MIT-BIH files (e.g., data/mitbih)')
    parser.add_argument('--record', type=str, default='100', help='Record id, e.g., 100')
    parser.add_argument('--seconds', type=float, default=10.0, help='Seconds to visualize')
    parser.add_argument('--start', type=float, default=0.0, help='Start time (s)')
    parser.add_argument('--notch', type=int, default=50, choices=[0, 50, 60], help='Mains notch (0 to disable)')
    args = parser.parse_args()

    x, fs = load_mitbih_record(args.record, args.data_dir)
    start = int(args.start * fs)
    stop = int((args.start + args.seconds) * fs)
    x = x[start:stop]
    notch = None if args.notch == 0 else args.notch
    y = bandpass_ecg(x, fs, low=0.5, high=40.0, order=4, notch=notch)

    t = time_axis(len(x), fs)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title(f"Raw ECG — record {args.record}")
    plt.xlabel("Time (s)")
    plt.ylabel("mV")

    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.title("Filtered ECG (0.5–40 Hz bandpass" + (f", notch {args.notch} Hz" if notch else "") + ")")
    plt.xlabel("Time (s)")
    plt.ylabel("mV")
    plt.tight_layout()
    plt.show()
    f_raw, Pxx_raw = welch(x, fs=fs, nperseg=min(1024, len(x)))
    f_flt, Pxx_flt = welch(y, fs=fs, nperseg=min(1024, len(y)))
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_raw, Pxx_raw, label='Raw')
    plt.semilogy(f_flt, Pxx_flt, label='Filtered')
    plt.xlim(0, 100)
    plt.title("PSD Before/After")
    plt.xlabel("Hz")
    plt.ylabel("Power/Hz")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()