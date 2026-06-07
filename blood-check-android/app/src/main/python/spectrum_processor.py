import os
import re

import numpy as np
import pywt
from scipy import signal, stats
from scipy.interpolate import interp1d


class CompleteSpectrumProcessor:
    """完整的光谱数据处理类 - 与训练代码完全一致"""

    def __init__(self):
        self.data_quality_stats = {}

    def extract_id_from_filename(self, filename):
        filename_no_ext = os.path.splitext(filename)[0]
        numbers = re.findall(r"\d+", filename_no_ext)
        if numbers:
            return max(numbers, key=len).lstrip("0")
        return filename_no_ext

    def normalize_spectrum_signal(self, signal_data):
        if len(signal_data) == 0:
            return np.array([5.0])

        signal_data = np.array(signal_data).astype(float)
        if len(signal_data) > 10:
            z_scores = np.abs(stats.zscore(signal_data, nan_policy="omit"))
            if len(z_scores) == len(signal_data):
                signal_clean = signal_data[z_scores < 3]
                if len(signal_clean) > 0:
                    signal_data = signal_clean

        if len(signal_data) == 0:
            return np.array([5.0])

        signal_min = np.min(signal_data)
        signal_max = np.max(signal_data)
        if signal_max - signal_min < 1e-10:
            return np.ones_like(signal_data) * 5.0

        normalized = (signal_data - signal_min) / (signal_max - signal_min) * 10.0
        return np.clip(normalized, 0, 12)

    def robust_ratio_calculation(self, red_signal, ir_signal):
        red_signal = np.array(red_signal, dtype=float)
        ir_signal = np.array(ir_signal, dtype=float)

        min_length = min(len(red_signal), len(ir_signal))
        if min_length == 0:
            return np.ones(1) * 1.2

        red_signal = red_signal[:min_length]
        ir_signal = ir_signal[:min_length]
        mask = ~(
            np.isnan(red_signal)
            | np.isnan(ir_signal)
            | np.isinf(red_signal)
            | np.isinf(ir_signal)
            | (ir_signal == 0)
        )
        if np.sum(mask) < 10:
            return np.ones_like(red_signal) * 1.2

        ratio = red_signal[mask] / (ir_signal[mask] + 1e-10)
        result = np.ones_like(red_signal) * 1.2
        if len(mask) == len(result):
            temp_result = np.ones_like(red_signal) * 1.2
            if np.sum(mask) == len(ratio):
                temp_result[mask] = ratio
            else:
                temp_result[mask] = (
                    ratio[: np.sum(mask)]
                    if len(ratio) >= np.sum(mask)
                    else np.concatenate([ratio, np.ones(np.sum(mask) - len(ratio)) * 1.2])
                )
            result = temp_result
        return result

    def validate_and_align_signals(self, red_signal, ir_signal, filename=""):
        if len(red_signal) != len(ir_signal):
            min_len = min(len(red_signal), len(ir_signal))
            if min_len >= 10:
                return red_signal[:min_len], ir_signal[:min_len]
            return None, None
        return red_signal, ir_signal

    def calculate_signal_quality_score(self, red_signal, ir_signal):
        try:
            red_snr = np.mean(red_signal) / (np.std(red_signal) + 1e-10)
            ir_snr = np.mean(ir_signal) / (np.std(ir_signal) + 1e-10)
            snr = (red_snr + ir_snr) / 2
            snr_score = max(0, min(10, (snr - 1) / 6 * 10)) if snr > 1 else 0

            red_peaks, _ = signal.find_peaks(red_signal, distance=20, prominence=0.1 * np.std(red_signal))
            ir_peaks, _ = signal.find_peaks(ir_signal, distance=20, prominence=0.1 * np.std(ir_signal))
            pulse_count = min(len(red_peaks), len(ir_peaks))
            pulse_score = max(0, min(10, (pulse_count - 3) / 12 * 10)) if pulse_count >= 3 else 0

            signal_length = len(red_signal)
            length_score = max(0, min(10, (signal_length - 100) / 900 * 10)) if signal_length >= 100 else 0
            total_score = snr_score * 0.4 + pulse_score * 0.3 + length_score * 0.3
            return total_score, snr_score, pulse_score, length_score, pulse_count
        except Exception:
            return 0, 0, 0, 0, 0

    def repair_low_quality_signal(self, signal_data, score, pulse_count, signal_length):
        repaired_signal = signal_data.copy()
        try:
            if signal_length < 500 and len(signal_data) > 10:
                interpolator = interp1d(
                    np.linspace(0, 1, len(signal_data)),
                    signal_data,
                    kind="linear",
                    fill_value="extrapolate",
                )
                repaired_signal = interpolator(np.linspace(0, 1, 500))
                signal_length = 500

            if pulse_count < 5 and len(repaired_signal) > 20:
                peaks, _ = signal.find_peaks(repaired_signal, distance=15, prominence=0.05 * np.std(repaired_signal))
                if len(peaks) > pulse_count:
                    pulse_count = len(peaks)

            if score < 3 and len(repaired_signal) >= 64:
                try:
                    coeffs = pywt.wavedec(repaired_signal, "db4", level=3)
                    threshold = 0.1 * np.std(repaired_signal)
                    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
                    repaired_signal = pywt.waverec(coeffs_thresh, "db4")
                except Exception:
                    window_size = min(5, len(repaired_signal) // 10)
                    if window_size >= 3:
                        repaired_signal = np.convolve(repaired_signal, np.ones(window_size) / window_size, mode="same")

            return repaired_signal, pulse_count, signal_length
        except Exception:
            return signal_data, pulse_count, len(signal_data)

    def extract_ac_dc_adaptive(self, signal_data, heart_rate=None):
        window_size = min(100, len(signal_data) // 10)
        if window_size < 3:
            window_size = 3

        if heart_rate is not None and (heart_rate > 100 or heart_rate < 50):
            if heart_rate > 100:
                window_size = min(50, len(signal_data) // 8)
            elif heart_rate < 50:
                window_size = min(150, len(signal_data) // 4)
            window_size = max(window_size, 3)

        dc = np.convolve(signal_data, np.ones(window_size) / window_size, mode="same")
        ac = signal_data - dc
        if heart_rate is not None and (heart_rate > 100 or heart_rate < 50) and len(ac) >= 5:
            ac = np.convolve(ac, np.ones(5) / 5, mode="same")

        return np.mean(np.abs(ac)) / (np.mean(np.abs(dc)) + 1e-10)

    def extract_frequency_features(self, signal_data, fs=100):
        if len(signal_data) < 64:
            return {"pulse_power_ratio": 0.1, "resp_power_ratio": 0.05, "dominant_freq": 1.0}

        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))
        total_power = np.sum(psd)
        pulse_band = (freqs >= 0.5) & (freqs <= 4)
        resp_band = (freqs >= 0.1) & (freqs <= 0.5)
        dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0

        return {
            "pulse_power_ratio": np.sum(psd[pulse_band]) / (total_power + 1e-10),
            "resp_power_ratio": np.sum(psd[resp_band]) / (total_power + 1e-10),
            "dominant_freq": dominant_freq,
        }

    def extract_pulse_features(self, signal_data, quality_score):
        if quality_score < 3:
            prominence_factor = 0.05
        elif quality_score < 6:
            prominence_factor = 0.08
        else:
            prominence_factor = 0.1

        peaks, _ = signal.find_peaks(signal_data, distance=20, prominence=prominence_factor * np.std(signal_data))
        if len(peaks) >= 2:
            intervals = np.diff(peaks)
            amplitudes = signal_data[peaks]
            return {
                "pulse_count": len(peaks),
                "interval_mean": np.mean(intervals),
                "interval_std": np.std(intervals),
                "amp_mean": np.mean(amplitudes),
                "amp_std": np.std(amplitudes),
                "peak_variability": np.std(amplitudes) / (np.mean(amplitudes) + 1e-10),
            }

        return {
            "pulse_count": len(peaks),
            "interval_mean": 0,
            "interval_std": 0,
            "amp_mean": 0,
            "amp_std": 0,
            "peak_variability": 0,
        }

    def process_spectrum_signals(self, red_signal_original, ir_signal_original, filename=""):
        features = {}
        red_signal_original, ir_signal_original = self.validate_and_align_signals(red_signal_original, ir_signal_original, filename)
        if red_signal_original is None or ir_signal_original is None:
            return self.get_default_spectrum_features(self.extract_id_from_filename(filename), filename)

        red_signal_original = red_signal_original[~np.isnan(red_signal_original)]
        ir_signal_original = ir_signal_original[~np.isnan(ir_signal_original)]
        red_signal_original, ir_signal_original = self.validate_and_align_signals(
            red_signal_original,
            ir_signal_original,
            f"{filename} (去除NaN后)",
        )
        if red_signal_original is None or ir_signal_original is None:
            return self.get_default_spectrum_features(self.extract_id_from_filename(filename), filename)
        if len(red_signal_original) < 10 or len(ir_signal_original) < 10:
            return self.get_default_spectrum_features(self.extract_id_from_filename(filename), filename)

        red_signal = self.normalize_spectrum_signal(red_signal_original)
        ir_signal = self.normalize_spectrum_signal(ir_signal_original)

        quality_score, snr_score, pulse_score, length_score, pulse_count = self.calculate_signal_quality_score(red_signal, ir_signal)
        features["信号质量评分"] = quality_score
        features["信噪比评分"] = snr_score
        features["脉搏数评分"] = pulse_score
        features["信号长度评分"] = length_score

        if quality_score < 3:
            red_signal, _, _ = self.repair_low_quality_signal(red_signal, quality_score, pulse_count, len(red_signal))
            ir_signal, _, _ = self.repair_low_quality_signal(ir_signal, quality_score, pulse_count, len(ir_signal))
            features["信号质量评分_修复后"], _, _, _, _ = self.calculate_signal_quality_score(red_signal, ir_signal)
            features["信号修复标记"] = 1
        elif quality_score < 6:
            window_size = min(5, len(red_signal) // 10)
            if window_size >= 3:
                red_signal = np.convolve(red_signal, np.ones(window_size) / window_size, mode="same")
                ir_signal = np.convolve(ir_signal, np.ones(window_size) / window_size, mode="same")
            features["信号修复标记"] = 0
        else:
            features["信号修复标记"] = 0

        ratio = self.robust_ratio_calculation(red_signal, ir_signal)
        ratio[ratio <= 0] = 0.1
        ratio = np.clip(ratio, 0.3, 3.0)
        features["R值_mean"] = np.mean(ratio)
        features["R值_std"] = np.std(ratio)
        features["R值_min"] = np.min(ratio)
        features["R值_max"] = np.max(ratio)
        features["R值异常标记"] = 1 if (features["R值_mean"] < 0.5 or features["R值_mean"] > 2.0) else 0
        features["R值极端异常标记"] = 1 if (features["R值_mean"] < 0.3 or features["R值_mean"] > 3.0) else 0

        features["红光_mean"] = np.mean(red_signal)
        features["红光_std"] = np.std(red_signal)
        features["红外光_mean"] = np.mean(ir_signal)
        features["红外光_std"] = np.std(ir_signal)

        red_ac_dc_ratio = self.extract_ac_dc_adaptive(red_signal)
        ir_ac_dc_ratio = self.extract_ac_dc_adaptive(ir_signal)
        features["红光_AC_DC_ratio"] = red_ac_dc_ratio
        features["红外光_AC_DC_ratio"] = ir_ac_dc_ratio
        features["AC_DC_ratio_diff"] = red_ac_dc_ratio - ir_ac_dc_ratio

        pi_estimate = (red_ac_dc_ratio + ir_ac_dc_ratio) / 2 * 100
        features["PI_估算"] = pi_estimate

        for key, value in self.extract_frequency_features(red_signal).items():
            features[f"红光_{key}"] = value
        for key, value in self.extract_frequency_features(ir_signal).items():
            features[f"红外光_{key}"] = value
        for key, value in self.extract_pulse_features(red_signal, quality_score).items():
            features[f"红光_{key}"] = value
        for key, value in self.extract_pulse_features(ir_signal, quality_score).items():
            features[f"红外光_{key}"] = value

        features["信号信噪比"] = np.mean(red_signal) / (np.std(red_signal) + 1e-10)
        features["信号长度"] = len(red_signal)
        features["信号质量_红光"] = 1.0 if features["红光_mean"] > 0 else 0.0
        features["信号质量_红外光"] = 1.0 if features["红外光_mean"] > 0 else 0.0
        features["PI_ac_dc_ratio"] = pi_estimate / 100
        features["PI_variability"] = np.std(red_signal) / (np.std(ir_signal) + 1e-10)
        features["统一ID"] = self.extract_id_from_filename(filename)
        features["源文件名"] = filename
        return features

    def get_default_spectrum_features(self, patient_id, filename):
        return {
            "统一ID": patient_id,
            "源文件名": filename,
            "信号质量评分": 0,
            "R值_mean": 1.2,
            "红外光_mean": 5.0,
            "红光_mean": 5.0,
            "R值异常标记": 1,
            "PI_估算": 0.5,
            "红光_std": 1.0,
            "红外光_std": 1.0,
            "R值_std": 0.1,
            "R值_min": 0.9,
            "R值_max": 1.5,
            "红光_AC_DC_ratio": 0.02,
            "红外光_AC_DC_ratio": 0.015,
            "AC_DC_ratio_diff": 0.005,
            "红光_pulse_power_ratio": 0.3,
            "红外光_pulse_power_ratio": 0.28,
            "红光_resp_power_ratio": 0.05,
            "红外光_resp_power_ratio": 0.05,
            "红光_dominant_freq": 1.2,
            "红外光_dominant_freq": 1.2,
            "红光_pulse_count": 10,
            "红外光_pulse_count": 10,
            "信号信噪比": 5.0,
            "信号长度": 1000,
        }
