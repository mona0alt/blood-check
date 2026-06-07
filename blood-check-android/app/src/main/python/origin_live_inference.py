import json
import math
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt, hilbert, correlate, welch
from scipy.stats import skew, kurtosis, zscore


_service = None


class StandardScalerLite:
    def __init__(self, data):
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)

    def transform(self, values):
        array = values.values if hasattr(values, "values") else np.asarray(values)
        return (np.asarray(array, dtype=np.float64) - self.mean) / self.scale


class JsonLabelEncoder:
    def __init__(self, classes):
        self.mapping = {str(value): index for index, value in enumerate(classes)}

    def transform(self, values):
        return np.asarray([self.mapping.get(str(value), 0) for value in values], dtype=np.int64)


class XGBoostJsonRegressor:
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as model_file:
            model = json.load(model_file)
        learner = model["learner"]
        booster_model = learner["gradient_booster"]["model"]
        self.base_score = np.float32(float(learner["learner_model_param"]["base_score"].strip("[]"))).item()
        self.trees = booster_model["trees"]

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        return np.asarray([self.base_score + sum(self._predict_tree(tree, row) for tree in self.trees) for row in array])

    def _predict_tree(self, tree, row):
        node = 0
        while tree["left_children"][node] != -1:
            feature_index = tree["split_indices"][node]
            value = np.float32(row[feature_index]).item()
            if math.isnan(value):
                node = tree["left_children"][node] if int(tree["default_left"][node]) == 1 else tree["right_children"][node]
            elif int(tree["split_type"][node]) != 0:
                raise ValueError("Categorical XGBoost splits are not supported")
            else:
                node = tree["left_children"][node] if value < np.float32(tree["split_conditions"][node]).item() else tree["right_children"][node]
        return np.float32(tree["base_weights"][node]).item()


class TreeJsonRegressor:
    def __init__(self, tree):
        self.tree = tree

    def predict_one(self, row):
        node = 0
        while self.tree["children_left"][node] != -1:
            feature = self.tree["feature"][node]
            threshold = self.tree["threshold"][node]
            node = self.tree["children_left"][node] if float(row[feature]) <= threshold else self.tree["children_right"][node]
        return self.tree["value"][node]


class GradientBoostingJsonRegressor:
    def __init__(self, model_data):
        self.learning_rate = float(model_data["learning_rate"])
        self.init = float(model_data["init"])
        self.trees = [TreeJsonRegressor(tree) for tree in model_data["estimators"]]

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        outputs = []
        for row in np.asarray(array, dtype=np.float64):
            value = self.init
            for tree in self.trees:
                value += self.learning_rate * tree.predict_one(row)
            outputs.append(value)
        return np.asarray(outputs)


class RidgeJsonRegressor:
    def __init__(self, model_data):
        self.coef = np.asarray(model_data["coef"], dtype=np.float64)
        self.intercept = float(model_data["intercept"])

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        return np.dot(np.asarray(array, dtype=np.float64), self.coef) + self.intercept


class LightGBMJsonRegressor:
    def __init__(self, model_data):
        self.trees = model_data["dump"]["tree_info"]

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        outputs = []
        for row in np.asarray(array, dtype=np.float64):
            outputs.append(sum(self._predict_tree(tree["tree_structure"], row) for tree in self.trees))
        return np.asarray(outputs)

    def _predict_tree(self, node, row):
        while "leaf_value" not in node:
            feature = node["split_feature"]
            value = float(row[feature])
            threshold = float(node["threshold"])
            if math.isnan(value):
                go_left = bool(node.get("default_left", True))
            else:
                decision = node.get("decision_type", "<=")
                go_left = value <= threshold if decision == "<=" else value < threshold
            node = node["left_child"] if go_left else node["right_child"]
        return float(node["leaf_value"])


class IsolationForestJson:
    def __init__(self, data):
        self.max_samples = int(data["max_samples"])
        self.offset = float(data["offset"])
        self.estimators = data["estimators"]
        self.average_max_samples_path = self._average_path_length(self.max_samples)

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        outputs = []
        for row in np.asarray(array, dtype=np.float64):
            score_samples = -self._anomaly_score(row)
            outputs.append(1 if score_samples - self.offset >= 0 else -1)
        return np.asarray(outputs)

    def _anomaly_score(self, row):
        depths = []
        for estimator in self.estimators:
            features = estimator["features"]
            selected = row[features]
            depths.append(self._path_length(estimator["tree"], estimator["n_node_samples"], selected))
        mean_depth = np.mean(depths)
        return 2 ** (-mean_depth / self.average_max_samples_path) if self.average_max_samples_path > 0 else 1.0

    def _path_length(self, tree, n_node_samples, row):
        node = 0
        depth = 0
        while tree["children_left"][node] != -1:
            feature = tree["feature"][node]
            threshold = tree["threshold"][node]
            node = tree["children_left"][node] if float(row[feature]) <= threshold else tree["children_right"][node]
            depth += 1
        return depth + self._average_path_length(n_node_samples[node])

    def _average_path_length(self, n_samples_leaf):
        n = float(n_samples_leaf)
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        return 2.0 * (math.log(n - 1.0) + 0.5772156649015329) - (2.0 * (n - 1.0) / n)


class BasePhysioPredictor:
    HR_RANGE = (40, 200)
    SPO2_RANGE = (70, 100)
    R_RANGE = (0.3, 1.2)

    def __init__(self, data):
        self.feature_names = data["feature_names"]
        self.models = {
            key: LightGBMJsonRegressor(value)
            for key, value in data["models"].items()
        }
        self.outlier_detector = IsolationForestJson(data["outlier_detector"]) if data.get("outlier_detector") else None

    def _bandpass_filter(self, signal, fs=100, low=0.8, high=3.0, order=4):
        if len(signal) < 15:
            return signal
        try:
            b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], "band")
            return filtfilt(b, a, signal)
        except Exception:
            return signal

    def _extract_acdc(self, signal):
        signal = np.asarray(signal, dtype=float)
        signal = signal[~np.isnan(signal)]
        if len(signal) < 20:
            return np.nan
        win = max(5, len(signal) // 5)
        if win % 2 == 0:
            win += 1
        try:
            detrended = signal - savgol_filter(signal, win, 2)
        except Exception:
            detrended = signal - np.mean(signal)
        dc = np.mean(signal)
        if dc <= 0:
            return np.nan
        ac = (np.max(detrended) - np.min(detrended)) / 2 if not np.isnan(detrended).any() else 1.5 * np.std(detrended)
        return ac / dc

    def _compute_single_channel_features(self, signal, fs=100):
        sig = np.asarray(signal, dtype=float)
        sig = sig[~np.isnan(sig)]
        if len(sig) < 2 * fs:
            return None
        feat = {
            "mean": np.mean(sig),
            "std": np.std(sig),
            "skew": skew(sig) if len(sig) > 3 else 0.0,
            "kurt": kurtosis(sig) if len(sig) > 4 else 0.0,
            "iqr": np.percentile(sig, 75) - np.percentile(sig, 25),
            "min": np.min(sig),
            "max": np.max(sig),
            "ac_dc": self._extract_acdc(sig),
            "rms": np.sqrt(np.mean(sig ** 2)),
            "zero_crossing_rate": np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0) / len(sig),
            "wave_length": np.sum(np.abs(np.diff(sig))) / len(sig),
        }
        sig_filt = self._bandpass_filter(sig, fs)
        smooth = savgol_filter(sig_filt, 11, 2)
        prominence = max(0.1 * np.std(smooth), 0.05)
        peaks, _ = find_peaks(smooth, distance=int(fs * 0.3), prominence=prominence, width=(5, 60))
        feat["peak_count"] = len(peaks)
        if len(peaks) >= 2:
            ibis = np.diff(peaks) / fs
            valid_mask = (ibis >= 0.33) & (ibis <= 1.5)
            if len(ibis) >= 2:
                valid_mask[1:] &= (np.abs(np.diff(ibis)) / ibis[:-1] < 0.3)
            valid_ibis = ibis[valid_mask]
            if len(valid_ibis) >= 2:
                feat["mean_ibi"] = np.mean(valid_ibis)
                feat["ibi_std"] = np.std(valid_ibis)
                feat["hr_peak"] = 60.0 / feat["mean_ibi"]
                peak_vals = smooth[peaks[: len(valid_ibis) + 1]] if len(peaks) >= len(valid_ibis) + 1 else smooth[peaks]
                feat["peak_amp_cv"] = np.std(peak_vals) / np.mean(peak_vals) if np.mean(peak_vals) > 0 else 0
            else:
                feat.update({"mean_ibi": 0.0, "ibi_std": 0.0, "hr_peak": 0.0, "peak_amp_cv": 0.0})
        else:
            feat.update({"mean_ibi": 0.0, "ibi_std": 0.0, "hr_peak": 0.0, "peak_amp_cv": 0.0})

        n = len(sig_filt)
        freq = np.fft.rfftfreq(n, 1 / fs)
        mag = np.abs(np.fft.rfft(sig_filt))
        band = (freq >= 0.8) & (freq <= 3.0)
        if np.any(band):
            mag_band = mag[band]
            total = np.sum(mag_band)
            if total > 1e-9:
                peak_idx = np.argmax(mag_band)
                feat["fft_hr"] = freq[band][peak_idx] * 60
                feat["fft_power_ratio"] = mag_band[peak_idx] / total
                sorted_mag = np.sort(mag_band)
                feat["fft_promin"] = sorted_mag[-1] / sorted_mag[-2] if len(sorted_mag) > 1 else 10.0
                psd = mag_band / total
                feat["psd_entropy"] = -np.sum(psd * np.log2(psd + 1e-12))
                half_max = mag_band[peak_idx] / 2
                left_idx = np.where(mag_band[:peak_idx] <= half_max)[0]
                right_idx = np.where(mag_band[peak_idx:] <= half_max)[0]
                bw = (right_idx[0] + peak_idx) - left_idx[-1] if len(left_idx) and len(right_idx) else 0
                feat["fft_peak_width"] = bw * (freq[1] - freq[0]) * 60
                for low_f, high_f, label in [(0.8, 1.4, "low"), (1.4, 2.2, "mid"), (2.2, 3.0, "high")]:
                    mask = (freq[band] >= low_f) & (freq[band] <= high_f)
                    feat[f"psd_{label}_energy"] = np.sum(mag_band[mask]) / total
            else:
                self._fill_frequency_defaults(feat)
        else:
            self._fill_frequency_defaults(feat)

        analytic = hilbert(sig_filt)
        envelope = np.abs(analytic)
        mag_env = np.abs(np.fft.rfft(envelope))
        if np.any(band):
            mag_env_band = mag_env[band]
            total_env = np.sum(mag_env_band)
            if total_env > 1e-9:
                feat["env_hr"] = freq[band][np.argmax(mag_env_band)] * 60
                feat["env_power_ratio"] = np.max(mag_env_band) / total_env
            else:
                feat["env_hr"] = 0.0
                feat["env_power_ratio"] = 0.0
        else:
            feat["env_hr"] = 0.0
            feat["env_power_ratio"] = 0.0

        try:
            acf = correlate(sig_filt, sig_filt, mode="full")[len(sig_filt) - 1:]
            min_lag = int(fs * 0.33)
            max_lag = int(fs * 1.5)
            if len(acf) > max_lag:
                acf_band = acf[min_lag:max_lag]
                peak_lag = np.argmax(acf_band) + min_lag
                feat["acf_hr"] = 60 * fs / peak_lag if peak_lag > 0 else 0
                feat["acf_peak_val"] = acf_band.max() / acf[0] if acf[0] > 0 else 0
            else:
                feat["acf_hr"] = 0.0
                feat["acf_peak_val"] = 0.0
        except Exception:
            feat["acf_hr"] = 0.0
            feat["acf_peak_val"] = 0.0

        try:
            coeffs = pywt.wavedec(sig_filt, "db4", level=4)
            total_energy = np.sum([np.sum(c ** 2) for c in coeffs])
            for i, c in enumerate(coeffs):
                feat[f"wav_energy_l{i}"] = np.sum(c ** 2) / total_energy if total_energy > 0 else 0.0
        except Exception:
            for i in range(5):
                feat[f"wav_energy_l{i}"] = 0.0

        if len(peaks) >= 2:
            slopes = []
            valleys = np.concatenate([[0], peaks[:-1] + (peaks[1:] - peaks[:-1]) // 2, [len(smooth) - 1]])
            for p, v1 in zip(peaks, valleys[:-1]):
                if p - v1 > 1:
                    slopes.append((smooth[p] - smooth[v1]) / (p - v1))
            feat["mean_upslope"] = np.mean(slopes) if slopes else 0
            feat["max_upslope"] = np.max(slopes) if slopes else 0
        else:
            feat["mean_upslope"] = 0
            feat["max_upslope"] = 0
        return feat

    def _fill_frequency_defaults(self, feat):
        for k in ["fft_hr", "fft_power_ratio", "fft_promin", "psd_entropy", "fft_peak_width", "psd_low_energy", "psd_mid_energy", "psd_high_energy"]:
            feat[k] = 0.0

    def _extract_all_features(self, red_sig, ir_sig):
        red_feat = self._compute_single_channel_features(red_sig)
        ir_feat = self._compute_single_channel_features(ir_sig)
        if red_feat is None or ir_feat is None:
            return None
        feat = {f"red_{k}": v for k, v in red_feat.items()}
        feat.update({f"ir_{k}": v for k, v in ir_feat.items()})
        feat["R_value"] = np.clip(feat["red_ac_dc"] / feat["ir_ac_dc"], 0.3, 1.2) if feat["ir_ac_dc"] and feat["ir_ac_dc"] > 1e-9 else 0.5
        feat.update({f"quality_{k}": v for k, v in self._evaluate_signal_quality(feat).items()})
        return feat

    def _evaluate_signal_quality(self, features):
        red_acdc = features.get("red_ac_dc", 0)
        ir_acdc = features.get("ir_ac_dc", 0)
        r_val = features.get("R_value", 0.5)
        red_band = features.get("red_psd_low_energy", 0) + features.get("red_psd_mid_energy", 0) + features.get("red_psd_high_energy", 0)
        ir_band = features.get("ir_psd_low_energy", 0) + features.get("ir_psd_mid_energy", 0) + features.get("ir_psd_high_energy", 0)
        hr_methods = [
            features.get(f"{prefix}_{key}", 0)
            for prefix in ["red", "ir"]
            for key in ["acf_hr", "fft_hr", "hr_peak"]
            if 40 <= features.get(f"{prefix}_{key}", 0) <= 200
        ]
        consistency = max(0.0, 1.0 - np.mean(np.abs(np.diff(sorted(hr_methods)))) / 30) if len(hr_methods) >= 3 else 0.0
        quality = {
            "acdc_ok": 1.0 if (0.0005 <= red_acdc <= 0.1) and (0.0005 <= ir_acdc <= 0.1) else 0.0,
            "r_ok": 1.0 if self.R_RANGE[0] <= r_val <= self.R_RANGE[1] else 0.0,
            "band_energy_ok": min(1.0, ((red_band + ir_band) / 2.0) * 2),
            "peaks_ok": 1.0 if features.get("red_peak_count", 0) >= 2 and features.get("ir_peak_count", 0) >= 2 else 0.0,
            "hr_consistency_ok": consistency,
        }
        quality["total_score"] = sum(quality[k] * 0.2 for k in quality)
        return quality

    def _robust_hr_estimate(self, features):
        estimates = [
            features.get(f"{prefix}_{key}", 0)
            for prefix in ["red", "ir"]
            for key in ["acf_hr", "fft_hr", "hr_peak", "env_hr"]
            if 40 <= features.get(f"{prefix}_{key}", 0) <= 200
        ]
        if not estimates:
            return 70.0
        estimates = np.asarray(estimates)
        median = np.median(estimates)
        mad = np.median(np.abs(estimates - median))
        if mad == 0:
            return median
        inliers = estimates[np.abs(estimates - median) < 2.5 * mad]
        return np.mean(inliers) if len(inliers) >= 2 else median

    def predict(self, red_sig, ir_sig):
        feat = self._extract_all_features(red_sig, ir_sig)
        if feat is None:
            raise ValueError("特征提取失败")
        x_test = pd.DataFrame([feat])[self.feature_names].fillna(0)
        quality_score = feat.get("quality_total_score", 0.5)
        if quality_score < 0.3:
            hr_pred = 0
            pred_spo2 = 0
            pred_fo2hb = 0
        else:
            use_ml = self.outlier_detector is None or self.outlier_detector.predict(x_test)[0] != -1
            ml_hr = np.clip(self.models["hr"].predict(x_test)[0], *self.HR_RANGE) if use_ml else None
            fused_hr = self._robust_hr_estimate(feat)
            if ml_hr is not None:
                hr_pred = int(np.clip((ml_hr + fused_hr) / 2.0 if abs(ml_hr - fused_hr) < 15 else ml_hr, *self.HR_RANGE))
            else:
                hr_pred = int(np.clip(fused_hr, *self.HR_RANGE))
            classic_spo2 = np.clip(110 - 25 * feat.get("R_value", 0.5), *self.SPO2_RANGE)
            model_spo2 = self.models["spo2"].predict(x_test)[0]
            pred_spo2 = int(classic_spo2 if abs(model_spo2 - classic_spo2) > 0.1 * classic_spo2 else np.clip(model_spo2, *self.SPO2_RANGE))
            pred_fo2hb = int(np.clip(self.models["fo2hb"].predict(x_test)[0], *self.SPO2_RANGE))
        return {
            "预测心率": hr_pred,
            "预测血氧": pred_spo2,
            "预测FO2Hb": pred_fo2hb,
            "预测血糖": round(float(self.models["glucose"].predict(x_test)[0]), 1),
            "预测K+": round(float(self.models["k"].predict(x_test)[0]), 2),
            "预测Na+": round(float(self.models["na"].predict(x_test)[0]), 1),
            "信号质量": round(float(quality_score), 2),
        }


class EnhancedDataLoader:
    def normalize_spectrum_signal(self, signal_data):
        signal_data = np.asarray(signal_data, dtype=float)
        if len(signal_data) == 0:
            return np.array([5.0])
        if len(signal_data) > 10:
            scores = np.abs(zscore(signal_data, nan_policy="omit"))
            if len(scores) == len(signal_data):
                cleaned = signal_data[scores < 3]
                if len(cleaned) > 0:
                    signal_data = cleaned
        if len(signal_data) == 0:
            return np.array([5.0])
        mn, mx = np.min(signal_data), np.max(signal_data)
        if mx - mn < 1e-10:
            return np.ones_like(signal_data) * 5.0
        return np.clip((signal_data - mn) / (mx - mn) * 10.0, 0, 12)

    def process_spectrum_signals(self, red_original, ir_original):
        red_original = np.asarray(red_original, dtype=float)
        ir_original = np.asarray(ir_original, dtype=float)
        min_len = min(len(red_original), len(ir_original))
        if min_len < 10:
            return None
        red_norm = self.normalize_spectrum_signal(red_original[:min_len])
        ir_norm = self.normalize_spectrum_signal(ir_original[:min_len])
        score, snr, pulse_score, length_score, pulse_count = self._quality(red_norm, ir_norm)
        features = {"信号质量评分": score, "信噪比评分": snr, "脉搏数评分": pulse_score, "信号长度评分": length_score}
        if score < 3:
            red_norm = self._repair(red_norm, score, pulse_count)
            ir_norm = self._repair(ir_norm, score, pulse_count)
            score, snr, pulse_score, length_score, pulse_count = self._quality(red_norm, ir_norm)
            features["信号质量评分_修复后"] = score
            features["信号修复标记"] = 1
        elif score < 6:
            window = min(5, len(red_norm) // 10)
            if window >= 3:
                red_norm = np.convolve(red_norm, np.ones(window) / window, mode="same")
                ir_norm = np.convolve(ir_norm, np.ones(window) / window, mode="same")
            features["信号修复标记"] = 0
        else:
            features["信号修复标记"] = 0
        ratio = self._ratio(red_norm, ir_norm)
        ratio[ratio <= 0] = 0.1
        ratio = np.clip(ratio, 0.3, 3.0)
        features.update({
            "R值_mean": np.mean(ratio),
            "R值_std": np.std(ratio),
            "R值_min": np.min(ratio),
            "R值_max": np.max(ratio),
            "R值异常标记": 1 if (np.mean(ratio) < 0.5 or np.mean(ratio) > 2.0) else 0,
            "R值极端异常标记": 1 if (np.mean(ratio) < 0.3 or np.mean(ratio) > 3.0) else 0,
            "红光_mean": np.mean(red_norm),
            "红光_std": np.std(red_norm),
            "红外光_mean": np.mean(ir_norm),
            "红外光_std": np.std(ir_norm),
        })
        red_acdc = self._acdc(red_norm)
        ir_acdc = self._acdc(ir_norm)
        pi_est = (red_acdc + ir_acdc) / 2 * 100
        features.update({"红光_AC_DC_ratio": red_acdc, "红外光_AC_DC_ratio": ir_acdc, "AC_DC_ratio_diff": red_acdc - ir_acdc, "PI_估算": pi_est})
        for prefix, sig in [("红光", red_norm), ("红外光", ir_norm)]:
            features.update({f"{prefix}_{k}": v for k, v in self._frequency(sig).items()})
            features.update({f"{prefix}_{k}": v for k, v in self._pulse(sig, score).items()})
        features["信号信噪比"] = np.mean(red_norm) / (np.std(red_norm) + 1e-10)
        features["信号长度"] = len(red_norm)
        features["PI_ac_dc_ratio"] = pi_est / 100
        features["PI_variability"] = np.std(red_norm) / (np.std(ir_norm) + 1e-10)
        return features

    def _quality(self, red, ir):
        red_snr = np.mean(red) / (np.std(red) + 1e-10)
        ir_snr = np.mean(ir) / (np.std(ir) + 1e-10)
        snr = (red_snr + ir_snr) / 2
        snr_score = max(0, min(10, (snr - 1) / 6 * 10)) if snr > 1 else 0
        pulse_count = min(len(find_peaks(red, distance=20, prominence=0.1 * np.std(red))[0]), len(find_peaks(ir, distance=20, prominence=0.1 * np.std(ir))[0]))
        pulse_score = max(0, min(10, (pulse_count - 3) / 12 * 10)) if pulse_count >= 3 else 0
        length_score = max(0, min(10, (len(red) - 100) / 900 * 10)) if len(red) >= 100 else 0
        return snr_score * 0.4 + pulse_score * 0.3 + length_score * 0.3, snr_score, pulse_score, length_score, pulse_count

    def _repair(self, signal_data, score, pulse_count):
        repaired = signal_data.copy()
        if len(repaired) < 500 and len(repaired) > 10:
            repaired = np.interp(np.linspace(0, 1, 500), np.linspace(0, 1, len(repaired)), repaired)
        if score < 3 and len(repaired) >= 64:
            try:
                coeffs = pywt.wavedec(repaired, "db4", level=3)
                threshold = 0.1 * np.std(repaired)
                repaired = pywt.waverec([coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]], "db4")
            except Exception:
                pass
        return repaired

    def _ratio(self, red, ir):
        min_len = min(len(red), len(ir))
        if min_len == 0:
            return np.ones(1) * 1.2
        red = red[:min_len]
        ir = ir[:min_len]
        mask = ~(np.isnan(red) | np.isnan(ir) | np.isinf(red) | np.isinf(ir) | (ir == 0))
        if np.sum(mask) < 10:
            return np.ones_like(red) * 1.2
        result = np.ones_like(red) * 1.2
        result[mask] = red[mask] / (ir[mask] + 1e-10)
        return result

    def _acdc(self, sig):
        window = min(100, len(sig) // 10)
        window = max(window, 3)
        dc = np.convolve(sig, np.ones(window) / window, mode="same")
        ac = sig - dc
        return np.mean(np.abs(ac)) / (np.mean(np.abs(dc)) + 1e-10)

    def _frequency(self, sig):
        if len(sig) < 64:
            return {"脉搏功率比": 0.1, "主导频率": 1.0}
        freqs, psd = welch(sig, fs=100, nperseg=min(256, len(sig)))
        total = np.sum(psd) + 1e-10
        return {
            "脉搏功率比": np.sum(psd[(freqs >= 0.5) & (freqs <= 4)]) / total,
            "主导频率": freqs[np.argmax(psd)] if len(psd) > 0 else 0,
        }

    def _pulse(self, sig, quality_score):
        prom = 0.05 if quality_score < 3 else 0.08 if quality_score < 6 else 0.1
        peaks, _ = find_peaks(sig, distance=20, prominence=prom * np.std(sig))
        if len(peaks) >= 2:
            return {"pulse_count": len(peaks), "interval_mean": np.mean(np.diff(peaks)), "amp_mean": np.mean(sig[peaks])}
        return {"pulse_count": len(peaks), "interval_mean": 0, "amp_mean": 0}


class EnhancedFeatureEngineer:
    def create_advanced_features(self, df):
        out = df.copy()
        if "PI_估算" in out:
            out["PI_log"] = np.log1p(out["PI_估算"].clip(lower=0.01))
            out["PI_平方"] = out["PI_估算"] ** 2
        if "血氧饱和度" in out and "PI_估算" in out:
            out["PI_血氧交互"] = out["PI_估算"] * out["血氧饱和度"]
        if "血氧饱和度" in out and "R值_mean" in out:
            out["血氧_R交互"] = out["血氧饱和度"] * out["R值_mean"]
        if "氧合血红蛋白分数(FO2Hb)" in out and "R值_mean" in out:
            out["FO2Hb_R交互"] = out["氧合血红蛋白分数(FO2Hb)"] * out["R值_mean"]
        if "血糖" in out and "血氧饱和度" in out:
            out["血糖_血氧交互"] = out["血糖"] * out["血氧饱和度"]
        if "K+" in out and "Na+" in out:
            out["K_Na比"] = out["K+"] / (out["Na+"] + 1e-10)
        for col in ["血氧饱和度", "氧合血红蛋白分数(FO2Hb)", "血糖", "K+", "Na+", "R值_mean", "红光_mean", "红外光_mean", "信号信噪比"]:
            if col in out and out[col].min() > 0:
                out[f"{col}_log"] = np.log1p(out[col])
                out[f"{col}_sqrt"] = np.sqrt(out[col])
        if "信号信噪比" in out and "红光_pulse_count" in out:
            out["信号质量指数"] = out["信号信噪比"] * np.log1p(out["红光_pulse_count"] + 1)
        if "信号质量评分" in out:
            out["信号质量等级"] = pd.cut(out["信号质量评分"], bins=[0, 3, 6, 10], labels=["差", "中", "优"]).astype(str)
        return out


class MultiTaskPredictor:
    def __init__(self, model_data, hb_model_path):
        self.models = {
            "hemoglobin": XGBoostJsonRegressor(hb_model_path),
            "pi": GradientBoostingJsonRegressor(model_data["models"]["pi"]),
            "lactate": RidgeJsonRegressor(model_data["models"]["lactate"]),
        }
        self.scalers = {key: StandardScalerLite(value) for key, value in model_data["scalers"].items()}
        self.label_encoders = {key: JsonLabelEncoder(value) for key, value in model_data.get("label_encoders", {}).items()}
        self.feature_sets = model_data["feature_sets"]
        self.pi_transform = model_data.get("pi_transform", "log")
        self.loader = EnhancedDataLoader()
        self.engineer = EnhancedFeatureEngineer()

    def predict(self, red_sig, ir_sig, clinical):
        spec = self.loader.process_spectrum_signals(red_sig, ir_sig)
        if spec is None:
            raise ValueError("光谱特征提取失败")
        df = self.engineer.create_advanced_features(pd.DataFrame([{**clinical, **spec}]))
        df = self._fill_label_dependent_features(df)
        results = {}
        for task in ["hemoglobin", "pi", "lactate"]:
            feat_list = self.feature_sets.get(task, [])
            x_raw = pd.DataFrame(index=[0])
            for feature in feat_list:
                x_raw[feature] = df[feature].values[0] if feature in df.columns else 0.0
            x_encoded = self._encode_categorical_features(x_raw)
            scaler_key = "hemoglobin" if task == "hemoglobin" else task
            x_scaled = self.scalers[scaler_key].transform(x_encoded)
            pred = self.models[task].predict(x_scaled)[0]
            if task == "pi" and self.pi_transform == "log":
                pred = np.expm1(pred)
            if task == "lactate" and pred < 0:
                pred = 0.0
            results[task] = round(float(pred), 2)
        return {"血红蛋白": results.get("hemoglobin"), "PI": results.get("pi"), "乳酸": results.get("lactate")}

    def _fill_label_dependent_features(self, df):
        out = df.copy()
        if "血红蛋白" not in out:
            out["血红蛋白"] = 120.0
        if "血红蛋白区间" not in out:
            hb = out["血红蛋白"].values[0]
            if hb < 80:
                interval, num = "极重度贫血", 0
            elif hb < 120:
                interval, num = "贫血", 1
            elif hb <= 160:
                interval, num = "正常", 2
            elif hb <= 170:
                interval, num = "偏高", 3
            else:
                interval, num = "重度偏高", 4
            out["血红蛋白区间"] = interval
            out["血红蛋白区间_num"] = num
        if "乳酸" not in out:
            out["乳酸"] = 2.0
        if "乳酸_log" not in out:
            out["乳酸_log"] = np.log1p(max(out["乳酸"].values[0], 0.01))
        if "乳酸_平方根" not in out:
            out["乳酸_平方根"] = np.sqrt(max(out["乳酸"].values[0], 0) + 1)
        return out

    def _encode_categorical_features(self, df):
        out = df.copy()
        for col in out.columns:
            is_text_column = (
                pd.api.types.is_object_dtype(out[col].dtype)
                or pd.api.types.is_string_dtype(out[col].dtype)
            )
            if is_text_column:
                if col in self.label_encoders:
                    out[col] = self.label_encoders[col].transform(out[col].astype(str))
                elif col == "信号质量等级":
                    out[col] = out[col].map({"差": 0, "中": 1, "优": 2}).fillna(0).astype(int)
                elif col == "血红蛋白区间":
                    out[col] = out[col].map({"极重度贫血": 0, "贫血": 1, "正常": 2, "偏高": 3, "重度偏高": 4}).fillna(2).astype(int)
                else:
                    out[col] = 0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        return out


class OriginLiveInferenceService:
    def __init__(self, data_dir):
        origin_model_dir = os.path.join(data_dir, "origin_model")
        with open(os.path.join(origin_model_dir, "origin_base_models.json"), "r", encoding="utf-8") as f:
            self.base = BasePhysioPredictor(json.load(f))
        with open(os.path.join(origin_model_dir, "origin_multitask_models.json"), "r", encoding="utf-8") as f:
            multi_data = json.load(f)
        self.multi = MultiTaskPredictor(multi_data, os.path.join(origin_model_dir, "origin_multitask_hb_model.json"))

    def predict(self, patient_id, red_values, ir_values):
        red = _to_float_array(red_values)
        ir = _to_float_array(ir_values)
        base = self.base.predict(red, ir)
        clinical = {
            "血氧饱和度": base["预测血氧"],
            "氧合血红蛋白分数(FO2Hb)": base["预测FO2Hb"],
            "血糖": base["预测血糖"] if base["预测血糖"] is not None else 5.5,
            "K+": base["预测K+"] if base["预测K+"] is not None else 4.0,
            "Na+": base["预测Na+"] if base["预测Na+"] is not None else 140.0,
        }
        multi = self.multi.predict(red, ir, clinical)
        patient_info = {
            "编号": str(patient_id),
            "心率": str(base["预测心率"]),
            "血氧饱和度": str(base["预测血氧"]),
            "氧合血红蛋白分数(FO2Hb)": str(base["预测FO2Hb"]),
            "血糖": str(base["预测血糖"]),
            "K+": str(base["预测K+"]),
            "Na+": str(base["预测Na+"]),
            "信号质量": str(base["信号质量"]),
        }
        prediction = {
            "patient_id": str(patient_id),
            "hemoglobin": {"value": round(float(multi["血红蛋白"]), 1), "unit": "g/L", "clinical_interpretation": self._interpret_hb(multi["血红蛋白"])},
            "perfusion_index": {"value": round(float(multi["PI"]), 3), "classification": self._classify_pi(multi["PI"]), "interpretation": "血流灌注指数良好"},
            "lactate": {"value": round(float(multi["乳酸"]), 1), "unit": "mmol/L", "clinical_interpretation": self._interpret_lactate(multi["乳酸"])},
            "prediction_time": datetime.now().isoformat(),
            "prediction_id": f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }
        return {"success": True, "patient_info": patient_info, "prediction": prediction, "error": None}

    def _classify_pi(self, value):
        if value < 0.3:
            return "弱灌注"
        if value < 1.0:
            return "可接受"
        return "最佳"

    def _interpret_hb(self, value):
        if value < 80:
            return "极重度贫血可能"
        if value < 100:
            return "中度贫血可能"
        if value < 120:
            return "轻度贫血可能"
        if value <= 160:
            return "正常范围"
        if value <= 170:
            return "偏高"
        return "重度偏高"

    def _interpret_lactate(self, value):
        if value < 1.0:
            return "正常"
        if value < 2.0:
            return "轻度升高"
        if value < 4.0:
            return "中度升高"
        return "重度升高"


def _to_float_array(values):
    try:
        return np.asarray(list(values), dtype=float)
    except TypeError:
        if hasattr(values, "size") and hasattr(values, "get"):
            return np.asarray([values.get(i) for i in range(values.size())], dtype=float)
        raise


def predict_live(patient_id, red_values, ir_values, data_dir):
    global _service
    try:
        if _service is None:
            _service = OriginLiveInferenceService(data_dir)
        return json.dumps(_service.predict(patient_id, red_values, ir_values), ensure_ascii=False)
    except Exception as exc:
        traceback.print_exc()
        return json.dumps({"success": False, "patient_info": None, "prediction": None, "error": f"实时推理失败: {exc}"}, ensure_ascii=False)
