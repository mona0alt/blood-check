import pandas as pd
import numpy as np
import pywt
import json
import joblib
import warnings
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt, hilbert, correlate
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import optuna
import os
import glob
import re
from typing import Dict, Optional, Union, List, Tuple

warnings.filterwarnings('ignore')


class ImprovedPhysioPredictor:
    """增强型生理指标预测器：集成信号质量评估、多源心率融合、鲁棒回退与极端值处理
       新增 K+、Na+，并将所有模型融合为单一 pkl 文件"""

    def __init__(self,
                 spectrum_dir: str,
                 crf_path: str,
                 model_dir: Optional[str] = None,
                 train_ratio: float = 0.8,
                 random_state: int = 42,
                 debug: bool = False):
        self.SPECTRUM_DIR = spectrum_dir
        self.CRF_PATH = crf_path

        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.MODEL_DIR = model_dir
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # ===== 单一模型包 =====
        self.MODEL_PACKAGE_PATH = os.path.join(self.MODEL_DIR, "models.pkl")

        # 通道配置
        self.CHANNEL_CONFIG = {"red": "C4", "ir": "C5", "fs": 100}
        self.HR_RANGE = (40, 200)
        self.SPO2_RANGE = (70, 100)
        self.R_RANGE = (0.3, 1.2)
        self.DEBUG = debug
        self.random_state = random_state

        self.crf_df = self._load_crf_data()
        self.train_ids, self.test_ids = self._split_dataset(train_ratio, random_state)

        # 模型占位
        self.model_hr = None
        self.model_spo2 = None
        self.model_fo2hb = None
        self.model_glu = None
        self.model_k = None
        self.model_na = None
        self.outlier_detector = None
        self.feature_names = None

    # ===================== 基础工具 =====================
    def _unify_id(self, id_str: Union[str, int]) -> str:
        if pd.isna(id_str):
            return ""
        return re.sub(r"[^0-9]", "", str(id_str)).lstrip("0") or "0"

    def _load_crf_data(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(self.CRF_PATH, engine="openpyxl")
        except:
            df = pd.read_csv(self.CRF_PATH, encoding='utf-8-sig')
        df["统一ID"] = df["编号（日期+住院号）"].apply(self._unify_id)
        return df

    def _find_spectrum_file(self, patient_id: str) -> Optional[str]:
        target = self._unify_id(patient_id)
        for f in glob.glob(os.path.join(self.SPECTRUM_DIR, "*.csv")):
            if self._unify_id(os.path.splitext(os.path.basename(f))[0]) == target:
                return f
        return None

    def _load_spectrum_signals(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        red_col = self.CHANNEL_CONFIG["red"]
        ir_col = self.CHANNEL_CONFIG["ir"]
        if red_col not in df.columns or ir_col not in df.columns:
            raise ValueError(f"CSV中缺少必要的列：{red_col} 或 {ir_col}")
        red = pd.to_numeric(df[red_col], errors='coerce').values.astype(float)
        ir = pd.to_numeric(df[ir_col], errors='coerce').values.astype(float)
        return red, ir

    def _bandpass_filter(self, signal, fs=100, low=0.8, high=3.0, order=4):
        nyq = 0.5 * fs
        if len(signal) < 15:
            return signal
        try:
            b, a = butter(order, [low / nyq, high / nyq], 'band')
            return filtfilt(b, a, signal)
        except:
            return signal

    def _extract_acdc(self, signal: np.ndarray) -> float:
        signal = np.asarray(signal, dtype=float)
        signal = signal[~np.isnan(signal)]
        if len(signal) < 20:
            return np.nan
        win = max(5, len(signal) // 5)
        if win % 2 == 0:
            win += 1
        try:
            detrended = signal - savgol_filter(signal, win, 2)
        except:
            detrended = signal - np.mean(signal)
        dc = np.mean(signal)
        if dc <= 0:
            return np.nan
        ac = (np.max(detrended) - np.min(detrended)) / 2 if not np.isnan(detrended).any() else 1.5 * np.std(detrended)
        return ac / dc

    # ===================== 数据集划分 =====================
    def _get_all_valid_ids(self) -> List[str]:
        valid = []
        for f in glob.glob(os.path.join(self.SPECTRUM_DIR, "*.csv")):
            uid = self._unify_id(os.path.splitext(os.path.basename(f))[0])
            if uid in self.crf_df["统一ID"].values:
                match = self.crf_df[self.crf_df["统一ID"] == uid]
                if not match.empty:
                    hr = match.iloc[0]["心率"] if pd.notna(match.iloc[0]["心率"]) else None
                    spo2 = match.iloc[0]["血氧饱和度"] if pd.notna(match.iloc[0]["血氧饱和度"]) else None
                    if hr is not None and spo2 is not None:
                        valid.append(uid)
        return list(set(valid))

    def _split_dataset(self, train_ratio, random_state):
        all_ids = self._get_all_valid_ids()
        if len(all_ids) < 2:
            print(f"⚠️ 有效样本数: {len(all_ids)}，无法划分训练/测试集")
            return all_ids, []
        train_ids, test_ids = train_test_split(all_ids, train_size=train_ratio,
                                               random_state=random_state, shuffle=True)
        print(f"数据集：训练 {len(train_ids)} 人，测试 {len(test_ids)} 人")
        return train_ids, test_ids

    # ===================== 信号质量评估 =====================
    def _evaluate_signal_quality(self, red_sig, ir_sig, features: dict) -> dict:
        quality = {}
        red_acdc = features.get('red_ac_dc', 0)
        ir_acdc = features.get('ir_ac_dc', 0)
        acdc_valid = (0.0005 <= red_acdc <= 0.1) and (0.0005 <= ir_acdc <= 0.1)
        quality['acdc_ok'] = 1.0 if acdc_valid else 0.0

        r_val = features.get('R_value', 0.5)
        r_valid = self.R_RANGE[0] <= r_val <= self.R_RANGE[1]
        quality['r_ok'] = 1.0 if r_valid else 0.0

        red_band_energy = features.get('red_psd_low_energy', 0) + features.get('red_psd_mid_energy', 0) + features.get(
            'red_psd_high_energy', 0)
        ir_band_energy = features.get('ir_psd_low_energy', 0) + features.get('ir_psd_mid_energy', 0) + features.get(
            'ir_psd_high_energy', 0)
        band_energy_ratio = (red_band_energy + ir_band_energy) / 2.0
        quality['band_energy_ok'] = min(1.0, band_energy_ratio * 2)

        peak_count_r = features.get('red_peak_count', 0)
        peak_count_ir = features.get('ir_peak_count', 0)
        valid_peaks = (peak_count_r >= 2 and peak_count_ir >= 2)
        quality['peaks_ok'] = 1.0 if valid_peaks else 0.0

        hr_methods = []
        for prefix in ['red', 'ir']:
            for k in ['acf_hr', 'fft_hr', 'hr_peak']:
                val = features.get(f'{prefix}_{k}', 0)
                if 40 <= val <= 200:
                    hr_methods.append(val)
        consistency = 0.0
        if len(hr_methods) >= 3:
            pairwise_diff = np.mean(np.abs(np.diff(sorted(hr_methods))))
            consistency = max(0.0, 1.0 - pairwise_diff / 30)
        quality['hr_consistency_ok'] = consistency

        quality['total_score'] = (quality['acdc_ok'] * 0.2 + quality['r_ok'] * 0.2 +
                                  quality['band_energy_ok'] * 0.2 + quality['peaks_ok'] * 0.2 +
                                  quality['hr_consistency_ok'] * 0.2)
        return quality

    # ===================== 单通道特征提取（与之前相同，略） =====================
    def _compute_single_channel_features(self, signal: np.ndarray, fs=100) -> dict:
        signal = np.asarray(signal, dtype=float)
        sig = signal[~np.isnan(signal)]
        if len(sig) < 2 * fs:
            return None

        feat = {}
        feat['mean'] = np.mean(sig)
        feat['std'] = np.std(sig)
        feat['skew'] = skew(sig) if len(sig) > 3 else 0.0
        feat['kurt'] = kurtosis(sig) if len(sig) > 4 else 0.0
        feat['iqr'] = np.percentile(sig, 75) - np.percentile(sig, 25)
        feat['min'] = np.min(sig)
        feat['max'] = np.max(sig)
        feat['ac_dc'] = self._extract_acdc(signal)

        feat['rms'] = np.sqrt(np.mean(sig ** 2))
        feat['zero_crossing_rate'] = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0) / len(sig)
        feat['wave_length'] = np.sum(np.abs(np.diff(sig))) / len(sig)

        sig_filt = self._bandpass_filter(sig, fs)
        smooth = savgol_filter(sig_filt, 11, 2)
        prominence = max(0.1 * np.std(smooth), 0.05)
        peaks, props = find_peaks(smooth, distance=int(fs * 0.3),
                                  prominence=prominence, width=(5, 60))
        feat['peak_count'] = len(peaks)

        if len(peaks) >= 2:
            ibis = np.diff(peaks) / fs
            valid_mask = (ibis >= 0.33) & (ibis <= 1.5)
            if len(ibis) >= 2:
                ibi_diff = np.abs(np.diff(ibis))
                valid_mask[1:] &= (ibi_diff / ibis[:-1] < 0.3)
            valid_ibis = ibis[valid_mask]
            if len(valid_ibis) >= 2:
                mean_ibi = np.mean(valid_ibis)
                feat['mean_ibi'] = mean_ibi
                feat['ibi_std'] = np.std(valid_ibis)
                feat['hr_peak'] = 60.0 / mean_ibi
                peak_vals = smooth[peaks[:len(valid_ibis) + 1]] if len(peaks) >= len(valid_ibis) + 1 else smooth[peaks]
                feat['peak_amp_cv'] = np.std(peak_vals) / np.mean(peak_vals) if np.mean(peak_vals) > 0 else 0
            else:
                feat['mean_ibi'] = 0.0;
                feat['ibi_std'] = 0.0;
                feat['hr_peak'] = 0.0;
                feat['peak_amp_cv'] = 0.0
        else:
            feat['mean_ibi'] = 0.0;
            feat['ibi_std'] = 0.0;
            feat['hr_peak'] = 0.0;
            feat['peak_amp_cv'] = 0.0

        n = len(sig_filt)
        freq = np.fft.rfftfreq(n, 1 / fs)
        mag = np.abs(np.fft.rfft(sig_filt))
        band = (freq >= 0.8) & (freq <= 3.0)
        if np.any(band):
            mag_band = mag[band]
            total = np.sum(mag_band)
            if total > 1e-9:
                peak_idx = np.argmax(mag_band)
                feat['fft_hr'] = freq[band][peak_idx] * 60
                feat['fft_power_ratio'] = mag_band[peak_idx] / total
                sorted_mag = np.sort(mag_band)
                feat['fft_promin'] = sorted_mag[-1] / sorted_mag[-2] if len(sorted_mag) > 1 else 10.0
                psd = mag_band / total
                feat['psd_entropy'] = -np.sum(psd * np.log2(psd + 1e-12))
                half_max = mag_band[peak_idx] / 2
                left_idx = np.where(mag_band[:peak_idx] <= half_max)[0]
                right_idx = np.where(mag_band[peak_idx:] <= half_max)[0]
                bw = (right_idx[0] + peak_idx) - left_idx[-1] if len(left_idx) and len(right_idx) else 0
                feat['fft_peak_width'] = bw * (freq[1] - freq[0]) * 60
                for low_f, high_f, label in [(0.8, 1.4, 'low'), (1.4, 2.2, 'mid'), (2.2, 3.0, 'high')]:
                    mask = (freq[band] >= low_f) & (freq[band] <= high_f)
                    feat[f'psd_{label}_energy'] = np.sum(mag_band[mask]) / total
            else:
                for k in ['fft_hr', 'fft_power_ratio', 'fft_promin', 'psd_entropy', 'fft_peak_width',
                          'psd_low_energy', 'psd_mid_energy', 'psd_high_energy']:
                    feat[k] = 0.0
        else:
            for k in ['fft_hr', 'fft_power_ratio', 'fft_promin', 'psd_entropy', 'fft_peak_width',
                      'psd_low_energy', 'psd_mid_energy', 'psd_high_energy']:
                feat[k] = 0.0

        analytic = hilbert(sig_filt)
        envelope = np.abs(analytic)
        mag_env = np.abs(np.fft.rfft(envelope))
        if np.any(band):
            mag_env_band = mag_env[band]
            total_env = np.sum(mag_env_band)
            if total_env > 1e-9:
                feat['env_hr'] = freq[band][np.argmax(mag_env_band)] * 60
                feat['env_power_ratio'] = np.max(mag_env_band) / total_env
            else:
                feat['env_hr'] = 0.0;
                feat['env_power_ratio'] = 0.0
        else:
            feat['env_hr'] = 0.0;
            feat['env_power_ratio'] = 0.0

        try:
            acf = correlate(sig_filt, sig_filt, mode='full')
            acf = acf[len(acf) // 2:]
            min_lag = int(fs * 0.33)
            max_lag = int(fs * 1.5)
            if len(acf) > max_lag:
                acf_band = acf[min_lag:max_lag]
                peak_lag = np.argmax(acf_band) + min_lag
                feat['acf_hr'] = 60 * fs / peak_lag if peak_lag > 0 else 0
                feat['acf_peak_val'] = acf_band.max() / acf[0] if acf[0] > 0 else 0
            else:
                feat['acf_hr'] = 0.0;
                feat['acf_peak_val'] = 0.0
        except:
            feat['acf_hr'] = 0.0;
            feat['acf_peak_val'] = 0.0

        try:
            coeffs = pywt.wavedec(sig_filt, 'db4', level=4)
            total_energy = np.sum([np.sum(c ** 2) for c in coeffs])
            if total_energy > 0:
                for i, c in enumerate(coeffs):
                    feat[f'wav_energy_l{i}'] = np.sum(c ** 2) / total_energy
            else:
                for i in range(5): feat[f'wav_energy_l{i}'] = 0.0
        except:
            for i in range(5): feat[f'wav_energy_l{i}'] = 0.0

        try:
            if len(peaks) >= 2:
                valleys = np.concatenate([[0], peaks[:-1] + (peaks[1:] - peaks[:-1]) // 2, [len(smooth) - 1]])
                slopes = []
                for p, v1, v2 in zip(peaks, valleys[:-1], valleys[1:]):
                    up_segment = smooth[v1:p]
                    if len(up_segment) > 1:
                        slope = (smooth[p] - smooth[v1]) / (p - v1)
                        slopes.append(slope)
                feat['mean_upslope'] = np.mean(slopes) if slopes else 0
                feat['max_upslope'] = np.max(slopes) if slopes else 0
            else:
                feat['mean_upslope'] = 0;
                feat['max_upslope'] = 0
        except:
            feat['mean_upslope'] = 0;
            feat['max_upslope'] = 0

        return feat

    def _extract_all_features(self, red_sig, ir_sig) -> Optional[dict]:
        feat = {}
        red_feat = self._compute_single_channel_features(red_sig)
        ir_feat = self._compute_single_channel_features(ir_sig)
        if red_feat is None or ir_feat is None:
            return None
        for k, v in red_feat.items():
            feat[f'red_{k}'] = v
        for k, v in ir_feat.items():
            feat[f'ir_{k}'] = v
        if feat['red_ac_dc'] and feat['ir_ac_dc'] and feat['ir_ac_dc'] > 1e-9:
            feat['R_value'] = np.clip(feat['red_ac_dc'] / feat['ir_ac_dc'], 0.3, 1.2)
        else:
            feat['R_value'] = 0.5
        quality = self._evaluate_signal_quality(red_sig, ir_sig, feat)
        for k, v in quality.items():
            feat[f'quality_{k}'] = v
        return feat

    # ===================== 异常清洗（不再单独保存模型） =====================
    def _remove_outliers(self, X, y_hr):
        iso = IsolationForest(contamination=0.02, random_state=self.random_state)
        outliers = iso.fit_predict(X)
        mask = outliers == 1
        print(f"移除异常样本: {np.sum(~mask)} / {len(X)}")
        self.outlier_detector = iso
        return X[mask], y_hr[mask]

    # ===================== LightGBM 训练 =====================
    def _train_lgb_model(self, X, y, task_name):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            return np.mean(-scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'random_state': self.random_state, 'n_jobs': -1, 'verbosity': -1})
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X, y)
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        print(f"   {task_name} MAE(训练集): {mae:.2f}")
        return model

    # ===================== 训练主流程 =====================
    def train_ml_models(self):
        print("🔄 提取训练集特征...")
        X_list, y_hr, y_spo2, y_fo2 = [], [], [], []
        y_glu_dict, y_k_dict, y_na_dict = {}, {}, {}
        reasons = {"no_file": 0, "no_match": 0, "bad_label": 0, "bad_signal": 0, "feat_none": 0}

        for uid in self.train_ids:
            file = self._find_spectrum_file(uid)
            if not file:
                reasons["no_file"] += 1
                continue
            match = self.crf_df[self.crf_df["统一ID"] == uid]
            if match.empty:
                reasons["no_match"] += 1
                continue
            hr_val = match.iloc[0]["心率"]
            spo2_val = match.iloc[0]["血氧饱和度"]
            if pd.isna(hr_val) or pd.isna(spo2_val):
                reasons["bad_label"] += 1
                continue
            try:
                red, ir = self._load_spectrum_signals(file)
                if len(red) < 2 * 100 or len(ir) < 2 * 100:
                    reasons["bad_signal"] += 1
                    continue
                feat = self._extract_all_features(red, ir)
                if feat is None:
                    reasons["feat_none"] += 1
                    continue
                X_list.append(feat)
                y_hr.append(int(hr_val))
                y_spo2.append(int(spo2_val))
                fo2_val = match.iloc[0]["氧合血红蛋白分数(FO2Hb)"]
                y_fo2.append(int(fo2_val) if pd.notna(fo2_val) else int(spo2_val))

                if "血糖" in match.columns:
                    glu_val = match.iloc[0]["血糖"]
                    if pd.notna(glu_val):
                        y_glu_dict[uid] = float(glu_val)
                if "K+" in match.columns:
                    k_val = match.iloc[0]["K+"]
                    if pd.notna(k_val):
                        y_k_dict[uid] = float(k_val)
                if "Na+" in match.columns:
                    na_val = match.iloc[0]["Na+"]
                    if pd.notna(na_val):
                        y_na_dict[uid] = float(na_val)
            except Exception as e:
                if self.DEBUG:
                    print(f"[DEBUG] 跳过 {uid}: {e}")
                reasons["bad_signal"] += 1
                continue

        print(f"训练样本提取完成: {len(X_list)} 有效, 跳过原因: {reasons}")
        if len(X_list) < 30:
            raise Exception(f"训练样本不足 ({len(X_list)})，请检查路径、ID匹配、信号长度")

        X = pd.DataFrame(X_list)
        X = X.loc[:, X.apply(pd.Series.nunique) != 1]
        self.feature_names = list(X.columns)

        y_hr_arr = np.array(y_hr)
        X_np = X.values
        X_clean, y_hr_clean = self._remove_outliers(X_np, y_hr_arr)
        X_clean = pd.DataFrame(X_clean, columns=self.feature_names)

        print("⏳ 训练心率模型...")
        self.model_hr = self._train_lgb_model(X_clean, y_hr_clean, "心率")

        print("⏳ 训练血氧模型...")
        self.model_spo2 = self._train_lgb_model(X, np.array(y_spo2), "血氧")

        print("⏳ 训练FO2Hb模型...")
        self.model_fo2hb = self._train_lgb_model(X, np.array(y_fo2), "FO2Hb")

        # 血糖
        if y_glu_dict:
            glu_X, glu_y = [], []
            for i, uid in enumerate(self.train_ids):
                if uid in y_glu_dict and i < len(X):
                    glu_X.append(X.iloc[i].values)
                    glu_y.append(y_glu_dict[uid])
            if len(glu_X) >= 20:
                X_glu = pd.DataFrame(glu_X, columns=self.feature_names)
                y_glu = np.array(glu_y)
                print(f"⏳ 训练血糖模型 (样本数: {len(X_glu)})...")
                self.model_glu = self._train_lgb_model(X_glu, y_glu, "血糖")
            else:
                print("⚠️ 血糖样本不足，跳过")
        else:
            print("⚠️ 未检测到血糖列")

        # K+
        if y_k_dict:
            k_X, k_y = [], []
            for i, uid in enumerate(self.train_ids):
                if uid in y_k_dict and i < len(X):
                    k_X.append(X.iloc[i].values)
                    k_y.append(y_k_dict[uid])
            if len(k_X) >= 20:
                X_k = pd.DataFrame(k_X, columns=self.feature_names)
                y_k = np.array(k_y)
                print(f"⏳ 训练K+模型 (样本数: {len(X_k)})...")
                self.model_k = self._train_lgb_model(X_k, y_k, "K+")
            else:
                print("⚠️ K+样本不足，跳过")
        else:
            print("⚠️ 未检测到K+列")

        # Na+
        if y_na_dict:
            na_X, na_y = [], []
            for i, uid in enumerate(self.train_ids):
                if uid in y_na_dict and i < len(X):
                    na_X.append(X.iloc[i].values)
                    na_y.append(y_na_dict[uid])
            if len(na_X) >= 20:
                X_na = pd.DataFrame(na_X, columns=self.feature_names)
                y_na = np.array(na_y)
                print(f"⏳ 训练Na+模型 (样本数: {len(X_na)})...")
                self.model_na = self._train_lgb_model(X_na, y_na, "Na+")
            else:
                print("⚠️ Na+样本不足，跳过")
        else:
            print("⚠️ 未检测到Na+列")

        # ===== 将所有模型打包成一个 pkl 文件 =====
        package = {
            'model_hr': self.model_hr,
            'model_spo2': self.model_spo2,
            'model_fo2hb': self.model_fo2hb,
            'model_glu': self.model_glu,
            'model_k': self.model_k,
            'model_na': self.model_na,
            'outlier_detector': self.outlier_detector,
            'feature_names': self.feature_names
        }
        joblib.dump(package, self.MODEL_PACKAGE_PATH)
        print(f"✅ 所有模型已打包至: {self.MODEL_PACKAGE_PATH}")

    # ===================== 模型加载（从单一包） =====================
    def _load_models(self):
        if not os.path.exists(self.MODEL_PACKAGE_PATH):
            raise FileNotFoundError(f"模型包未找到: {self.MODEL_PACKAGE_PATH}，请先训练模型")
        package = joblib.load(self.MODEL_PACKAGE_PATH)
        self.model_hr = package['model_hr']
        self.model_spo2 = package['model_spo2']
        self.model_fo2hb = package['model_fo2hb']
        self.model_glu = package.get('model_glu')
        self.model_k = package.get('model_k')
        self.model_na = package.get('model_na')
        self.outlier_detector = package.get('outlier_detector')
        self.feature_names = package['feature_names']
        print(f"✅ 已加载模型包（来自 {self.MODEL_PACKAGE_PATH}）")

    # ===================== 鲁棒心率估计 =====================
    def _robust_hr_estimate(self, features: dict) -> Tuple[float, str]:
        estimates = []
        for prefix in ['red', 'ir']:
            for key in ['acf_hr', 'fft_hr', 'hr_peak', 'env_hr']:
                val = features.get(f'{prefix}_{key}', 0)
                if 40 <= val <= 200:
                    estimates.append(val)

        if len(estimates) == 0:
            return 70.0, "default"

        estimates = np.array(estimates)
        median = np.median(estimates)
        mad = np.median(np.abs(estimates - median))
        if mad == 0:
            return median, "consensus_median"

        inliers = estimates[np.abs(estimates - median) < 2.5 * mad]
        if len(inliers) >= 2:
            fused = np.mean(inliers)
            return fused, "consensus_mean"

        return median, "consensus_median_low_conf"

    # ===================== 预测API =====================
    def predict_by_patient_id(self, patient_id: Union[str, int]) -> Dict:
        if self.model_hr is None:
            return {"code": 500, "msg": "模型未加载", "data": None}

        file_path = self._find_spectrum_file(patient_id)
        if not file_path:
            return {"code": 404, "msg": f"未找到患者 {patient_id} 的文件", "data": None}

        uid = self._unify_id(patient_id)
        crf_row = self.crf_df[self.crf_df["统一ID"] == uid]
        real = {"心率": 0, "血氧": 0, "FO2Hb": 0, "血糖": None, "K+": None, "Na+": None}
        if not crf_row.empty:
            hr_val = crf_row.iloc[0]["心率"]
            real["心率"] = int(hr_val) if pd.notna(hr_val) else 0
            spo2_val = crf_row.iloc[0]["血氧饱和度"]
            real["血氧"] = int(spo2_val) if pd.notna(spo2_val) else 0
            fo2_val = crf_row.iloc[0]["氧合血红蛋白分数(FO2Hb)"]
            real["FO2Hb"] = int(fo2_val) if pd.notna(fo2_val) else 0
            if "血糖" in crf_row.columns:
                glu_val = crf_row.iloc[0]["血糖"]
                real["血糖"] = float(glu_val) if pd.notna(glu_val) else None
            if "K+" in crf_row.columns:
                k_val = crf_row.iloc[0]["K+"]
                real["K+"] = float(k_val) if pd.notna(k_val) else None
            if "Na+" in crf_row.columns:
                na_val = crf_row.iloc[0]["Na+"]
                real["Na+"] = float(na_val) if pd.notna(na_val) else None

        try:
            red_sig, ir_sig = self._load_spectrum_signals(file_path)
        except Exception as e:
            return {"code": 500, "msg": f"文件读取失败: {e}", "data": None}

        feat = self._extract_all_features(red_sig, ir_sig)
        if feat is None:
            return {"code": 500, "msg": "特征提取失败（信号过短或无效）", "data": None}

        X_test = pd.DataFrame([feat])[self.feature_names].fillna(0)

        quality_score = feat.get('quality_total_score', 0.5)

        # --- 心率 ---
        hr_pred, hr_method = None, "unknown"
        if quality_score < 0.3:
            hr_pred = 0
            hr_method = "rejected_low_quality"
        else:
            use_ml = True
            if self.outlier_detector is not None:
                if self.outlier_detector.predict(X_test)[0] == -1:
                    use_ml = False
            if use_ml:
                ml_hr = self.model_hr.predict(X_test)[0]
                ml_hr = np.clip(ml_hr, *self.HR_RANGE)
            else:
                ml_hr = None
            fused_hr, fuse_method = self._robust_hr_estimate(feat)
            if use_ml and ml_hr is not None:
                if abs(ml_hr - fused_hr) < 15:
                    hr_pred = (ml_hr + fused_hr) / 2.0
                    hr_method = "ML_fused"
                else:
                    hr_pred = ml_hr
                    hr_method = "ML_priority"
            else:
                hr_pred = fused_hr
                hr_method = "fused_only"
            hr_pred = int(np.clip(hr_pred, *self.HR_RANGE))

        # --- 血氧 ---
        R_val = feat.get('R_value', 0.5)
        classic_spo2 = np.clip(110 - 25 * R_val, *self.SPO2_RANGE)
        model_spo2 = self.model_spo2.predict(X_test)[0]
        if abs(model_spo2 - classic_spo2) > 0.1 * classic_spo2:
            pred_spo2 = int(classic_spo2)
            method_spo2 = "classic"
        else:
            pred_spo2 = int(np.clip(model_spo2, *self.SPO2_RANGE))
            method_spo2 = "ML"
        if quality_score < 0.3:
            pred_spo2 = 0
            method_spo2 = "rejected_low_quality"

        # FO2Hb
        pred_fo2hb = int(np.clip(self.model_fo2hb.predict(X_test)[0], *self.SPO2_RANGE))
        if quality_score < 0.3:
            pred_fo2hb = 0

        # 血糖
        pred_glu = None
        if self.model_glu is not None:
            pred_glu = round(float(self.model_glu.predict(X_test)[0]), 1)

        # K+
        pred_k = None
        if self.model_k is not None:
            pred_k = round(float(self.model_k.predict(X_test)[0]), 2)

        # Na+
        pred_na = None
        if self.model_na is not None:
            pred_na = round(float(self.model_na.predict(X_test)[0]), 1)

        # 误差计算
        hr_error = abs(hr_pred - real["心率"]) if hr_pred > 0 else None
        spo2_error = abs(pred_spo2 - real["血氧"]) if pred_spo2 > 0 else None
        fo2_error = abs(pred_fo2hb - real["FO2Hb"]) if pred_fo2hb > 0 else None
        glu_error = abs(pred_glu - real["血糖"]) if pred_glu is not None and real["血糖"] is not None else None
        k_error = abs(pred_k - real["K+"]) if pred_k is not None and real["K+"] is not None else None
        na_error = abs(pred_na - real["Na+"]) if pred_na is not None and real["Na+"] is not None else None

        red_acdc = self._extract_acdc(red_sig)
        ir_acdc = self._extract_acdc(ir_sig)
        R = R_val if (not np.isnan(red_acdc) and not np.isnan(ir_acdc)) else 0.5

        result = {
            "患者ID": patient_id, "统一ID": uid, "光谱文件": os.path.basename(file_path),
            "真实心率": real["心率"], "预测心率": hr_pred,
            "心率偏差": hr_error if hr_error is not None else "N/A",
            "心率方法": hr_method, "信号质量": round(quality_score, 2),
            "真实血氧": real["血氧"], "预测血氧": pred_spo2,
            "血氧偏差": spo2_error if spo2_error is not None else "N/A",
            "血氧方法": method_spo2,
            "真实FO2Hb": real["FO2Hb"], "预测FO2Hb": pred_fo2hb,
            "FO2Hb偏差": fo2_error if fo2_error is not None else "N/A",
            "真实血糖": real["血糖"] if real["血糖"] is not None else "",
            "预测血糖": pred_glu if pred_glu is not None else "",
            "血糖偏差": glu_error if glu_error is not None else "",
            "真实K+": real["K+"] if real["K+"] is not None else "",
            "预测K+": pred_k if pred_k is not None else "",
            "K+偏差": k_error if k_error is not None else "",
            "真实Na+": real["Na+"] if real["Na+"] is not None else "",
            "预测Na+": pred_na if pred_na is not None else "",
            "Na+偏差": na_error if na_error is not None else "",
            "红光AC/DC": round(red_acdc, 8) if not np.isnan(red_acdc) else 0,
            "红外光AC/DC": round(ir_acdc, 8) if not np.isnan(ir_acdc) else 0,
            "R值": round(R, 4), "特征数": len(self.feature_names)
        }
        return {"code": 200, "msg": "预测成功", "data": result}

    # ===================== 测试集评估 =====================
    def evaluate_test_set(self, output_csv="test_predictions_improved.csv") -> pd.DataFrame:
        if not self.test_ids:
            print("⚠️ 测试集为空")
            return pd.DataFrame()
        if self.model_hr is None:
            print("⚠️ 模型未加载，请先训练")
            return pd.DataFrame()

        records = []
        for pid in self.test_ids:
            res = self.predict_by_patient_id(pid)
            if res["code"] == 200:
                records.append(res["data"])
            else:
                print(f"跳过 {pid}: {res['msg']}")

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ 测试结果已保存至: {os.path.abspath(output_csv)}")
        return df


if __name__ == "__main__":
    # 请根据实际路径修改
    SPECTRUM_DIR = r"D:\科研项目资料\血氧检测项目\数据"
    CRF_PATH = r"D:\科研项目资料\血氧检测项目\2026无创血红蛋白监测CRF表.xlsx"

    predictor = ImprovedPhysioPredictor(
        spectrum_dir=SPECTRUM_DIR,
        crf_path=CRF_PATH,
        train_ratio=0.8,
        random_state=42,
        debug=True
    )

    # 训练并自动打包为 models.pkl
    predictor.train_ml_models()
    # 加载模型包
    predictor._load_models()

    # 评估并输出
    df = predictor.evaluate_test_set("test_results_improved_final.csv")

    if not df.empty:
        df_valid = df[df["预测心率"] > 0]
        print("\n" + "=" * 60)
        print("测试集评估摘要（单文件模型包）")
        print(f"总样本数: {len(df)}, 有效预测: {len(df_valid)}")
        if len(df_valid) > 0:
            print(f"心率 MAE: {df_valid['心率偏差'].mean():.1f} bpm")
            print(f"血氧 MAE: {df_valid['血氧偏差'].mean():.1f} %")
            print(f"FO2Hb MAE: {df_valid['FO2Hb偏差'].mean():.1f} %")
            if "血糖偏差" in df_valid.columns and df_valid["血糖偏差"].dtype != object:
                print(f"血糖 MAE: {df_valid['血糖偏差'].mean():.1f} mmol/L")
            if "K+偏差" in df_valid.columns and df_valid["K+偏差"].dtype != object:
                print(f"K+ MAE: {df_valid['K+偏差'].mean():.2f} mmol/L")
            if "Na+偏差" in df_valid.columns and df_valid["Na+偏差"].dtype != object:
                print(f"Na+ MAE: {df_valid['Na+偏差'].mean():.2f} mmol/L")
            good_hr = (df_valid['心率偏差'] < 10).sum()
            print(f"心率偏差<10的样本: {good_hr}/{len(df_valid)} ({100 * good_hr / len(df_valid):.1f}%)")
        print("=" * 60)