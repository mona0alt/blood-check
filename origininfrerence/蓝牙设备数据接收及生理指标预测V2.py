import asyncio
import csv
import logging
import os
import re
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError
from scipy import stats

# 预测相关库
import joblib
import pywt
import warnings
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt, hilbert, correlate
from scipy.stats import skew, kurtosis
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ---------- 配置 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_DEVICE_NAME = "Nordic_UART"
SAVE_DIR = os.path.join(SCRIPT_DIR, "receive_data")
RAW_FILE_NAME = "nordic_data.txt"
CSV_FILE_NAME = "spectral_data.csv"
BASE_CSV_FILE_NAME = "base_data.csv"
MULTI_TASK_MODEL_NAME = "多任务模型.pkl"
PATIENT_INFO_FILE = "huanzhe_inf.csv"

BASE_CSV_HEADER = ['Time', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
PATIENT_INFO_HEADER = ['Time', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

PREDICTION_INTERVAL = 5
MIN_DATA_ROWS = 200

NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===================== 多任务模型所需的特征提取类 =====================
class EnhancedDataLoader:
    """用于从原始光谱信号提取特征（与训练时完全一致）"""

    @staticmethod
    def normalize_spectrum_signal(signal_data):
        if len(signal_data) == 0:
            return np.array([5.0])
        signal_data = np.array(signal_data).astype(float)
        if len(signal_data) > 10:
            z_scores = np.abs(stats.zscore(signal_data, nan_policy='omit'))
            if len(z_scores) == len(signal_data):
                signal_clean = signal_data[z_scores < 3]
                if len(signal_clean) > 0:
                    signal_data = signal_clean
        if len(signal_data) == 0:
            return np.array([5.0])
        signal_min, signal_max = np.min(signal_data), np.max(signal_data)
        if signal_max - signal_min < 1e-10:
            return np.ones_like(signal_data) * 5.0
        normalized = (signal_data - signal_min) / (signal_max - signal_min) * 10.0
        return np.clip(normalized, 0, 12)

    @staticmethod
    def robust_ratio_calculation(red_signal, ir_signal):
        red_signal, ir_signal = np.array(red_signal, dtype=float), np.array(ir_signal, dtype=float)
        min_length = min(len(red_signal), len(ir_signal))
        if min_length == 0:
            return np.ones(1) * 1.2
        red_signal, ir_signal = red_signal[:min_length], ir_signal[:min_length]
        mask = ~(np.isnan(red_signal) | np.isnan(ir_signal) | np.isinf(red_signal) | np.isinf(ir_signal) | (
                    ir_signal == 0))
        if np.sum(mask) < 10:
            return np.ones_like(red_signal) * 1.2
        red_clean, ir_clean = red_signal[mask], ir_signal[mask] + 1e-10
        ratio = red_clean / ir_clean
        result = np.ones_like(red_signal) * 1.2
        if len(mask) == len(result):
            temp_result = np.ones_like(red_signal) * 1.2
            if np.sum(mask) == len(ratio):
                temp_result[mask] = ratio
            else:
                temp_result[mask] = ratio[:np.sum(mask)] if len(ratio) >= np.sum(mask) else np.concatenate(
                    [ratio, np.ones(np.sum(mask) - len(ratio)) * 1.2])
            result = temp_result
        return result

    @staticmethod
    def validate_and_align_signals(red_signal, ir_signal, filename=""):
        if len(red_signal) != len(ir_signal):
            min_len = min(len(red_signal), len(ir_signal))
            if min_len >= 10:
                red_signal, ir_signal = red_signal[:min_len], ir_signal[:min_len]
            else:
                return None, None
        return red_signal, ir_signal

    @staticmethod
    def calculate_signal_quality_score(red_signal, ir_signal):
        try:
            red_snr = np.mean(red_signal) / (np.std(red_signal) + 1e-10)
            ir_snr = np.mean(ir_signal) / (np.std(ir_signal) + 1e-10)
            snr = (red_snr + ir_snr) / 2
            snr_score = max(0, min(10, (snr - 1) / (7 - 1) * 10)) if snr > 1 else 0
            red_peaks, _ = find_peaks(red_signal, distance=20, prominence=0.1 * np.std(red_signal))
            ir_peaks, _ = find_peaks(ir_signal, distance=20, prominence=0.1 * np.std(ir_signal))
            pulse_count = min(len(red_peaks), len(ir_peaks))
            pulse_score = max(0, min(10, (pulse_count - 3) / (15 - 3) * 10)) if pulse_count >= 3 else 0
            signal_length = len(red_signal)
            length_score = max(0, min(10, (signal_length - 100) / (1000 - 100) * 10)) if signal_length >= 100 else 0
            total_score = snr_score * 0.4 + pulse_score * 0.3 + length_score * 0.3
            return total_score, snr_score, pulse_score, length_score, pulse_count
        except:
            return 0, 0, 0, 0, 0

    @staticmethod
    def repair_low_quality_signal(signal_data, score, pulse_count, signal_length):
        repaired_signal = signal_data.copy()
        try:
            if signal_length < 500 and len(signal_data) > 10:
                from scipy.interpolate import interp1d
                x_original = np.linspace(0, 1, len(signal_data))
                x_target = np.linspace(0, 1, 500)
                repaired_signal = interp1d(x_original, signal_data, kind='linear', fill_value="extrapolate")(x_target)
                signal_length = 500
            if pulse_count < 5 and len(repaired_signal) > 20:
                peaks, _ = find_peaks(repaired_signal, distance=15, prominence=0.05 * np.std(repaired_signal))
                if len(peaks) > pulse_count:
                    pulse_count = len(peaks)
            if score < 3 and len(repaired_signal) >= 64:
                try:
                    coeffs = pywt.wavedec(repaired_signal, 'db4', level=3)
                    threshold = 0.1 * np.std(repaired_signal)
                    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                    repaired_signal = pywt.waverec(coeffs_thresh, 'db4')
                except:
                    window_size = min(5, len(repaired_signal) // 10)
                    if window_size >= 3:
                        repaired_signal = np.convolve(repaired_signal, np.ones(window_size) / window_size, mode='same')
            return repaired_signal, pulse_count, signal_length
        except:
            return signal_data, pulse_count, len(signal_data)

    def process_spectrum_signals(self, red_signal_original, ir_signal_original, filename=""):
        features = {}
        red_signal_original, ir_signal_original = self.validate_and_align_signals(red_signal_original,
                                                                                  ir_signal_original, filename)
        if red_signal_original is None or len(red_signal_original) < 10:
            return None

        red_norm = self.normalize_spectrum_signal(red_signal_original)
        ir_norm = self.normalize_spectrum_signal(ir_signal_original)
        quality_score, snr_score, pulse_score, length_score, pulse_count = self.calculate_signal_quality_score(red_norm,
                                                                                                               ir_norm)
        features['信号质量评分'] = quality_score
        features['信噪比评分'] = snr_score
        features['脉搏数评分'] = pulse_score
        features['信号长度评分'] = length_score

        if quality_score < 3:
            red_norm, _, _ = self.repair_low_quality_signal(red_norm, quality_score, pulse_count, len(red_norm))
            ir_norm, _, _ = self.repair_low_quality_signal(ir_norm, quality_score, pulse_count, len(ir_norm))
            quality_score, snr_score, pulse_score, length_score, pulse_count = self.calculate_signal_quality_score(
                red_norm, ir_norm)
            features['信号质量评分_修复后'] = quality_score
            features['信号修复标记'] = 1
        elif quality_score < 6:
            window_size = min(5, len(red_norm) // 10)
            if window_size >= 3:
                red_norm = np.convolve(red_norm, np.ones(window_size) / window_size, mode='same')
                ir_norm = np.convolve(ir_norm, np.ones(window_size) / window_size, mode='same')
            features['信号修复标记'] = 0
        else:
            features['信号修复标记'] = 0

        ratio = self.robust_ratio_calculation(red_norm, ir_norm)
        ratio[ratio <= 0] = 0.1
        ratio = np.clip(ratio, 0.3, 3.0)
        features['R值_mean'] = np.mean(ratio)
        features['R值_std'] = np.std(ratio)
        features['R值_min'] = np.min(ratio)
        features['R值_max'] = np.max(ratio)
        rv = features['R值_mean']
        features['R值异常标记'] = 1 if (rv < 0.5 or rv > 2.0) else 0
        features['R值极端异常标记'] = 1 if (rv < 0.3 or rv > 3.0) else 0

        features['红光_mean'] = np.mean(red_norm)
        features['红光_std'] = np.std(red_norm)
        features['红外光_mean'] = np.mean(ir_norm)
        features['红外光_std'] = np.std(ir_norm)

        def extract_ac_dc(sig):
            window = min(100, len(sig) // 10) or 3
            dc = np.convolve(sig, np.ones(window) / window, mode='same')
            ac = sig - dc
            return np.mean(np.abs(ac)) / (np.mean(np.abs(dc)) + 1e-10)

        red_acdc = extract_ac_dc(red_norm)
        ir_acdc = extract_ac_dc(ir_norm)
        features['红光_AC_DC_ratio'] = red_acdc
        features['红外光_AC_DC_ratio'] = ir_acdc
        features['AC_DC_ratio_diff'] = red_acdc - ir_acdc
        pi_estimate = (red_acdc + ir_acdc) / 2 * 100
        features['PI_估算'] = pi_estimate

        if len(red_norm) >= 64:
            from scipy import signal
            freqs, psd = signal.welch(red_norm, fs=100, nperseg=min(256, len(red_norm)))
            total = np.sum(psd) + 1e-10
            features['红光_脉搏功率比'] = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)]) / total
            features['红光_主导频率'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
        else:
            features['红光_脉搏功率比'] = 0.1
            features['红光_主导频率'] = 1.0

        peaks, _ = find_peaks(red_norm, distance=20, prominence=0.1 * np.std(red_norm))
        if len(peaks) >= 2:
            intervals = np.diff(peaks)
            amplitudes = red_norm[peaks]
            features['红光_pulse_count'] = len(peaks)
            features['红光_interval_mean'] = np.mean(intervals)
            features['红光_amp_mean'] = np.mean(amplitudes)
        else:
            features['红光_pulse_count'] = len(peaks)
            features['红光_interval_mean'] = 0
            features['红光_amp_mean'] = 0

        features['信号信噪比'] = np.mean(red_norm) / (np.std(red_norm) + 1e-10)
        features['信号长度'] = len(red_norm)
        features['PI_ac_dc_ratio'] = pi_estimate / 100
        features['PI_variability'] = np.std(red_norm) / (np.std(ir_norm) + 1e-10)
        return features


class EnhancedFeatureEngineer:
    """创建高级特征（与训练时一致）"""

    def create_advanced_features(self, df):
        df_enhanced = df.copy()
        if 'PI_估算' in df.columns:
            df_enhanced['PI_log'] = np.log1p(df_enhanced['PI_估算'].clip(lower=0.01))
            df_enhanced['PI_平方'] = df_enhanced['PI_估算'] ** 2
        if '血氧饱和度' in df.columns and 'PI_估算' in df.columns:
            df_enhanced['PI_血氧交互'] = df_enhanced['PI_估算'] * df['血氧饱和度']
        if '血氧饱和度' in df.columns and 'R值_mean' in df_enhanced.columns:
            df_enhanced['血氧_R交互'] = df['血氧饱和度'] * df_enhanced['R值_mean']
        if '氧合血红蛋白分数(FO2Hb)' in df.columns and 'R值_mean' in df_enhanced.columns:
            df_enhanced['FO2Hb_R交互'] = df['氧合血红蛋白分数(FO2Hb)'] * df_enhanced['R值_mean']
        if '血糖' in df.columns and '血氧饱和度' in df.columns:
            df_enhanced['血糖_血氧交互'] = df['血糖'] * df['血氧饱和度']
        if 'K+' in df.columns and 'Na+' in df.columns:
            df_enhanced['K_Na比'] = df['K+'] / (df['Na+'] + 1e-10)

        for col in ['血氧饱和度', '氧合血红蛋白分数(FO2Hb)', '血糖', 'K+', 'Na+',
                    'R值_mean', '红光_mean', '红外光_mean', '信号信噪比']:
            if col in df_enhanced.columns and df_enhanced[col].min() > 0:
                df_enhanced[f'{col}_log'] = np.log1p(df_enhanced[col])
                df_enhanced[f'{col}_sqrt'] = np.sqrt(df_enhanced[col])

        if '信号信噪比' in df.columns and '红光_pulse_count' in df.columns:
            df_enhanced['信号质量指数'] = df['信号信噪比'] * np.log1p(df['红光_pulse_count'] + 1)

        if '信号质量评分' in df.columns:
            df_enhanced['信号质量等级'] = pd.cut(df['信号质量评分'], bins=[0, 3, 6, 10],
                                                 labels=['差', '中', '优']).astype(str)

        return df_enhanced


# ===================== 多任务模型预测器 =====================
class MultiTaskPredictor:
    def __init__(self, model_path: str):
        logger.info(f"正在加载多任务模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"多任务模型文件不存在: {model_path}")
        self.model_data = joblib.load(model_path)
        self.models = self.model_data['models']
        self.scalers = self.model_data['scalers']
        self.label_encoders = self.model_data.get('label_encoders', {})
        self.feature_sets = self.model_data['feature_sets']
        self.pi_transform = self.model_data.get('pi_transform', 'log')
        self.loader = EnhancedDataLoader()
        self.engineer = EnhancedFeatureEngineer()
        logger.info(f"多任务模型加载成功，可用任务: {list(self.models.keys())}")
        for task, feats in self.feature_sets.items():
            logger.debug(f"任务 {task} 特征数量: {len(feats)}")

    def _fill_missing_label_dependent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为推理阶段补充缺失的、依赖真实标签的特征（如血红蛋白区间、乳酸_log等）。
        使用合理的默认值代替真实值。
        """
        df_filled = df.copy()
        # 如果缺失 '血红蛋白' 列，用默认值 120
        if '血红蛋白' not in df_filled.columns:
            df_filled['血红蛋白'] = 120.0
        # 如果缺失 '血红蛋白区间' 和 '血红蛋白区间_num'，基于默认血红蛋白生成
        if '血红蛋白区间' not in df_filled.columns:
            hb_val = df_filled['血红蛋白'].values[0]
            if hb_val < 80:
                interval = '极重度贫血'
                num = 0
            elif hb_val < 120:
                interval = '贫血'
                num = 1
            elif hb_val <= 160:
                interval = '正常'
                num = 2
            elif hb_val <= 170:
                interval = '偏高'
                num = 3
            else:
                interval = '重度偏高'
                num = 4
            df_filled['血红蛋白区间'] = interval
            df_filled['血红蛋白区间_num'] = num
        # 如果缺失 '乳酸' 列，用默认值 2.0
        if '乳酸' not in df_filled.columns:
            df_filled['乳酸'] = 2.0
        # 如果缺失 '乳酸_log' 和 '乳酸_平方根'
        if '乳酸_log' not in df_filled.columns:
            lac_val = df_filled['乳酸'].values[0]
            df_filled['乳酸_log'] = np.log1p(max(lac_val, 0.01))
        if '乳酸_平方根' not in df_filled.columns:
            lac_val = df_filled['乳酸'].values[0]
            df_filled['乳酸_平方根'] = np.sqrt(max(lac_val, 0) + 1)
        return df_filled

    def _encode_categorical_features(self, df: pd.DataFrame, task: str) -> pd.DataFrame:
        """对DataFrame中的字符串列应用训练时保存的LabelEncoder"""
        df_enc = df.copy()
        for col in df.columns:
            is_text_column = (
                pd.api.types.is_object_dtype(df_enc[col].dtype)
                or pd.api.types.is_string_dtype(df_enc[col].dtype)
            )
            if is_text_column:  # 字符串类型
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    try:
                        df_enc[col] = le.transform(df_enc[col].astype(str))
                    except ValueError as e:
                        logger.warning(f"列 {col} 出现未知类别，用0填充。错误: {e}")
                        df_enc[col] = 0
                else:
                    # 手动映射常见分类
                    if col == '信号质量等级':
                        mapping = {'差': 0, '中': 1, '优': 2}
                        df_enc[col] = df_enc[col].map(mapping).fillna(0).astype(int)
                    elif col == '血红蛋白区间':
                        mapping = {'极重度贫血': 0, '贫血': 1, '正常': 2, '偏高': 3, '重度偏高': 4}
                        df_enc[col] = df_enc[col].map(mapping).fillna(2).astype(int)
                    else:
                        logger.warning(f"列 {col} 是字符串但无编码器，将转换为0")
                        df_enc[col] = 0
            # 确保数值类型
            df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce').fillna(0)
        return df_enc

    def predict(self, red_sig: np.ndarray, ir_sig: np.ndarray, clinical: Dict[str, float]) -> Dict[str, float]:
        logger.debug(f"开始多任务预测，信号长度 red={len(red_sig)}, ir={len(ir_sig)}")
        # 1. 提取光谱特征
        spec_features = self.loader.process_spectrum_signals(red_sig, ir_sig)
        if spec_features is None:
            logger.error("光谱特征提取失败 -> spec_features is None")
            raise ValueError("光谱特征提取失败")
        logger.debug(f"光谱特征提取成功，包含键: {list(spec_features.keys())[:10]}...")

        # 2. 合并临床特征，构建一行DataFrame
        row = {**clinical, **spec_features}
        df = pd.DataFrame([row])
        logger.debug(f"合并后DataFrame列数: {len(df.columns)}")

        # 3. 创建高级特征
        df = self.engineer.create_advanced_features(df)
        logger.debug(f"高级特征创建完成，列数: {len(df.columns)}")

        # 3.5 补充依赖目标变量的缺失特征
        df = self._fill_missing_label_dependent_features(df)
        logger.debug(f"补充标签相关特征后，列数: {len(df.columns)}")

        # 4. 分别预测三个指标
        results = {}
        for task in ['hemoglobin', 'pi', 'lactate']:
            if task not in self.models:
                logger.warning(f"任务 {task} 不在模型中，跳过")
                continue
            feat_list = self.feature_sets.get(task, [])
            if not feat_list:
                if hasattr(self.scalers[task], 'feature_names_in_'):
                    feat_list = list(self.scalers[task].feature_names_in_)
                else:
                    raise ValueError(f"任务 {task} 无特征列表")
            logger.debug(f"任务 {task} 所需特征数: {len(feat_list)}")

            # 构造特征矩阵，缺失列补0
            X_raw = pd.DataFrame(index=[0])
            missing_feats = []
            for f in feat_list:
                if f in df.columns:
                    X_raw[f] = df[f].values[0]
                else:
                    X_raw[f] = 0.0
                    missing_feats.append(f)
            if missing_feats:
                logger.warning(f"任务 {task} 缺失特征: {missing_feats[:5]}... 已用0填充")

            # 对字符串列进行编码
            X_encoded = self._encode_categorical_features(X_raw, task)

            # 标准化
            X_scaled = self.scalers[task].transform(X_encoded)
            # 预测
            pred = self.models[task].predict(X_scaled)[0]
            logger.debug(f"任务 {task} 原始预测值: {pred}")
            if task == 'pi' and self.pi_transform == 'log':
                pred = np.expm1(pred)
                logger.debug(f"PI 经 expm1 变换后: {pred}")
            # 对乳酸预测值进行后处理：负数置为0
            if task == 'lactate' and pred < 0:
                logger.warning(f"乳酸预测值为负({pred:.2f})，已调整为0")
                pred = 0.0
            results[task] = round(float(pred), 2)

        output = {
            '血红蛋白': results.get('hemoglobin', None),
            'PI': results.get('pi', None),
            '乳酸': results.get('lactate', None)
        }
        logger.info(f"多任务预测结果: {output}")
        return output


# ===================== 原有生理指标预测类（推理模式） =====================
class ImprovedPhysioPredictor:
    """增强型生理指标预测器（推理模式）"""

    def __init__(self, model_dir: str = ".", debug: bool = False):
        self.MODEL_DIR = model_dir
        self.MODEL_PACKAGE_PATH = os.path.join(self.MODEL_DIR, "models.pkl")
        self.DEBUG = debug

        self.CHANNEL_CONFIG = {"red": "C4", "ir": "C5", "fs": 100}
        self.HR_RANGE = (40, 200)
        self.SPO2_RANGE = (70, 100)
        self.R_RANGE = (0.3, 1.2)

        self.model_hr = None
        self.model_spo2 = None
        self.model_fo2hb = None
        self.model_glu = None
        self.model_k = None
        self.model_na = None
        self.outlier_detector = None
        self.feature_names = None

    @staticmethod
    def _bandpass_filter(signal, fs=100, low=0.8, high=3.0, order=4):
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

    def _load_models(self):
        if not os.path.exists(self.MODEL_PACKAGE_PATH):
            raise FileNotFoundError(f"模型包未找到: {self.MODEL_PACKAGE_PATH}")
        package = joblib.load(self.MODEL_PACKAGE_PATH)
        self.model_hr = package['model_hr']
        self.model_spo2 = package['model_spo2']
        self.model_fo2hb = package['model_fo2hb']
        self.model_glu = package.get('model_glu')
        self.model_k = package.get('model_k')
        self.model_na = package.get('model_na')
        self.outlier_detector = package.get('outlier_detector')
        self.feature_names = package['feature_names']
        logger.info(f"✅ 已加载模型包（来自 {self.MODEL_PACKAGE_PATH}）")

    def predict_from_signals(self, red_sig: np.ndarray, ir_sig: np.ndarray) -> dict:
        if self.model_hr is None:
            return {"code": 500, "msg": "模型未加载", "data": None}

        feat = self._extract_all_features(red_sig, ir_sig)
        if feat is None:
            return {"code": 500, "msg": "特征提取失败", "data": None}

        X_test = pd.DataFrame([feat])[self.feature_names].fillna(0)
        quality_score = feat.get('quality_total_score', 0.5)

        # 心率
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

        # 血氧
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

        result = {
            "预测心率": hr_pred,
            "预测血氧": pred_spo2,
            "预测FO2Hb": pred_fo2hb,
            "预测血糖": pred_glu,
            "预测K+": pred_k,
            "预测Na+": pred_na,
            "信号质量": round(quality_score, 2)
        }
        return {"code": 200, "msg": "预测成功", "data": result}


# ===================== 文件与数据工具 =====================
def ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def init_csv_file(csv_path, header):
    ensure_save_dir()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        if first_line == ','.join(header):
            return
        backup_name = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        os.rename(csv_path, backup_name)
        logger.info(f"旧 CSV 已备份: {backup_name}")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    logger.info(f"创建新 CSV: {csv_path}")


def parse_array_from_text(text_str):
    match = re.search(r'array:\s*\[([^\]]+)\]', text_str)
    if not match:
        return None
    content = match.group(1)
    numbers = re.findall(r'-?\d+', content)
    if len(numbers) == 5:
        try:
            return [int(x) for x in numbers]
        except ValueError:
            return None
    return None


# ===================== 蓝牙主流程 =====================
async def check_bluetooth_enabled():
    try:
        scanner = BleakScanner()
        await scanner.discover(timeout=0.5, return_adv=True)
        logger.info("蓝牙功能正常")
        return True
    except BleakError as e:
        logger.error(f"蓝牙未开启或不可用: {e}")
        return False
    except Exception as e:
        logger.error(f"蓝牙检测失败: {e}")
        return False


async def prediction_loop(csv_file_path, base_csv_path, patient_info_path, predictor, multitask_predictor, stop_event):
    logger.info("定时预测任务已启动（使用全部累积数据）")
    while not stop_event.is_set():
        try:
            if not os.path.exists(csv_file_path):
                await asyncio.sleep(PREDICTION_INTERVAL)
                continue

            df = None
            for attempt in range(3):
                try:
                    df = pd.read_csv(csv_file_path)
                    break
                except Exception as e:
                    logger.warning(f"读取 CSV 失败 (尝试 {attempt + 1}/3): {e}")
                    await asyncio.sleep(0.2)
            if df is None:
                logger.warning("无法读取光谱数据，跳过本轮预测")
                await asyncio.sleep(PREDICTION_INTERVAL)
                continue

            total_rows = len(df)
            if total_rows < MIN_DATA_ROWS:
                logger.debug(f"数据不足 {MIN_DATA_ROWS} 行，当前 {total_rows} 行，跳过预测")
                await asyncio.sleep(PREDICTION_INTERVAL)
                continue

            red_sig = df["C4"].values.astype(float)
            ir_sig = df["C5"].values.astype(float)
            logger.debug(f"使用全部 {total_rows} 行数据进行预测")

            # ---- 基础预测 ----
            res = predictor.predict_from_signals(red_sig, ir_sig)
            if res["code"] != 200:
                logger.warning(f"基础预测失败: {res['msg']}")
                await asyncio.sleep(PREDICTION_INTERVAL)
                continue

            data = res["data"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # 写入 base_data.csv
            base_row = [
                timestamp,
                data['预测血氧'] if data['预测血氧'] != 0 else '',
                data['预测心率'] if data['预测心率'] != 0 else '',
                data['预测FO2Hb'] if data['预测FO2Hb'] != 0 else '',
                data['预测血糖'] if data['预测血糖'] is not None else '',
                data['预测K+'] if data['预测K+'] is not None else '',
                data['预测Na+'] if data['预测Na+'] is not None else ''
            ]
            with open(base_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(base_row)
            logger.info(f"基础预测结果已写入: {dict(zip(BASE_CSV_HEADER, base_row))}")

            # ---- 多任务预测 ----
            clinical_dict = {
                '血氧饱和度': data['预测血氧'],
                '氧合血红蛋白分数(FO2Hb)': data['预测FO2Hb'],
                '血糖': data['预测血糖'] if data['预测血糖'] is not None else 5.5,
                'K+': data['预测K+'] if data['预测K+'] is not None else 4.0,
                'Na+': data['预测Na+'] if data['预测Na+'] is not None else 140.0
            }
            logger.info(f"多任务预测临床输入: {clinical_dict}")
            logger.info(f"光谱信号长度: red={len(red_sig)}, ir={len(ir_sig)}")

            try:
                multi_result = multitask_predictor.predict(red_sig, ir_sig, clinical_dict)
                logger.info(f"多任务预测原始结果: {multi_result}")
                hb = multi_result.get('血红蛋白', '')
                pi = multi_result.get('PI', '')
                lac = multi_result.get('乳酸', '')
                logger.info(f"解析后 -> Hb={hb}, PI={pi}, 乳酸={lac}")
            except Exception as e:
                import traceback
                logger.error(f"多任务预测失败: {e}\n{traceback.format_exc()}")
                hb, pi, lac = '', '', ''

            patient_row = [
                timestamp,
                data['预测血氧'] if data['预测血氧'] != 0 else '',
                data['预测心率'] if data['预测心率'] != 0 else '',
                data['预测FO2Hb'] if data['预测FO2Hb'] != 0 else '',
                data['预测血糖'] if data['预测血糖'] is not None else '',
                data['预测K+'] if data['预测K+'] is not None else '',
                data['预测Na+'] if data['预测Na+'] is not None else '',
                hb,
                pi,
                lac
            ]
            with open(patient_info_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(patient_row)
            logger.info(
                f"患者综合信息已写入: 血氧={patient_row[1]}, 心率={patient_row[2]}, FO2Hb={patient_row[3]}, 血糖={patient_row[4]}, K+={patient_row[5]}, Na+={patient_row[6]}, Hb={hb}, PI={pi}, 乳酸={lac}")

        except Exception as e:
            logger.error(f"预测循环异常: {e}")
        await asyncio.sleep(PREDICTION_INTERVAL)

    logger.info("定时预测任务已停止")


async def scan_and_connect():
    logger.info(f"开始扫描设备: {TARGET_DEVICE_NAME}")
    device = None
    scanner = BleakScanner()

    while device is None:
        try:
            devices = await scanner.discover(timeout=2.0, return_adv=True)
        except BleakError as e:
            logger.error(f"扫描错误: {e}")
            return
        except Exception as e:
            logger.error(f"扫描异常: {e}")
            return

        for dev, adv_data in devices.values():
            if dev.name == TARGET_DEVICE_NAME:
                logger.info(f"发现目标设备: {dev.name} [{dev.address}] RSSI={adv_data.rssi}")
                device = dev
                break

        if device is not None:
            break
        logger.info("未找到设备，继续扫描...")

    logger.info(f"正在连接: {device.address}")
    try:
        async with BleakClient(device) as client:
            logger.info("连接成功，正在等待服务就绪...")
            await asyncio.sleep(1)

            uart_service = client.services.get_service(NUS_SERVICE_UUID)
            if uart_service is None:
                logger.error("未找到 NUS 服务")
                return
            tx_char = uart_service.get_characteristic(NUS_TX_CHAR_UUID)
            if tx_char is None:
                logger.error("TX 特征未找到")
                return

            ensure_save_dir()
            raw_file_path = os.path.join(SAVE_DIR, RAW_FILE_NAME)
            csv_file_path = os.path.join(SAVE_DIR, CSV_FILE_NAME)
            base_csv_path = os.path.join(SAVE_DIR, BASE_CSV_FILE_NAME)
            patient_info_path = os.path.join(SAVE_DIR, PATIENT_INFO_FILE)

            init_csv_file(csv_file_path, ['Time', 'C1', 'C2', 'C3', 'C4', 'C5'])
            init_csv_file(base_csv_path, BASE_CSV_HEADER)
            init_csv_file(patient_info_path, PATIENT_INFO_HEADER)

            # 加载基础预测模型
            predictor = ImprovedPhysioPredictor(model_dir=SAVE_DIR, debug=True)
            try:
                predictor._load_models()
            except FileNotFoundError as e:
                logger.error(f"基础模型加载失败: {e}")
                return

            # 加载多任务模型
            multitask_model_path = os.path.join(SAVE_DIR, MULTI_TASK_MODEL_NAME)
            logger.info(f"多任务模型路径: {multitask_model_path}")
            if not os.path.exists(multitask_model_path):
                logger.error(f"多任务模型文件不存在: {multitask_model_path}")
                return
            else:
                logger.info(f"找到多任务模型文件，大小={os.path.getsize(multitask_model_path)} bytes")
            try:
                multitask_predictor = MultiTaskPredictor(multitask_model_path)
            except Exception as e:
                logger.error(f"多任务模型加载失败: {e}")
                return

            def notification_handler(sender, data: bytearray):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                hex_str = data.hex()
                try:
                    text_str = data.decode("utf-8").strip()
                except UnicodeDecodeError:
                    text_str = "<binary>"

                raw_line = f"{timestamp} | hex: {hex_str} | text: {text_str}\n"
                try:
                    with open(raw_file_path, "a", encoding="utf-8") as f:
                        f.write(raw_line)
                except Exception as e:
                    logger.error(f"写入原始文件失败: {e}")

                array_data = parse_array_from_text(text_str)
                if array_data is not None:
                    try:
                        with open(csv_file_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp] + array_data)
                        logger.info(f"写入CSV: {timestamp}, {array_data}")
                    except Exception as e:
                        logger.error(f"写入CSV失败: {e}")
                else:
                    logger.warning(f"无法解析: {text_str}")

                logger.info(f"收到数据: {text_str}")

            await client.start_notify(tx_char, notification_handler)
            logger.info("已启用通知，开始实时接收数据")

            stop_event = asyncio.Event()
            pred_task = asyncio.create_task(prediction_loop(csv_file_path, base_csv_path, patient_info_path,
                                                            predictor, multitask_predictor, stop_event))

            logger.info("按 Ctrl+C 停止接收...")
            try:
                while client.is_connected:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("用户中断接收")
            finally:
                stop_event.set()
                pred_task.cancel()
                try:
                    await pred_task
                except asyncio.CancelledError:
                    pass
                await client.stop_notify(tx_char)
                logger.info("通知已停止")
    except BleakError as e:
        logger.error(f"连接错误: {e}")
    except Exception as e:
        logger.error(f"其它错误: {e}")


async def main():
    logger.info("===== BLE 生理监测系统启动 =====")
    if not await check_bluetooth_enabled():
        logger.error("蓝牙未开启，请打开后重试")
        return
    await scan_and_connect()
    logger.info("程序结束")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序已退出")
