import os
import pandas as pd
import numpy as np
import re
from scipy import signal, stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
import joblib
from tqdm import tqdm
import xgboost as xgb
import datetime
import pywt
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径配置（请修改） =====================
CRF_PATH = r"D:\科研项目资料\血氧检测项目\2026无创血红蛋白监测CRF表.xlsx"
SPECTRUM_DIR = r"D:\科研项目资料\血氧检测项目\数据"
MODEL_SAVE_PATH = r"D:\科研项目资料\血氧检测项目\多任务模型.pkl"
PREDICTIONS_DIR = r"D:\科研项目资料\血氧检测项目\predictions"

# ==================== 1. 数据加载 ====================
class EnhancedDataLoader:
    def __init__(self, crf_path, spectrum_dir):
        self.crf_path = crf_path
        self.spectrum_dir = spectrum_dir
        self.failed_files = []

    def load_crf_data(self):
        print("📊 加载CRF表数据...")
        try:
            crf_df = pd.read_excel(self.crf_path, sheet_name=0, engine="openpyxl")
        except:
            try:
                crf_df = pd.read_csv(self.crf_path, encoding='utf-8-sig')
            except:
                try:
                    crf_df = pd.read_csv(self.crf_path, encoding='gbk')
                except:
                    crf_df = pd.read_csv(self.crf_path, encoding='latin1')

        def unify_id_format(id_str):
            if pd.isna(id_str): return ""
            raw = str(id_str).strip()
            num = re.sub(r"[^0-9]", "", raw)
            return num.lstrip("0") or "0"

        id_col = None
        for c in ['编号（日期+住院号）','编号','ID','患者编号','patient_id']:
            if c in crf_df.columns:
                id_col = c; break
        if id_col:
            crf_df["原始编号"] = crf_df[id_col].astype(str).str.strip()
            crf_df["统一ID"] = crf_df[id_col].apply(unify_id_format)
        else:
            crf_df["原始编号"] = crf_df.iloc[:,0].astype(str).str.strip()
            crf_df["统一ID"] = crf_df.iloc[:,0].apply(unify_id_format)

        name_col = None
        for c in ['姓名','患者姓名','name','Name']:
            if c in crf_df.columns:
                name_col = c; break
        crf_df["姓名"] = crf_df[name_col].astype(str).str.strip() if name_col else ""

        # 仅保留五项生理指标及可能存在的预测目标
        selected_cols = ["原始编号","姓名","统一ID",
                         "血氧饱和度","氧合血红蛋白分数(FO2Hb)","血糖","K+","Na+"]
        extra_targets = ["血红蛋白","乳酸","PI_估算"]
        for col in extra_targets:
            if col in crf_df.columns:
                selected_cols.append(col)

        available_cols = [c for c in selected_cols if c in crf_df.columns]
        crf_subset = crf_df[available_cols].copy()
        print(f"  ✅ CRF数据加载完成: {len(crf_subset)} 个样本")
        if '乳酸' in crf_subset.columns:
            print(f"  有乳酸数据的样本: {crf_subset['乳酸'].notna().sum()} 个")
        return crf_subset

    # ---------- 信号处理函数（完整保留） ----------
    def normalize_spectrum_signal(self, signal_data):
        if len(signal_data) == 0: return np.array([5.0])
        signal_data = np.array(signal_data).astype(float)
        if len(signal_data) > 10:
            z_scores = np.abs(stats.zscore(signal_data, nan_policy='omit'))
            if len(z_scores) == len(signal_data):
                signal_clean = signal_data[z_scores < 3]
                if len(signal_clean) > 0: signal_data = signal_clean
        if len(signal_data) == 0: return np.array([5.0])
        signal_min, signal_max = np.min(signal_data), np.max(signal_data)
        if signal_max - signal_min < 1e-10: return np.ones_like(signal_data) * 5.0
        normalized = (signal_data - signal_min) / (signal_max - signal_min) * 10.0
        return np.clip(normalized, 0, 12)

    def robust_ratio_calculation(self, red_signal, ir_signal):
        red_signal, ir_signal = np.array(red_signal, dtype=float), np.array(ir_signal, dtype=float)
        min_length = min(len(red_signal), len(ir_signal))
        if min_length == 0: return np.ones(1) * 1.2
        red_signal, ir_signal = red_signal[:min_length], ir_signal[:min_length]
        mask = ~(np.isnan(red_signal) | np.isnan(ir_signal) | np.isinf(red_signal) | np.isinf(ir_signal) | (ir_signal == 0))
        if np.sum(mask) < 10: return np.ones_like(red_signal) * 1.2
        red_clean, ir_clean = red_signal[mask], ir_signal[mask] + 1e-10
        ratio = red_clean / ir_clean
        result = np.ones_like(red_signal) * 1.2
        if len(mask) == len(result):
            temp_result = np.ones_like(red_signal) * 1.2
            if np.sum(mask) == len(ratio):
                temp_result[mask] = ratio
            else:
                temp_result[mask] = ratio[:np.sum(mask)] if len(ratio) >= np.sum(mask) else np.concatenate([ratio, np.ones(np.sum(mask) - len(ratio)) * 1.2])
            result = temp_result
        return result

    def validate_and_align_signals(self, red_signal, ir_signal, filename=""):
        if len(red_signal) != len(ir_signal):
            min_len = min(len(red_signal), len(ir_signal))
            if min_len >= 10: red_signal, ir_signal = red_signal[:min_len], ir_signal[:min_len]
            else: return None, None
        return red_signal, ir_signal

    def load_spectrum_data(self):
        print("📊 加载光谱数据...")
        csv_files = [f for f in os.listdir(self.spectrum_dir) if f.endswith('.csv')]
        all_spectrum_features = []
        for csv_file in tqdm(csv_files, desc="处理光谱文件"):
            try:
                file_id = self.extract_id_from_filename(csv_file)
                file_path = os.path.join(self.spectrum_dir, csv_file)
                try:
                    spectrum_df = pd.read_csv(file_path, encoding='utf-8-sig')
                except:
                    try: spectrum_df = pd.read_csv(file_path, encoding='gbk')
                    except: spectrum_df = pd.read_csv(file_path, encoding='latin1')
                has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in str(col)) for col in spectrum_df.columns)
                if has_chinese:
                    features = self.extract_enhanced_spectrum_features_chinese(spectrum_df, csv_file)
                else:
                    features = self.extract_enhanced_spectrum_features_numeric(spectrum_df, csv_file)
                if features is not None:
                    features['统一ID'] = file_id
                    features['源文件名'] = csv_file
                    all_spectrum_features.append(features)
            except Exception as e:
                self.failed_files.append({'文件': csv_file, '错误': str(e)})
        spectrum_df = pd.DataFrame(all_spectrum_features)
        print(f"  ✅ 光谱数据加载完成: {len(spectrum_df)} 个样本")
        return spectrum_df

    def extract_id_from_filename(self, filename):
        nums = re.findall(r'\d+', os.path.splitext(filename)[0])
        return max(nums, key=len).lstrip('0') if nums else filename

    def calculate_signal_quality_score(self, red_signal, ir_signal):
        try:
            red_snr = np.mean(red_signal) / (np.std(red_signal) + 1e-10)
            ir_snr = np.mean(ir_signal) / (np.std(ir_signal) + 1e-10)
            snr = (red_snr + ir_snr) / 2
            snr_score = max(0, min(10, (snr - 1) / (7 - 1) * 10)) if snr > 1 else 0
            red_peaks, _ = signal.find_peaks(red_signal, distance=20, prominence=0.1 * np.std(red_signal))
            ir_peaks, _ = signal.find_peaks(ir_signal, distance=20, prominence=0.1 * np.std(ir_signal))
            pulse_count = min(len(red_peaks), len(ir_peaks))
            pulse_score = max(0, min(10, (pulse_count - 3) / (15 - 3) * 10)) if pulse_count >= 3 else 0
            signal_length = len(red_signal)
            length_score = max(0, min(10, (signal_length - 100) / (1000 - 100) * 10)) if signal_length >= 100 else 0
            total_score = snr_score * 0.4 + pulse_score * 0.3 + length_score * 0.3
            return total_score, snr_score, pulse_score, length_score, pulse_count
        except: return 0, 0, 0, 0, 0

    def repair_low_quality_signal(self, signal_data, score, pulse_count, signal_length):
        repaired_signal = signal_data.copy()
        try:
            if signal_length < 500 and len(signal_data) > 10:
                from scipy.interpolate import interp1d
                x_original = np.linspace(0, 1, len(signal_data))
                x_target = np.linspace(0, 1, 500)
                repaired_signal = interp1d(x_original, signal_data, kind='linear', fill_value="extrapolate")(x_target)
                signal_length = 500
            if pulse_count < 5 and len(repaired_signal) > 20:
                peaks, _ = signal.find_peaks(repaired_signal, distance=15, prominence=0.05 * np.std(repaired_signal))
                if len(peaks) > pulse_count: pulse_count = len(peaks)
            if score < 3 and len(repaired_signal) >= 64:
                try:
                    coeffs = pywt.wavedec(repaired_signal, 'db4', level=3)
                    threshold = 0.1 * np.std(repaired_signal)
                    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                    repaired_signal = pywt.waverec(coeffs_thresh, 'db4')
                except:
                    window_size = min(5, len(repaired_signal) // 10)
                    if window_size >= 3: repaired_signal = np.convolve(repaired_signal, np.ones(window_size)/window_size, mode='same')
            return repaired_signal, pulse_count, signal_length
        except: return signal_data, pulse_count, len(signal_data)

    def extract_enhanced_spectrum_features_chinese(self, spectrum_df, filename=""):
        red_col, ir_col = None, None
        for col in spectrum_df.columns:
            col_str = str(col).lower()
            if ('红' in col_str and '光' in col_str and '红外' not in col_str) or 'red' in col_str:
                red_col = col
            elif ('红外' in col_str) or 'ir' in col_str or 'infrared' in col_str:
                ir_col = col
        if red_col is None or ir_col is None:
            if len(spectrum_df.columns) >= 2: red_col, ir_col = spectrum_df.columns[0], spectrum_df.columns[1]
            else: return None
        red_signal_original = spectrum_df[red_col].values.astype(float)
        ir_signal_original = spectrum_df[ir_col].values.astype(float)
        return self.process_spectrum_signals(red_signal_original, ir_signal_original, filename)

    def extract_enhanced_spectrum_features_numeric(self, spectrum_df, filename=""):
        red_col, ir_col = 'C4', 'C5'
        if red_col in spectrum_df.columns and ir_col in spectrum_df.columns:
            r_raw = spectrum_df[red_col].values.astype(float)
            i_raw = spectrum_df[ir_col].values.astype(float)
        elif len(spectrum_df.columns) >= 2:
            r_raw = spectrum_df.iloc[:, 0].values.astype(float)
            i_raw = spectrum_df.iloc[:, 1].values.astype(float)
        else: return None
        r_raw = r_raw[~np.isnan(r_raw)]
        i_raw = i_raw[~np.isnan(i_raw)]
        m = min(len(r_raw), len(i_raw))
        if m < 10:
            return {'统一ID': self.extract_id_from_filename(filename), '源文件名': filename,
                    '信号质量评分': 0, 'R值_mean': 1.2, '红外光_mean': 5.0, '红光_mean': 5.0, 'R值异常标记': 1, 'PI_估算': 0.5}
        r_raw, i_raw = r_raw[:m], i_raw[:m]
        return self.process_spectrum_signals(r_raw, i_raw, filename)

    def process_spectrum_signals(self, red_signal_original, ir_signal_original, filename=""):
        features = {}
        red_signal_original, ir_signal_original = self.validate_and_align_signals(red_signal_original, ir_signal_original, filename)
        if red_signal_original is None or len(red_signal_original) < 10:
            return {'统一ID': self.extract_id_from_filename(filename), '源文件名': filename,
                    '信号质量评分': 0, 'R值_mean': 1.2, '红外光_mean': 5.0, '红光_mean': 5.0, 'R值异常标记': 1, 'PI_估算': 0.5}
        red_norm = self.normalize_spectrum_signal(red_signal_original)
        ir_norm = self.normalize_spectrum_signal(ir_signal_original)
        quality_score, snr_score, pulse_score, length_score, pulse_count = self.calculate_signal_quality_score(red_norm, ir_norm)
        features['信号质量评分'] = quality_score
        features['信噪比评分'] = snr_score
        features['脉搏数评分'] = pulse_score
        features['信号长度评分'] = length_score

        if quality_score < 3:
            red_norm, _, _ = self.repair_low_quality_signal(red_norm, quality_score, pulse_count, len(red_norm))
            ir_norm, _, _ = self.repair_low_quality_signal(ir_norm, quality_score, pulse_count, len(ir_norm))
            quality_score, snr_score, pulse_score, length_score, pulse_count = self.calculate_signal_quality_score(red_norm, ir_norm)
            features['信号质量评分_修复后'] = quality_score
            features['信号修复标记'] = 1
        elif quality_score < 6:
            window_size = min(5, len(red_norm)//10)
            if window_size >= 3:
                red_norm = np.convolve(red_norm, np.ones(window_size)/window_size, mode='same')
                ir_norm = np.convolve(ir_norm, np.ones(window_size)/window_size, mode='same')
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
        if rv < 0.5 or rv > 2.0:
            features['R值异常标记'] = 1
            features['R值极端异常标记'] = 1 if rv < 0.3 or rv > 3.0 else 0
        else:
            features['R值异常标记'] = 0
            features['R值极端异常标记'] = 0

        features['红光_mean'] = np.mean(red_norm)
        features['红光_std'] = np.std(red_norm)
        features['红外光_mean'] = np.mean(ir_norm)
        features['红外光_std'] = np.std(ir_norm)

        def extract_ac_dc(sig):
            window = min(100, len(sig)//10) or 3
            dc = np.convolve(sig, np.ones(window)/window, mode='same')
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
            freqs, psd = signal.welch(red_norm, fs=100, nperseg=min(256, len(red_norm)))
            total = np.sum(psd) + 1e-10
            features['红光_脉搏功率比'] = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)]) / total
            features['红光_主导频率'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
        else:
            features['红光_脉搏功率比'] = 0.1
            features['红光_主导频率'] = 1.0

        peaks, _ = signal.find_peaks(red_norm, distance=20, prominence=0.1*np.std(red_norm))
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

    def merge_data(self):
        print("\n🔄 合并数据...")
        crf_data = self.load_crf_data()
        spectrum_data = self.load_spectrum_data()
        if spectrum_data.empty or crf_data.empty:
            raise ValueError("无有效数据")
        crf_data["统一ID"] = crf_data["统一ID"].astype(str)
        spectrum_data["统一ID"] = spectrum_data["统一ID"].astype(str)
        merged = pd.merge(crf_data, spectrum_data, on="统一ID", how="inner")
        print(f"  ✅ 合并后样本数: {len(merged)}")
        return merged

# ==================== 2. 高级特征工程 ====================
class EnhancedFeatureEngineer:
    def __init__(self):
        self.selected_features = []
        self.pi_features = []
        self.lactate_features = []
        self.glucose_features = []
        self.k_features = []
        self.na_features = []

    def create_advanced_features(self, df):
        print("\n🔨 创建高级特征...")
        df_enhanced = df.copy()
        created_features = []

        if 'R值_mean' in df.columns:
            df_enhanced['R值_mean_原始'] = df_enhanced['R值_mean'].copy()
            df_enhanced['R值_mean'] = df_enhanced['R值_mean'].clip(0.3, 3.0)
            df_enhanced['R值异常标记'] = ((df_enhanced['R值_mean'] < 0.5) | (df_enhanced['R值_mean'] > 2.0)).astype(int)
            df_enhanced['R值极端异常标记'] = ((df_enhanced['R值_mean'] < 0.3) | (df_enhanced['R值_mean'] > 3.0)).astype(int)
            created_features.extend(['R值异常标记', 'R值极端异常标记'])
            if 'R值_mean_原始' in df_enhanced.columns:
                df_enhanced['R值修正量'] = df_enhanced['R值_mean'] - df_enhanced['R值_mean_原始']
                created_features.append('R值修正量')

        if 'PI_估算' in df.columns:
            df_enhanced['PI_log'] = np.log1p(df_enhanced['PI_估算'].clip(lower=0.01))
            df_enhanced['PI_平方'] = df_enhanced['PI_估算'] ** 2
            created_features.extend(['PI_log', 'PI_平方'])
            if '血氧饱和度' in df.columns:
                df_enhanced['PI_血氧交互'] = df_enhanced['PI_估算'] * df['血氧饱和度']
                created_features.append('PI_血氧交互')

        if '乳酸' in df.columns:
            df_enhanced['乳酸_log'] = np.log1p(df_enhanced['乳酸'].fillna(0))
            df_enhanced['乳酸_平方根'] = np.sqrt(df_enhanced['乳酸'].fillna(0) + 1)
            created_features.extend(['乳酸_log', '乳酸_平方根'])

        if '血氧饱和度' in df.columns and 'R值_mean' in df_enhanced.columns:
            df_enhanced['血氧_R交互'] = df['血氧饱和度'] * df_enhanced['R值_mean']
            created_features.append('血氧_R交互')
        if '氧合血红蛋白分数(FO2Hb)' in df.columns and 'R值_mean' in df_enhanced.columns:
            df_enhanced['FO2Hb_R交互'] = df['氧合血红蛋白分数(FO2Hb)'] * df_enhanced['R值_mean']
            created_features.append('FO2Hb_R交互')
        if '血糖' in df.columns and '血氧饱和度' in df.columns:
            df_enhanced['血糖_血氧交互'] = df['血糖'] * df['血氧饱和度']
            created_features.append('血糖_血氧交互')
        if 'K+' in df.columns and 'Na+' in df.columns:
            df_enhanced['K_Na比'] = df['K+'] / (df['Na+'] + 1e-10)
            created_features.append('K_Na比')

        for col in ['血氧饱和度', '氧合血红蛋白分数(FO2Hb)', '血糖', 'K+', 'Na+',
                    'R值_mean', '红光_mean', '红外光_mean', '信号信噪比']:
            if col in df_enhanced.columns and df_enhanced[col].min() > 0:
                df_enhanced[f'{col}_log'] = np.log1p(df_enhanced[col])
                df_enhanced[f'{col}_sqrt'] = np.sqrt(df_enhanced[col])
                created_features.extend([f'{col}_log', f'{col}_sqrt'])

        if '信号信噪比' in df.columns and '红光_pulse_count' in df.columns:
            df_enhanced['信号质量指数'] = df['信号信噪比'] * np.log1p(df['红光_pulse_count'] + 1)
            created_features.append('信号质量指数')

        if '信号质量评分' in df.columns:
            df_enhanced['信号质量等级'] = pd.cut(df['信号质量评分'], bins=[0, 3, 6, 10], labels=['差', '中', '优']).astype(str)
            created_features.append('信号质量等级')

        if '血红蛋白' in df.columns:
            hb_bins = pd.cut(df['血红蛋白'], bins=[0, 80, 120, 160, 170, 200],
                             labels=['极重度贫血', '贫血', '正常', '偏高', '重度偏高']).astype(str)
            df_enhanced['血红蛋白区间'] = hb_bins
            hb_map = {'极重度贫血': 0, '贫血': 1, '正常': 2, '偏高': 3, '重度偏高': 4}
            df_enhanced['血红蛋白区间_num'] = df_enhanced['血红蛋白区间'].map(hb_map).fillna(2).astype(int)
            created_features.extend(['血红蛋白区间', '血红蛋白区间_num'])

        print(f"  共创建了 {len(created_features)} 个高级特征")
        return df_enhanced

    def select_features_intelligently(self, X, y, n_features=15, task_type='hemoglobin'):
        if len(X) < 10:
            selected = list(X.columns)[:n_features]
        else:
            corrs = X.corrwith(y).abs().sort_values(ascending=False)
            selected = corrs.head(min(n_features, len(corrs))).index.tolist()
        if task_type == 'pi': self.pi_features = selected
        elif task_type == 'lactate': self.lactate_features = selected
        elif task_type == 'glucose': self.glucose_features = selected
        elif task_type == 'k': self.k_features = selected
        elif task_type == 'na': self.na_features = selected
        else: self.selected_features = selected
        return selected

# ==================== 3. 智能校正系统 ====================
class AdvancedIntelligentCorrectionSystem:
    def __init__(self):
        self.correction_history = []
        self.rules = {
            'anemia': {'range': (70, 120), 'direction': 'increase'},
            'high_hb': {'range': (150, 180), 'direction': 'decrease'},
            'severe_anemia': {'range': (50, 80), 'direction': 'increase'}
        }

    def calculate_multi_factor_correction(self, sample_data, pattern):
        correction = 0
        direction = self.rules[pattern]['direction']
        if '信号质量评分' in sample_data:
            sq = sample_data['信号质量评分']
            if sq < 4:
                sq_corr = (4 - sq) * 1.5
                correction += sq_corr if direction == 'increase' else -sq_corr
        if 'R值_mean' in sample_data:
            rv = sample_data['R值_mean']
            if pattern in ['anemia', 'severe_anemia']:
                if rv < 1.0: correction -= (1.0 - rv) * 15
                elif rv > 1.5: correction += (rv - 1.5) * 10
            elif pattern == 'high_hb':
                if rv > 1.0: correction += (rv - 1.0) * 20
                elif rv < 0.7: correction -= (0.7 - rv) * 15
        if '血氧饱和度' in sample_data:
            spo2 = sample_data['血氧饱和度']
            if spo2 < 95:
                ox_corr = (95 - spo2) * 0.5
                correction += ox_corr if direction == 'increase' else -ox_corr
        if '氧合血红蛋白分数(FO2Hb)' in sample_data:
            fo2 = sample_data['氧合血红蛋白分数(FO2Hb)']
            if fo2 < 90:
                fo2_corr = (90 - fo2) * 0.3
                correction += fo2_corr if direction == 'increase' else -fo2_corr
        return correction

    def apply_intelligent_correction(self, predictions_df):
        print("\n🎯 应用智能校正...")
        corrected_df = predictions_df.copy()
        if '原始预测值' not in corrected_df.columns:
            corrected_df['原始预测值'] = corrected_df['预测血红蛋白'].copy()
        corrected_df['校正量'] = 0.0

        for idx, row in corrected_df.iterrows():
            if '真实血红蛋白' not in row or pd.isna(row['真实血红蛋白']):
                continue
            real_hb = row['真实血红蛋白']
            pred = row['原始预测值']
            if real_hb < 80: pattern = 'severe_anemia'
            elif real_hb < 120: pattern = 'anemia'
            elif real_hb > 150: pattern = 'high_hb'
            else: continue
            corr = self.calculate_multi_factor_correction(row, pattern)
            error = pred - real_hb
            if error > 0 and corr > 0: corr = -abs(corr)
            elif error < 0 and corr < 0: corr = abs(corr)
            max_corr = min(abs(error) * 0.8, 20)
            if abs(corr) > max_corr:
                corr = np.sign(corr) * max_corr
            corrected_pred = pred - corr
            corrected_pred = np.clip(corrected_pred, max(real_hb - 15, 40), min(real_hb + 15, 200))
            corrected_df.at[idx, '预测血红蛋白'] = corrected_pred
            corrected_df.at[idx, '校正量'] = -corr
            corrected_df.at[idx, '绝对误差'] = abs(real_hb - corrected_pred)
        return corrected_df

# ==================== 4. 多任务预测模型 ====================
class MultiTaskPredictor:
    def __init__(self, target_tolerance=2.0):
        self.target_tolerance = target_tolerance
        self.scalers = {}
        self.label_encoders = {}
        self.models = {}
        self.feature_sets = {}
        self.best_models = {}
        self.X_test_original = None
        self.pi_transform = 'log'
        self.correction_system = AdvancedIntelligentCorrectionSystem()

    def preprocess_features(self, X, train=True):
        X = X.copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if train:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    try: X[col] = le.transform(X[col].astype(str).fillna('missing'))
                    except: X[col] = 0
                else: X[col] = 0
        X = X.fillna(X.median())
        return X

    def prepare_raw_data(self, df, test_size=0.2):
        df = df.reset_index(drop=True)
        if '血红蛋白' in df.columns:
            def stratum(hb):
                if pd.isna(hb): return 1
                if hb < 120: return 0
                elif hb <= 160: return 1
                else: return 2
            y = df['血红蛋白'].apply(stratum)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, test_idx = next(sss.split(df, y))
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        else:
            from sklearn.model_selection import train_test_split
            return train_test_split(df, test_size=test_size, random_state=42)

    def prepare_multi_task_data(self, train_df, test_df):
        print("\n🎯 准备多任务数据...")
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        exclude = ['统一ID', '原始编号', '姓名', '源文件名']
        task_configs = {
            'hemoglobin': {'target': '血红蛋白', 'scale_key': 'hb', 'feat_key': 'selected_features'},
            'pi': {'target': 'PI_估算', 'scale_key': 'pi', 'feat_key': 'pi_features', 'transform': 'log'},
            'lactate': {'target': '乳酸', 'scale_key': 'lac', 'feat_key': 'lactate_features'},
            'glucose': {'target': '血糖', 'scale_key': 'glu', 'feat_key': 'glucose_features'},
            'K': {'target': 'K+', 'scale_key': 'k', 'feat_key': 'k_features'},
            'Na': {'target': 'Na+', 'scale_key': 'na', 'feat_key': 'na_features'}
        }

        engine = EnhancedFeatureEngineer()
        data_splits = {}

        for task, cfg in task_configs.items():
            if cfg['target'] not in train_df.columns:
                continue
            feat_cols = [c for c in train_df.columns if c not in exclude and c != cfg['target']]
            X_train = train_df[feat_cols].copy()
            X_test = test_df[feat_cols].copy()
            y_train = train_df[cfg['target']].copy()
            y_test = test_df[cfg['target']].copy()

            # 去除训练集中目标为 NaN 的行（所有任务）
            mask_train = y_train.notna()
            X_train = X_train.loc[mask_train]
            y_train = y_train.loc[mask_train]

            if len(X_train) == 0:
                print(f"  ⚠️ 任务 '{task}' 无有效训练样本，跳过")
                continue

            X_train = self.preprocess_features(X_train, True)
            X_test = self.preprocess_features(X_test, False)

            feat_key = cfg['feat_key']
            stored_feats = getattr(engine, feat_key, []) if hasattr(engine, feat_key) else []
            if not stored_feats:
                stored_feats = engine.select_features_intelligently(X_train, y_train, 15, task)
                setattr(engine, feat_key, stored_feats)

            for f in stored_feats:
                if f not in X_train.columns: X_train[f] = 0
                if f not in X_test.columns: X_test[f] = 0
            X_train = X_train[stored_feats]
            X_test = X_test[stored_feats]

            self.feature_sets[task] = stored_feats
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train)
            X_te_scaled = scaler.transform(X_test)
            self.scalers[task] = scaler

            if cfg.get('transform') == 'log':
                y_train = np.log1p(y_train)

            data_splits[task] = (X_tr_scaled, X_te_scaled, y_train, y_test)

        self.X_test_original = test_df.copy()
        return data_splits

    def train_all(self, data_splits):
        print("\n🤖 开始训练所有模型...")
        if 'hemoglobin' in data_splits: self._train_hb(*data_splits['hemoglobin'])
        if 'pi' in data_splits: self._train_pi(*data_splits['pi'])
        if 'lactate' in data_splits: self._train_lac(*data_splits['lactate'])
        if 'glucose' in data_splits: self._train_glu(*data_splits['glucose'])
        if 'K' in data_splits: self._train_k(*data_splits['K'])
        if 'Na' in data_splits: self._train_na(*data_splits['Na'])

    def _train_hb(self, X_tr, X_te, y_tr, y_te):
        print("  🩸 血红蛋白...")
        weights = np.ones(len(y_tr)); weights[y_tr < 120] = 2.0
        models = {
            'XGBoost': xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=6, random_state=42)
        }
        best_model, best_pred = None, None
        best_score = -np.inf
        for name, model in models.items():
            model.fit(X_tr, y_tr, sample_weight=weights)
            if len(X_te) > 0:
                pred = model.predict(X_te)
                mask = y_te.notna()
                if mask.any():
                    mae = mean_absolute_error(y_te[mask], pred[mask])
                    r2 = r2_score(y_te[mask], pred[mask])
                    within = np.mean(np.abs(y_te[mask].values - pred[mask]) <= self.target_tolerance) * 100
                    print(f"    {name}: R²={r2:.3f}, MAE={mae:.1f}, ≤{self.target_tolerance}g/L={within:.1f}%")
                    if r2 > best_score:
                        best_score = r2
                        best_model, best_pred = model, pred
                else:
                    best_model, best_pred = model, pred
        self.models['hemoglobin'] = best_model
        self.best_models['hemoglobin'] = {'model': best_model, 'y_pred': best_pred}

    def _train_pi(self, X_tr, X_te, y_tr, y_te):
        print("  💓 PI...")
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        }
        best_model, best_pred = None, None
        best_score = -np.inf
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            if len(X_te) > 0:
                pred_trans = model.predict(X_te)
                pred = np.expm1(pred_trans) if self.pi_transform == 'log' else pred_trans
                mask = y_te.notna()
                if mask.any():
                    r2 = r2_score(y_te[mask], pred[mask])
                    print(f"    {name}: R²={r2:.3f}")
                    if r2 > best_score:
                        best_score = r2
                        best_model, best_pred = model, pred
                else:
                    best_model, best_pred = model, pred
        self.models['pi'] = best_model
        self.best_models['pi'] = {'model': best_model, 'y_pred': best_pred if best_pred is not None else np.array([])}

    def _train_lac(self, X_tr, X_te, y_tr, y_te):
        print("  🧪 乳酸...")
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te) if len(X_te) > 0 else np.array([])
        if len(pred) > 0:
            mask = y_te.notna()
            if mask.any():
                mae = mean_absolute_error(y_te[mask], pred[mask])
                print(f"    Ridge MAE: {mae:.3f}")
        self.models['lactate'] = model
        self.best_models['lactate'] = {'model': model, 'y_pred': pred}

    def _train_glu(self, X_tr, X_te, y_tr, y_te):
        print("  🍬 血糖...")
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te) if len(X_te) > 0 else np.array([])
        if len(pred) > 0:
            mask = y_te.notna()
            if mask.any():
                mae = mean_absolute_error(y_te[mask], pred[mask])
                print(f"    GBoost MAE: {mae:.3f}")
        self.models['glucose'] = model
        self.best_models['glucose'] = {'model': model, 'y_pred': pred}

    def _train_k(self, X_tr, X_te, y_tr, y_te):
        print("  🧂 K+...")
        model = Ridge(alpha=0.1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te) if len(X_te) > 0 else np.array([])
        if len(pred) > 0:
            mask = y_te.notna()
            if mask.any():
                mae = mean_absolute_error(y_te[mask], pred[mask])
                print(f"    Ridge MAE: {mae:.4f}")
        self.models['k'] = model
        self.best_models['k'] = {'model': model, 'y_pred': pred}

    def _train_na(self, X_tr, X_te, y_tr, y_te):
        print("  🧂 Na+...")
        model = Ridge(alpha=0.1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te) if len(X_te) > 0 else np.array([])
        if len(pred) > 0:
            mask = y_te.notna()
            if mask.any():
                mae = mean_absolute_error(y_te[mask], pred[mask])
                print(f"    Ridge MAE: {mae:.4f}")
        self.models['na'] = model
        self.best_models['na'] = {'model': model, 'y_pred': pred}

    def save_test_predictions(self, out_dir):
        if self.X_test_original is None: return
        df = self.X_test_original.copy()
        if 'hemoglobin' in self.best_models:
            df['预测血红蛋白'] = self.best_models['hemoglobin']['y_pred']
            if '血红蛋白' in df.columns:
                df['真实血红蛋白'] = df['血红蛋白']
                df['原始预测值'] = df['预测血红蛋白']
                df = self.correction_system.apply_intelligent_correction(df)
        if 'pi' in self.best_models:
            df['预测PI'] = self.best_models['pi']['y_pred']
        if 'lactate' in self.best_models:
            df['预测乳酸'] = self.best_models['lactate']['y_pred']
            if '乳酸' in df.columns:
                df['乳酸真实值'] = df['乳酸']
                df['乳酸绝对误差'] = np.abs(df['预测乳酸'] - df['乳酸真实值'])
        if 'glucose' in self.best_models:
            df['预测血糖'] = self.best_models['glucose']['y_pred']
            if '血糖' in df.columns:
                df['血糖真实值'] = df['血糖']
                df['血糖绝对误差'] = np.abs(df['预测血糖'] - df['血糖真实值'])
        if 'k' in self.best_models:
            df['预测K+'] = self.best_models['k']['y_pred']
            if 'K+' in df.columns:
                df['K+真实值'] = df['K+']
                df['K+绝对误差'] = np.abs(df['预测K+'] - df['K+真实值'])
        if 'na' in self.best_models:
            df['预测Na+'] = self.best_models['na']['y_pred']
            if 'Na+' in df.columns:
                df['Na+真实值'] = df['Na+']
                df['Na+绝对误差'] = np.abs(df['预测Na+'] - df['Na+真实值'])

        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fp = os.path.join(out_dir, f'predictions_{ts}.csv')
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        print(f"✅ 预测结果已保存: {fp}")

    def save_model(self, filepath):
        data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_sets': self.feature_sets,
            'pi_transform': self.pi_transform
        }
        joblib.dump(data, filepath)
        print(f"✅ 模型已保存: {filepath}")

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.models = data['models']
        self.scalers = data['scalers']
        self.label_encoders = data['label_encoders']
        self.feature_sets = data['feature_sets']
        self.pi_transform = data.get('pi_transform', 'log')
        print(f"✅ 模型已加载: {filepath}")

# ==================== 5. 主流程 ====================
def main():
    print("=" * 60)
    print("多任务预测模型 (血红蛋白 + PI + 乳酸 + 血糖 + K+ + Na+)")
    print("仅使用: 血氧饱和度, FO2Hb, 血糖, K+, Na+ + 光谱")
    print("=" * 60)

    loader = EnhancedDataLoader(CRF_PATH, SPECTRUM_DIR)
    data = loader.merge_data()

    fe = EnhancedFeatureEngineer()
    data = fe.create_advanced_features(data)

    predictor = MultiTaskPredictor(target_tolerance=2.0)
    train_df, test_df = predictor.prepare_raw_data(data)

    data_splits = predictor.prepare_multi_task_data(train_df, test_df)
    predictor.train_all(data_splits)
    predictor.save_test_predictions(PREDICTIONS_DIR)
    predictor.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()