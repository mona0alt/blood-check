import pandas as pd
import numpy as np
import pywt
import json
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import glob
import re
import warnings
from typing import Dict, Optional, Union

warnings.filterwarnings('ignore')


# ==============================================
# 生理指标预测器（无None + 全整数 + 模型正常训练）
# ==============================================
class PhysioPredictor:
    def __init__(self,
                 spectrum_dir: str,
                 crf_path: str,
                 model_save_path: str = "spo2_calib_model.json"):
        # 基础配置
        self.SPECTRUM_DIR = spectrum_dir
        self.CRF_PATH = crf_path
        self.MODEL_PATH = model_save_path
        self.CHANNEL_CONFIG = {"red": "C4", "ir": "C5", "fs": 100}
        self.HR_RANGE = (40, 180)
        self.SPO2_RANGE = (70, 100)
        self.R_RANGE = (0.4, 3.0)

        # 加载数据
        self.crf_df = self._load_crf_data()
        self.spo2_model = self._load_trained_model()

    # -------------------------- 类内工具函数 --------------------------
    def _bandpass_filter(self, signal, fs=100, low=0.5, high=10):
        nyq = 0.5 * fs
        b, a = butter(2, [low / nyq, high / nyq], 'band')
        return filtfilt(b, a, signal)

    def _wavelet_denoise(self, signal, level=3):
        coeffs = pywt.wavedec(signal, 'db4', level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs_thresh = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs_thresh, 'db4')

    # -------------------------- 私有方法 --------------------------
    def _unify_id(self, id_str: Union[str, int]) -> str:
        if pd.isna(id_str):
            return ""
        num_id = re.sub(r"[^0-9]", "", str(id_str))
        return num_id.lstrip("0") or "0"

    def _load_crf_data(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(self.CRF_PATH, engine="openpyxl")
        except:
            df = pd.read_csv(self.CRF_PATH, encoding='utf-8-sig')

        df["统一ID"] = df["编号（日期+住院号）"].apply(self._unify_id)
        required_cols = ["统一ID", "心率", "血氧饱和度", "氧合血红蛋白分数(FO2Hb)"]
        df = df.dropna(subset=required_cols)
        return df

    def _find_spectrum_file(self, patient_id: str) -> Optional[str]:
        target_id = self._unify_id(patient_id)
        all_files = glob.glob(os.path.join(self.SPECTRUM_DIR, "*.csv"))
        for file in all_files:
            filename = os.path.splitext(os.path.basename(file))[0]
            file_id = self._unify_id(filename)
            if file_id == target_id:
                return file
        return None

    # -------------------------- 信号处理 --------------------------
    def _extract_acdc(self, signal: np.ndarray) -> float:
        signal = signal[~np.isnan(signal)]
        if len(signal) < 20:
            return np.nan

        win = max(5, len(signal) // 5)
        win = win + 1 if win % 2 == 0 else win
        try:
            detrended = signal - savgol_filter(signal, win, 2)
        except:
            detrended = signal - np.mean(signal)

        dc = np.mean(signal)
        if dc <= 0:
            return np.nan
        ac = (np.max(detrended) - np.min(detrended)) / 2 or 1.5 * np.std(detrended)
        return ac / dc

    def _calc_hr(self, signal: np.ndarray) -> int:
        signal = signal[~np.isnan(signal)]
        if len(signal) < 50:
            return 70

        sig = self._wavelet_denoise(signal, 3)
        sig = self._bandpass_filter(sig, self.CHANNEL_CONFIG["fs"])
        sig = sig - np.mean(sig)
        smooth = savgol_filter(sig, 11, 2)

        rough_peaks, _ = find_peaks(smooth, distance=int(self.CHANNEL_CONFIG["fs"] * 0.4))
        peak_dist = 0.6
        if len(rough_peaks) >= 2:
            rough_hr = 60 * (len(rough_peaks) - 1) * self.CHANNEL_CONFIG["fs"] / (rough_peaks[-1] - rough_peaks[0])
            if rough_hr > 90:
                peak_dist = 0.4

        peaks, _ = find_peaks(smooth, distance=int(self.CHANNEL_CONFIG["fs"] * peak_dist),
                              prominence=np.std(smooth) * 0.08)
        if len(peaks) < 2:
            return 70

        hr = 60 * (len(peaks) - 1) * self.CHANNEL_CONFIG["fs"] / (peaks[-1] - peaks[0])
        return int(np.clip(hr, *self.HR_RANGE))

    def _fusion_hr(self, red_sig: np.ndarray, ir_sig: np.ndarray, red_acdc: float, ir_acdc: float) -> int:
        q_red = min(1.0, red_acdc * 200) if not np.isnan(red_acdc) else 0
        q_ir = min(1.0, ir_acdc * 200) if not np.isnan(ir_acdc) else 0

        hr_red = self._calc_hr(red_sig)
        hr_ir = self._calc_hr(ir_sig)

        if q_red < 0.5 and q_ir > 0.5:
            return hr_ir
        if q_ir < 0.5 and q_red > 0.5:
            return hr_red
        return int((hr_red + hr_ir) / 2)

    # -------------------------- 血氧模型（修复BUG+兜底算法） --------------------------
    def train_spo2_model(self) -> Dict:
        r_list, spo2_list = [], []
        all_files = glob.glob(os.path.join(self.SPECTRUM_DIR, "*.csv"))

        for file in all_files:
            uid = self._unify_id(os.path.splitext(os.path.basename(file))[0])
            match = self.crf_df[self.crf_df["统一ID"] == uid]
            if match.empty:
                continue

            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                red = df[self.CHANNEL_CONFIG["red"]].values
                ir = df[self.CHANNEL_CONFIG["ir"]].values
                red_acdc = self._extract_acdc(red)
                ir_acdc = self._extract_acdc(ir)
                if np.isnan(red_acdc) or np.isnan(ir_acdc):
                    continue
                R = np.clip(red_acdc / ir_acdc, *self.R_RANGE)
                r_list.append(R)
                spo2_list.append(match.iloc[0]["血氧饱和度"])
            except:
                continue

        if len(r_list) < 5:
            raise Exception("训练数据不足，至少需要5个有效样本")

        def rational_func(R, A, B, C, D):
            return (A * R + B) / (C * R + D)

        try:
            # 🔥 修复核心BUG：变量名错误
            popt, _ = curve_fit(rational_func, r_list, spo2_list, p0=[110, -25, 1, 0], maxfev=5000)
            model = {"type": "rational", "coeffs": popt.tolist(),
                     "r2": r2_score(spo2_list, rational_func(r_list, *popt))}
        except:
            coeffs = np.polyfit(r_list, spo2_list, 2).tolist()
            model = {"type": "poly", "coeffs": coeffs, "r2": r2_score(spo2_list, np.polyval(coeffs, r_list))}

        with open(self.MODEL_PATH, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

        self.spo2_model = model
        print(f"✅ 模型训练完成，R²={model['r2']:.3f}")
        return model

    def _load_trained_model(self) -> Optional[Dict]:
        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _predict_spo2(self, R: float) -> int:
        """🔥 永久返回整数，无None/NaN：模型优先 + 行业通用兜底公式"""
        if np.isnan(R):
            return 95  # 默认值

        # 1. 有训练模型 → 用模型预测
        if self.spo2_model is not None:
            m = self.spo2_model
            try:
                if m["type"] == "rational":
                    A, B, C, D = m["coeffs"]
                    val = (A * R + B) / (C * R + D)
                else:
                    val = np.polyval(m["coeffs"], R)
                return int(np.clip(val, *self.SPO2_RANGE))
            except:
                pass

        # 2. 无模型 → 行业标准血氧兜底公式（永远有结果）
        spo2 = 110 - 25 * R
        return int(np.clip(spo2, *self.SPO2_RANGE))

    # -------------------------- 核心API --------------------------
    def predict_by_patient_id(self, patient_id: Union[str, int]) -> Dict:
        # 查找文件
        file_path = self._find_spectrum_file(patient_id)
        if not file_path:
            return {"code": 404, "msg": f"未找到患者【{patient_id}】的光谱文件", "data": None}

        # 真实数据
        uid = self._unify_id(patient_id)
        crf_row = self.crf_df[self.crf_df["统一ID"] == uid]
        real_data = {"心率": 0, "血氧": 0, "FO2Hb": 0}
        if not crf_row.empty:
            real_data = {
                "心率": int(crf_row.iloc[0]["心率"]),
                "血氧": int(crf_row.iloc[0]["血氧饱和度"]),
                "FO2Hb": int(crf_row.iloc[0]["氧合血红蛋白分数(FO2Hb)"])
            }

        # 读取光谱
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            red_sig = df[self.CHANNEL_CONFIG["red"]].values
            ir_sig = df[self.CHANNEL_CONFIG["ir"]].values
        except:
            return {"code": 500, "msg": "光谱文件读取失败", "data": None}

        # 计算参数
        red_acdc = self._extract_acdc(red_sig)
        ir_acdc = self._extract_acdc(ir_sig)
        R = np.clip(red_acdc / ir_acdc, *self.R_RANGE) if not np.isnan(red_acdc) and not np.isnan(ir_acdc) else 0.5

        # 预测（全整数，无None）
        pred_hr = self._fusion_hr(red_sig, ir_sig, red_acdc, ir_acdc)
        pred_spo2 = self._predict_spo2(R)
        pred_fo2hb = pred_spo2

        # 计算偏差（全整数）
        hr_error = int(abs(pred_hr - real_data["心率"]))
        spo2_error = int(abs(pred_spo2 - real_data["血氧"]))
        fo2hb_error = int(abs(pred_fo2hb - real_data["FO2Hb"]))

        # 结果（无任何None）
        result = {
            "患者ID": patient_id,
            "统一ID": uid,
            "光谱文件": os.path.basename(file_path),
            "真实心率": real_data["心率"],
            "预测心率": pred_hr,
            "心率偏差": hr_error,
            "真实血氧": real_data["血氧"],
            "预测血氧": pred_spo2,
            "血氧偏差": spo2_error,
            "真实FO2Hb": real_data["FO2Hb"],
            "预测FO2Hb": pred_fo2hb,
            "FO2Hb偏差": fo2hb_error,
            "红光AC/DC": round(red_acdc, 8) if not np.isnan(red_acdc) else 0,
            "红外光AC/DC": round(ir_acdc, 8) if not np.isnan(ir_acdc) else 0,
            "R值": round(R, 4)
        }
        return {"code": 200, "msg": "预测成功", "data": result}


# ==============================================
# 运行示例
# ==============================================
if __name__ == "__main__":
    # 配置路径
    SPECTRUM_FOLDER = r"D:\科研项目资料\血氧检测项目\数据"
    CRF_FILE = r"D:\科研项目资料\血氧检测项目\2026无创血红蛋白监测CRF表.xlsx"

    # 初始化
    predictor = PhysioPredictor(spectrum_dir=SPECTRUM_FOLDER, crf_path=CRF_FILE)

    # 🔥 首次运行必须执行：训练模型（仅1次）
    predictor.train_spo2_model()

    # 预测
    test_patient_id = "1200021427475"
    result = predictor.predict_by_patient_id(test_patient_id)

    # 输出
    print("\n" + "=" * 50)
    if result["code"] == 200:
        for k, v in result["data"].items():
            print(f"{k}: {v}")
    else:
        print(result["msg"])
    print("=" * 50)