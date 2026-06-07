import json
import logging
import os
import re

import numpy as np
import pandas as pd

from feature_engineer import CompleteFeatureEngineer
from model_predictor import CompleteModelPredictor
from spectrum_processor import CompleteSpectrumProcessor


logger = logging.getLogger(__name__)

_predictor = None
_processor = None
_feature_engineer = None


def _to_json_compatible(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _get_predictor(data_dir):
    global _predictor
    if _predictor is None:
        _predictor = CompleteModelPredictor(
            os.path.join(data_dir, "model", "android_model_bundle.json"),
            os.path.join(data_dir, "model", "hb_model.json"),
        )
    return _predictor


def _get_processor():
    global _processor
    if _processor is None:
        _processor = CompleteSpectrumProcessor()
    return _processor


def _get_feature_engineer():
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = CompleteFeatureEngineer()
    return _feature_engineer


def _unify_id_format(id_str):
    if pd.isna(id_str):
        return ""
    unified = re.sub(r"[^0-9]", "", str(id_str).strip()).lstrip("0")
    return unified if unified else "0"


def _extract_patient_clinical_features(patient_id, crf_path):
    try:
        crf_df = pd.read_csv(crf_path, encoding="utf-8")
    except Exception:
        try:
            crf_df = pd.read_csv(crf_path, encoding="utf-8-sig")
        except Exception:
            crf_df = pd.read_csv(crf_path, encoding="gbk")

    id_column = None
    for column in ["编号（日期+住院号）", "编号", "ID", "患者编号", "patient_id", "编号(日期+住院号)", "住院号"]:
        if column in crf_df.columns:
            id_column = column
            break
    if id_column is None:
        id_column = crf_df.columns[0]

    crf_df["统一ID"] = crf_df[id_column].apply(_unify_id_format)
    patient_row = crf_df[crf_df["统一ID"] == patient_id]
    if len(patient_row) == 0:
        patient_row = crf_df[crf_df["统一ID"] == patient_id.lstrip("0")]
    if len(patient_row) == 0:
        return None, None

    row = patient_row.iloc[0]
    required_features = {
        "性别": "男",
        "年龄": 50,
        "血氧饱和度": 98.0,
        "氧分压": 95.0,
        "氧合血红蛋白分数(FO2Hb)": 96.0,
        "心率": 75,
        "体温": 36.5,
        "是否房颤": "否",
        "血红蛋白": np.nan,
        "乳酸": np.nan,
    }
    clinical = {}
    for feature, default in required_features.items():
        clinical[feature] = row[feature] if feature in row.index and not pd.isna(row[feature]) else default

    patient_info = {}
    for field in ["编号", "姓名", "性别", "年龄", "心率", "血氧饱和度", "氧分压", "氧合血红蛋白分数(FO2Hb)", "体温"]:
        if field == "编号":
            patient_info[field] = patient_id
            continue
        for column in crf_df.columns:
            if field in str(column):
                value = row[column]
                patient_info[field] = _to_json_compatible(value)
                break

    return clinical, patient_info


def _find_spectrum_file(patient_id, spectrum_dir):
    for extension in [".csv", ".CSV", ".txt", ".TXT"]:
        for name in [patient_id, patient_id.lstrip("0")]:
            candidate = os.path.join(spectrum_dir, f"{name}{extension}")
            if os.path.exists(candidate):
                return candidate

    try:
        for filename in os.listdir(spectrum_dir):
            if filename.lower().endswith(".csv") and (patient_id in filename or patient_id.lstrip("0") in filename):
                return os.path.join(spectrum_dir, filename)
    except Exception:
        pass
    return None


def _read_spectrum_file(filepath):
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(filepath, encoding="gbk")
        except Exception:
            df = pd.read_csv(filepath, encoding="latin1")

    if "C4" in df.columns and "C5" in df.columns:
        return df["C4"].values.astype(float), df["C5"].values.astype(float)

    has_chinese = any(any("\u4e00" <= char <= "\u9fff" for char in str(column)) for column in df.columns)
    if has_chinese:
        red_col = None
        ir_col = None
        for column in df.columns:
            column_str = str(column).lower()
            if ("红" in column_str and "光" in column_str and "红外" not in column_str) or "red" in column_str:
                red_col = column
            elif "红外" in column_str or "ir" in column_str or "infrared" in column_str:
                ir_col = column
        if red_col and ir_col:
            return df[red_col].values.astype(float), df[ir_col].values.astype(float)

    return df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)


def predict_by_patient_id(patient_id, data_dir):
    try:
        patient_id = str(patient_id).strip()
        extracted = _extract_patient_clinical_features(patient_id, os.path.join(data_dir, "baseinfo", "CRF_info.csv"))
        if extracted is None or extracted[0] is None:
            return json.dumps(
                {"success": False, "patient_info": None, "prediction": None, "error": "未找到该患者信息，请检查ID"},
                ensure_ascii=False,
            )

        clinical_features, patient_info = extracted
        spectrum_file = _find_spectrum_file(patient_id, os.path.join(data_dir, "spectral_data"))
        processor = _get_processor()
        if spectrum_file:
            red_signal, ir_signal = _read_spectrum_file(spectrum_file)
            spectrum_features = processor.process_spectrum_signals(red_signal, ir_signal, os.path.basename(spectrum_file))
        else:
            spectrum_features = processor.get_default_spectrum_features(patient_id, f"{patient_id}.csv")

        all_features = {**clinical_features, **spectrum_features, "统一ID": patient_id, "源文件名": f"{patient_id}.csv"}
        all_features.setdefault("乳酸", np.nan)
        all_features.setdefault("血红蛋白", np.nan)

        predictor = _get_predictor(data_dir)
        results_df, json_str = predictor.predict_single_patient(
            _get_feature_engineer().create_advanced_features(all_features),
            patient_id,
        )
        if results_df is None:
            return json.dumps(
                {"success": False, "patient_info": patient_info, "prediction": None, "error": "模型预测失败"},
                ensure_ascii=False,
            )

        prediction_list = json.loads(json_str)
        return json.dumps(
            {
                "success": True,
                "patient_info": patient_info,
                "prediction": prediction_list[0] if prediction_list else None,
                "error": None,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        return json.dumps(
            {
                "success": False,
                "patient_info": None,
                "prediction": None,
                "error": f"预测过程出错: {str(exc)}",
            },
            ensure_ascii=False,
        )
