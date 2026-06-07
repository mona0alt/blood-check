import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "blood-check-android" / "app" / "src" / "main" / "assets" / "origin_model"


def scalar(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def tree_to_json(tree):
    t = tree.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value.reshape((t.node_count, -1))[:, 0].tolist(),
    }


def scaler_to_json(scaler):
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def label_encoder_to_json(encoder):
    return [str(x) for x in encoder.classes_.tolist()]


def lightgbm_to_json(model):
    return {
        "format": "lightgbm_json",
        "dump": model.booster_.dump_model(),
    }


def gradient_boosting_to_json(model):
    init = model.init_
    init_value = 0.0
    if hasattr(init, "constant_"):
        init_value = float(np.asarray(init.constant_).reshape(-1)[0])
    return {
        "format": "gradient_boosting_json",
        "learning_rate": float(model.learning_rate),
        "init": init_value,
        "estimators": [tree_to_json(est[0]) for est in model.estimators_],
    }


def ridge_to_json(model):
    return {
        "format": "ridge_json",
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
    }


def xgboost_to_json(model, output_path):
    model.save_model(str(output_path))
    return {
        "format": "xgboost_json",
        "file": output_path.name,
    }


def export_base_models():
    package = joblib.load(ROOT / "receive_data" / "models.pkl")
    models = {
        "hr": package["model_hr"],
        "spo2": package["model_spo2"],
        "fo2hb": package["model_fo2hb"],
        "glucose": package["model_glu"],
        "k": package["model_k"],
        "na": package["model_na"],
    }
    return {
        "version": 1,
        "feature_names": package["feature_names"],
        "models": {name: lightgbm_to_json(model) for name, model in models.items()},
        "outlier_detector": isolation_forest_to_json(package.get("outlier_detector")),
    }


def isolation_forest_to_json(model):
    if model is None:
        return None
    return {
        "max_samples": int(model.max_samples_),
        "offset": float(model.offset_),
        "estimators": [
            {
                "features": features.tolist(),
                "tree": tree_to_json(estimator),
                "n_node_samples": estimator.tree_.n_node_samples.tolist(),
            }
            for estimator, features in zip(model.estimators_, model.estimators_features_)
        ],
    }


def export_multi_models():
    package = joblib.load(ROOT / "receive_data" / "多任务模型.pkl")
    hb_path = OUT / "origin_multitask_hb_model.json"
    models = {
        "hemoglobin": xgboost_to_json(package["models"]["hemoglobin"], hb_path),
        "pi": gradient_boosting_to_json(package["models"]["pi"]),
        "lactate": ridge_to_json(package["models"]["lactate"]),
        "glucose": gradient_boosting_to_json(package["models"]["glucose"]),
        "k": ridge_to_json(package["models"]["k"]),
        "na": ridge_to_json(package["models"]["na"]),
    }
    return {
        "version": 1,
        "models": models,
        "scalers": {key: scaler_to_json(value) for key, value in package["scalers"].items()},
        "label_encoders": {
            key: label_encoder_to_json(value)
            for key, value in package.get("label_encoders", {}).items()
        },
        "feature_sets": package["feature_sets"],
        "pi_transform": package.get("pi_transform", "log"),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "origin_base_models.json").write_text(
        json.dumps(export_base_models(), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (OUT / "origin_multitask_models.json").write_text(
        json.dumps(export_multi_models(), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Exported Android origin model bundle to {OUT}")


if __name__ == "__main__":
    main()
