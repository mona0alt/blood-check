import json
import os
import traceback
from datetime import datetime
from math import isnan

import numpy as np
import pandas as pd


class JsonLabelEncoder:
    def __init__(self, classes):
        self.mapping = {str(value): index for index, value in enumerate(classes)}

    def transform(self, values):
        return np.asarray([self.mapping.get(str(value), 0) for value in values], dtype=np.int64)


class StandardScalerLite:
    def __init__(self, scaler_data):
        self.mean = np.asarray(scaler_data["mean"], dtype=np.float64)
        self.scale = np.asarray(scaler_data["scale"], dtype=np.float64)

    def transform(self, values):
        array = values.values if hasattr(values, "values") else np.asarray(values)
        array = np.asarray(array, dtype=np.float64)
        return (array - self.mean) / self.scale


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
        outputs = []
        for row in array:
            prediction = self.base_score
            for tree in self.trees:
                prediction += self._predict_tree(tree, row)
            outputs.append(prediction)
        return np.asarray(outputs)

    def _predict_tree(self, tree, row):
        left_children = tree["left_children"]
        right_children = tree["right_children"]
        default_left = tree["default_left"]
        split_indices = tree["split_indices"]
        split_conditions = tree["split_conditions"]
        split_type = tree["split_type"]
        base_weights = tree["base_weights"]

        node = 0
        while left_children[node] != -1:
            feature_index = split_indices[node]
            value = np.float32(row[feature_index]).item()

            if isnan(value):
                node = left_children[node] if int(default_left[node]) == 1 else right_children[node]
                continue

            if int(split_type[node]) != 0:
                raise ValueError("Categorical XGBoost splits are not supported in Android fallback")

            threshold = np.float32(split_conditions[node]).item()
            node = left_children[node] if value < threshold else right_children[node]

        return np.float32(base_weights[node]).item()


class RandomForestJsonRegressor:
    def __init__(self, model_data):
        self.trees = model_data.get("estimators", [])

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        array = np.asarray(array, dtype=np.float64)
        predictions = np.zeros(array.shape[0], dtype=np.float64)
        for tree in self.trees:
            predictions += np.asarray([self._predict_tree(tree, row) for row in array], dtype=np.float64)
        if not self.trees:
            return predictions
        return predictions / len(self.trees)

    def _predict_tree(self, tree, row):
        node = 0
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        features = tree["feature"]
        thresholds = tree["threshold"]
        values = tree["value"]

        while children_left[node] != -1:
            feature_index = features[node]
            threshold = thresholds[node]
            value = float(row[feature_index])
            node = children_left[node] if value <= threshold else children_right[node]

        return values[node]


class RidgeJsonRegressor:
    def __init__(self, model_data):
        self.coef = np.asarray(model_data["coef"], dtype=np.float64)
        self.intercept = float(model_data["intercept"])

    def predict(self, x):
        array = x.values if hasattr(x, "values") else np.asarray(x)
        array = np.asarray(array, dtype=np.float64)
        return np.dot(array, self.coef) + self.intercept


class CompleteModelPredictor:
    def __init__(self, model_path, hemoglobin_model_path=None):
        self.model_path = model_path
        self.hemoglobin_model_path = hemoglobin_model_path
        self.model_data = None
        self.best_hemoglobin_model = None
        self.best_pi_model = None
        self.best_lactate_model = None
        self.hb_scaler = None
        self.pi_scaler = None
        self.lactate_scaler = None
        self.label_encoders = {}
        self.hb_feature_names = []
        self.pi_feature_names = []
        self.lactate_feature_names = []
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        with open(self.model_path, "r", encoding="utf-8") as bundle_file:
            self.model_data = json.load(bundle_file)

        scaler_data = self.model_data.get("scalers", {})
        self.hb_scaler = StandardScalerLite(scaler_data["hb"]) if "hb" in scaler_data else None
        self.pi_scaler = StandardScalerLite(scaler_data["pi"]) if "pi" in scaler_data else None
        self.lactate_scaler = StandardScalerLite(scaler_data["lactate"]) if "lactate" in scaler_data else None
        self.label_encoders = {
            name: JsonLabelEncoder(classes)
            for name, classes in self.model_data.get("label_encoders", {}).items()
        }

        feature_names = self.model_data.get("feature_names", {})
        self.hb_feature_names = feature_names.get("hb", [])
        self.pi_feature_names = feature_names.get("pi", [])
        self.lactate_feature_names = feature_names.get("lactate", [])

        models = self.model_data.get("models", {})
        hb_model = models.get("hemoglobin")
        if hb_model and hb_model.get("format") == "xgboost_json":
            resolved_path = self.hemoglobin_model_path or os.path.join(
                os.path.dirname(self.model_path),
                hb_model.get("file", "hb_model.json"),
            )
            self.best_hemoglobin_model = XGBoostJsonRegressor(resolved_path)

        pi_model = models.get("pi")
        if pi_model and pi_model.get("format") == "random_forest_json":
            self.best_pi_model = RandomForestJsonRegressor(pi_model)

        lactate_model = models.get("lactate")
        if lactate_model and lactate_model.get("format") == "ridge_json":
            self.best_lactate_model = RidgeJsonRegressor(lactate_model)

    def preprocess_features(self, frame, label_encoders):
        processed = frame.copy()
        for column in processed.select_dtypes(include=["object", "category"]).columns.tolist():
            if column in label_encoders:
                encoded = label_encoders[column].transform(processed[column].fillna("missing").astype(str))
                processed[column] = encoded
            else:
                processed[column] = 0

        for column in processed.select_dtypes(include=[np.number]).columns.tolist():
            if processed[column].isna().any():
                processed[column] = processed[column].fillna(0)
        return processed

    def predict_single_patient(self, patient_features, patient_id):
        try:
            frame = pd.DataFrame([patient_features])

            if self.best_hemoglobin_model and self.hb_feature_names:
                processed = self.preprocess_features(frame, self.label_encoders)
                for feature in set(self.hb_feature_names) - set(processed.columns):
                    processed[feature] = 0
                selected = processed[self.hb_feature_names]
                hb_predictions = self.best_hemoglobin_model.predict(
                    self.hb_scaler.transform(selected) if self.hb_scaler else selected.values
                )
            else:
                hb_predictions = [np.nan]

            if self.best_pi_model and self.pi_feature_names:
                processed = self.preprocess_features(frame, self.label_encoders)
                for feature in set(self.pi_feature_names) - set(processed.columns):
                    processed[feature] = 0
                selected = processed[self.pi_feature_names]
                pi_predictions = self.best_pi_model.predict(
                    self.pi_scaler.transform(selected) if self.pi_scaler else selected.values
                )
            else:
                pi_predictions = [np.nan]

            if self.best_lactate_model and self.lactate_feature_names:
                processed = self.preprocess_features(frame, self.label_encoders)
                for feature in set(self.lactate_feature_names) - set(processed.columns):
                    processed[feature] = 0
                selected = processed[self.lactate_feature_names]
                lactate_predictions = np.round(
                    self.best_lactate_model.predict(
                        self.lactate_scaler.transform(selected) if self.lactate_scaler else selected.values
                    ),
                    1,
                )
            else:
                lactate_predictions = [np.nan]

            def classify_pi(value):
                if pd.isna(value):
                    return "未知"
                if value < 0.3:
                    return "弱灌注"
                if value < 1.0:
                    return "可接受"
                return "最佳"

            results_df = pd.DataFrame(
                {
                    "预测血红蛋白": hb_predictions,
                    "预测PI": pi_predictions,
                    "PI分类": [classify_pi(value) for value in pi_predictions],
                    "预测乳酸": lactate_predictions,
                }
            )

            def interpret_hemoglobin(value):
                if pd.isna(value):
                    return "无法解读"
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

            def interpret_pi(value):
                if value == "弱灌注":
                    return "血流灌注指数较低，可能影响测量准确性"
                if value == "可接受":
                    return "血流灌注指数可接受"
                if value == "最佳":
                    return "血流灌注指数良好"
                return "无法解读"

            def interpret_lactate(value):
                if pd.isna(value):
                    return "无法解读"
                if value < 1.0:
                    return "正常"
                if value < 2.0:
                    return "轻度升高"
                if value < 4.0:
                    return "中度升高"
                return "重度升高"

            results = []
            for index in range(len(results_df)):
                hb = results_df.iloc[index]["预测血红蛋白"]
                pi = results_df.iloc[index]["预测PI"]
                lactate = results_df.iloc[index]["预测乳酸"]
                pi_class = results_df.iloc[index]["PI分类"]
                results.append(
                    {
                        "patient_id": patient_id,
                        "hemoglobin": {
                            "value": round(float(hb), 1) if not pd.isna(hb) else None,
                            "unit": "g/L",
                            "clinical_interpretation": interpret_hemoglobin(float(hb)) if not pd.isna(hb) else "无法解读",
                        },
                        "perfusion_index": {
                            "value": round(float(pi), 3) if not pd.isna(pi) else None,
                            "classification": pi_class,
                            "interpretation": interpret_pi(pi_class),
                        },
                        "lactate": {
                            "value": round(float(lactate), 1) if not pd.isna(lactate) else None,
                            "unit": "mmol/L",
                            "clinical_interpretation": interpret_lactate(float(lactate)) if not pd.isna(lactate) else "无法解读",
                        },
                        "prediction_time": datetime.now().isoformat(),
                        "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}",
                    }
                )

            return results_df, json.dumps(results, ensure_ascii=False, indent=2)
        except Exception:
            traceback.print_exc()
            return None, None
