import numpy as np
import pandas as pd


class CompleteFeatureEngineer:
    def __init__(self):
        self.selected_features = []
        self.pi_features = []
        self.lactate_features = []
        self.feature_names = []

    def create_advanced_features(self, features_dict):
        features = features_dict.copy()

        if "R值_mean" in features:
            features["R值_mean_原始"] = features["R值_mean"]
            features["R值_mean"] = max(features["R值_mean"], 0.1) if features["R值_mean"] > 0 else 0.1
            features["R值_mean"] = np.clip(features["R值_mean"], 0.3, 3.0)
            features["R值异常标记"] = 1 if (features["R值_mean"] < 0.5 or features["R值_mean"] > 2.0) else 0
            features["R值极端异常标记"] = 1 if (features["R值_mean"] < 0.3 or features["R值_mean"] > 3.0) else 0
            features["R值_平方"] = features["R值_mean"] ** 2
            features["R值_倒数"] = 1 / (features["R值_mean"] + 1e-10)

        if "PI_估算" in features:
            features["PI_原始"] = features["PI_估算"]
            features["PI_估算"] = max(features["PI_估算"], 0.01) if features["PI_估算"] > 0 else 0.01

            if features["PI_估算"] < 0.3:
                features["PI_分类"] = "弱灌注"
            elif features["PI_估算"] < 1.0:
                features["PI_分类"] = "可接受"
            else:
                features["PI_分类"] = "最佳"

            features["PI_log"] = np.log1p(features["PI_估算"])
            features["PI_平方"] = features["PI_估算"] ** 2
            if "心率" in features:
                features["PI_心率比"] = features["PI_估算"] / (features["心率"] + 1e-10)
            if "血氧饱和度" in features:
                features["PI_血氧交互"] = features["PI_估算"] * features["血氧饱和度"]
            if "年龄" in features:
                features["PI_年龄调整"] = features["PI_估算"] * (1 + features["年龄"] / 100)

        if "乳酸" in features and not pd.isna(features["乳酸"]):
            features["有乳酸数据"] = 1
            if "心率" in features:
                features["乳酸_心率交互"] = features["乳酸"] * features["心率"]
            if "血氧饱和度" in features:
                features["乳酸_血氧交互"] = features["乳酸"] * (100 - features["血氧饱和度"])
            if "体温" in features:
                features["乳酸_体温交互"] = features["乳酸"] * features["体温"]
            if "血红蛋白" in features:
                features["乳酸_血红蛋白比"] = features["乳酸"] / (features["血红蛋白"] + 1e-10)
            features["乳酸_log"] = np.log1p(features["乳酸"])
            features["乳酸_平方根"] = np.sqrt(features["乳酸"] + 1)
        else:
            features["有乳酸数据"] = 0
            features["乳酸"] = np.nan

        if "年龄" in features and "血氧饱和度" in features:
            features["年龄_血氧交互"] = features["年龄"] * (features["血氧饱和度"] / 100)
        if "氧分压" in features and "血氧饱和度" in features:
            features["氧合指数"] = features["氧分压"] / (features["血氧饱和度"] / 100 + 1e-10)
        if "R值_mean" in features and "血氧饱和度" in features:
            features["R值_血氧交互"] = features["R值_mean"] * features["血氧饱和度"]

        for column in ["年龄", "心率", "体温", "R值_mean", "红光_mean", "红外光_mean"]:
            if column in features and features[column] > 0:
                features[f"{column}_log"] = np.log1p(features[column])
                features[f"{column}_sqrt"] = np.sqrt(features[column])

        if "信号信噪比" in features and "红光_pulse_count" in features:
            features["信号质量指数"] = features["信号信噪比"] * np.log1p(features["红光_pulse_count"] + 1)

        if "信号质量评分" in features:
            score = features["信号质量评分"]
            if score <= 3:
                features["信号质量等级"] = "差"
                features["信号质量等级_num"] = 0
            elif score <= 6:
                features["信号质量等级"] = "中"
                features["信号质量等级_num"] = 1
            else:
                features["信号质量等级"] = "优"
                features["信号质量等级_num"] = 2

        if "血红蛋白" in features and not pd.isna(features["血红蛋白"]):
            hb = features["血红蛋白"]
            if hb < 80:
                features["血红蛋白区间"] = "极重度贫血"
                features["血红蛋白区间_num"] = 0
            elif hb < 120:
                features["血红蛋白区间"] = "贫血"
                features["血红蛋白区间_num"] = 1
            elif hb <= 160:
                features["血红蛋白区间"] = "正常"
                features["血红蛋白区间_num"] = 2
            elif hb <= 170:
                features["血红蛋白区间"] = "偏高"
                features["血红蛋白区间_num"] = 3
            else:
                features["血红蛋白区间"] = "重度偏高"
                features["血红蛋白区间_num"] = 4

        if "R值_mean_原始" in features and "R值_mean" in features:
            features["R值修正量"] = features["R值_mean"] - features["R值_mean_原始"]

        return features
