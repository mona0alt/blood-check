import json
import os
import sys
import types
import traceback


def _safe_version(mod):
    return getattr(mod, "__version__", "unknown")


def _register_pickle_stubs():
    class EnhancedFeatureEngineer:
        pass

    class AdvancedIntelligentCorrectionSystem:
        pass

    class EnhancedDataLoader:
        pass

    class EnhancedDataBalancer:
        pass

    class EnhancedMultiTaskPredictor:
        pass

    classes = [
        EnhancedFeatureEngineer,
        AdvancedIntelligentCorrectionSystem,
        EnhancedDataLoader,
        EnhancedDataBalancer,
        EnhancedMultiTaskPredictor,
    ]

    for mod_name in [__name__, "__main__"]:
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
        for cls in classes:
            setattr(mod, cls.__name__, cls)



def run_phase0_test(model_path: str) -> str:
    result = {
        "success": True,
        "python": sys.version,
        "package_versions": {},
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "model_load": None,
        "notes": [],
    }

    try:
        import numpy
        import pandas
        import scipy
        import sklearn
        import joblib
        import pywt

        result["package_versions"] = {
            "numpy": _safe_version(numpy),
            "pandas": _safe_version(pandas),
            "scipy": _safe_version(scipy),
            "scikit_learn": _safe_version(sklearn),
            "joblib": _safe_version(joblib),
            "pywt": _safe_version(pywt),
        }
    except Exception as e:
        result["success"] = False
        result["notes"].append(f"基础依赖导入失败: {e}")
        result["traceback"] = traceback.format_exc()
        return json.dumps(result, ensure_ascii=False, indent=2)

    try:
        import xgboost  # type: ignore
        result["package_versions"]["xgboost"] = _safe_version(xgboost)
    except Exception as e:
        result["notes"].append(f"xgboost 未安装或不可用: {e}")

    if result["model_exists"]:
        try:
            _register_pickle_stubs()
            import joblib

            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                result["model_load"] = {
                    "loaded": True,
                    "keys": sorted(list(model_data.keys()))[:20],
                }
            else:
                result["model_load"] = {
                    "loaded": True,
                    "type": str(type(model_data)),
                }
        except Exception as e:
            result["success"] = False
            result["model_load"] = {
                "loaded": False,
                "error": str(e),
                "error_type": e.__class__.__name__,
            }
            result["notes"].append(
                "如果这里失败且报错与 xgboost/pickle 相关，说明需要先把 XGBoost 模型改为原生格式或 ONNX。"
            )
            result["traceback"] = traceback.format_exc()
    else:
        result["success"] = False
        result["notes"].append("模型文件不存在，未执行 joblib.load")

    return json.dumps(result, ensure_ascii=False, indent=2)
