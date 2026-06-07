# Blood Check Android V2 Agent Notes

本文档给后续维护者或 agent 使用，记录当前 Android 项目的架构、技术实现、依赖、运行方式和注意事项。

## 项目结构

- `blood-check-android/`
  - Android 主工程，包名 `com.bloodcheck.app`。
  - 使用 Kotlin + XML View + Chaquopy，在 Android 端直接运行 Python 推理逻辑。
  - Debug APK 输出位置：`blood-check-android/app/build/outputs/apk/debug/app-debug.apk`。
- `origininfrerence/`
  - 原始 Windows 推理服务与训练/推理脚本。
  - `receive_data/models.pkl`：基础生理指标模型来源。
  - `receive_data/多任务模型.pkl`：血红蛋白、PI、乳酸多任务模型来源。
  - `export_android_origin_models.py`：把 `.pkl` 模型导出为 Android 可加载的 JSON 资产。
- `blood-check-android/app/src/main/assets/`
  - `model/`、`baseinfo/`、`spectral_data/`：旧单次预测链路资产。
  - `origin_model/`：从 `origininfrerence` 导出的实时推理模型 JSON。
- `blood-check-android/app/src/main/python/`
  - `predict_service.py`：旧按患者 ID + 本地 CSV 光谱做单次预测的入口。
  - `origin_live_inference.py`：当前实时 BLE 推理入口，完全在 Android 本地执行。
  - `model_predictor.py`、`spectrum_processor.py`、`feature_engineer.py`：旧预测链路的模型/光谱/特征模块。

## Android 架构

主界面在 `MainActivity.kt`，采用单 Activity 架构：

- `MainActivity.kt`
  - 负责 UI 绑定、患者信息录入、监测开始/停止、图表刷新、导出入口。
  - `MIN_LIVE_SIGNAL_ROWS = 200`，实时 BLE 采样至少 200 行后才触发一次推理。
  - 监护数据区展示：HB、LAC、PI、SpO2、PaO2、HR、FO2Hb、血糖、K+、Na+。
  - PaO2 没有实时数据模型，目前不填默认值，UI 保持空白。
- `PredictViewModel.kt`
  - 负责 Kotlin 调 Chaquopy Python。
  - `predictLiveForMonitoring(...)` 调用 `origin_live_inference.predict_live(...)`。
  - `predict(...)` / `predictForMonitoring(...)` 是旧患者 ID 预测入口。
- `BleMonitorManager.kt`
  - 扫描并连接 BLE 设备 `Nordic_UART`。
  - Nordic UART Service UUID：`6E400001-B5A3-F393-E0A9-E50E24DCCA9E`。
  - TX Notify Characteristic UUID：`6E400003-B5A3-F393-E0A9-E50E24DCCA9E`。
  - 解析通知文本中的 `array:[...]`，取第 4、5 个值作为红光 `C4` 和红外 `C5`。
  - 内存中最多保留 6000 个采样点。
- `PatientMonitorMerge.kt`
  - 将 Python 返回的 `PredictionResponse` 合并为首页监护状态。
- `PatientDataStorage.kt`
  - 保存患者录入信息、采集周期、告警阈值和历史记录。
- `PatientDataExport.kt`
  - 将历史记录导出 CSV 到下载目录。
- `AssetCopyHelper.kt`
  - 首次预测前把 assets 下的 `model`、`origin_model`、`baseinfo`、`spectral_data` 拷贝到 App 私有文件目录，供 Python 按文件路径读取。

## 实时推理链路

实时链路的核心入口：

```text
BLE Nordic_UART
  -> BleMonitorManager.snapshot()
  -> MainActivity.monitorPollRunnable
  -> PredictViewModel.predictLiveForMonitoring()
  -> Chaquopy Python origin_live_inference.predict_live()
  -> PredictionResponse.fromJson()
  -> PatientMonitorMerge.merge()
  -> MainActivity.bindMonitoringMetrics()
```

`origin_live_inference.py` 中的模型执行分两阶段：

1. `BasePhysioPredictor`
   - 输入红光/红外实时信号。
   - 提取时域、频域、AC/DC、R 值、信号质量等特征。
   - 输出 6 项基础生理指标：
     - 心率
     - 血氧饱和度
     - 氧合血红蛋白分数 FO2Hb
     - 血糖
     - K+
     - Na+
2. `MultiTaskPredictor`
   - 输入基础生理指标 + 光谱特征。
   - 输出：
     - 血红蛋白 Hb
     - 灌注指数 PI
     - 乳酸 Lac

Python 返回 JSON 后，Android 展示字段来自：

- `prediction.hemoglobin` -> HB
- `prediction.lactate` -> LAC
- `prediction.perfusion_index` -> PI
- `patient_info["血氧饱和度"]` -> SpO2
- `patient_info["心率"]` -> HR
- `patient_info["氧合血红蛋白分数(FO2Hb)"]` -> FO2Hb
- `patient_info["血糖"]` -> 血糖
- `patient_info["K+"]` -> K+
- `patient_info["Na+"]` -> Na+
- `patient_info["氧分压"]` -> PaO2；实时链路当前不提供该字段。

## 模型移植方式

Android 端没有直接加载原始 `.pkl`、LightGBM、XGBoost 原生库，而是使用 JSON 化模型：

- `origininfrerence/export_android_origin_models.py`
  - 读取：
    - `origininfrerence/receive_data/models.pkl`
    - `origininfrerence/receive_data/多任务模型.pkl`
  - 输出到：
    - `blood-check-android/app/src/main/assets/origin_model/origin_base_models.json`
    - `blood-check-android/app/src/main/assets/origin_model/origin_multitask_models.json`
    - `blood-check-android/app/src/main/assets/origin_model/origin_multitask_hb_model.json`

`origin_live_inference.py` 内置了轻量 JSON 推理器：

- `LightGBMJsonRegressor`
- `XGBoostJsonRegressor`
- `GradientBoostingJsonRegressor`
- `RidgeJsonRegressor`
- `IsolationForestJson`
- `StandardScalerLite`
- `JsonLabelEncoder`

这样做是为了避开 Chaquopy 上 XGBoost/LightGBM 原生依赖不可用或不可稳定打包的问题。

## 依赖

Android Gradle 依赖在 `blood-check-android/app/build.gradle.kts`：

- Android Gradle Plugin
- Kotlin Android
- Chaquopy
- AndroidX Core KTX
- AppCompat
- Material Components
- Activity KTX
- ConstraintLayout
- Lifecycle LiveData/ViewModel KTX

Chaquopy Python 版本：

- Python `3.10`
- `numpy==1.23.3`
- `pandas==2.1.3`
- `scipy==1.8.1`
- `scikit-learn==1.3.2`
- `PyWavelets==1.1.1`
- `joblib==1.3.2`

原始推理服务依赖在 `origininfrerence/requirements-origin-inference.txt`，包含：

- numpy
- pandas
- scipy
- scikit-learn
- PyWavelets
- joblib
- lightgbm
- bleak
- openpyxl
- xgboost

注意：原始服务依赖不等于 Android 可打包依赖。Android 当前通过 JSON 模型推理规避了 `xgboost`、`lightgbm` 原生包。

## BLE 与权限

Manifest 已声明：

- `android.hardware.bluetooth_le`
- Android 11 及以下：
  - `BLUETOOTH`
  - `BLUETOOTH_ADMIN`
  - `ACCESS_FINE_LOCATION`
- Android 12 及以上：
  - `BLUETOOTH_SCAN`
  - `BLUETOOTH_CONNECT`

运行时权限逻辑在 `MainActivity.requiredBlePermissions()`。

当前测试手机：

- ADB id：`6da31a21`
- 设备：OnePlus 8 / Android 11
- Android 11 需要位置权限才能 BLE 扫描。

常用 ADB 命令：

```bash
adb devices
adb -s 6da31a21 install -r blood-check-android/app/build/outputs/apk/debug/app-debug.apk
adb -s 6da31a21 shell monkey -p com.bloodcheck.app -c android.intent.category.LAUNCHER 1
adb -s 6da31a21 logcat -d -s BloodCheckBle python.stderr AndroidRuntime
```

`BloodCheckBle` 日志可确认扫描、连接、服务发现、通知解析和采样数量，例如 `parsedSamples=200`。

## 构建与运行

在 Android 工程目录执行：

```bash
cd blood-check-android
./gradlew :app:assembleDebug
```

APK 位置：

```text
blood-check-android/app/build/outputs/apk/debug/app-debug.apk
```

安装到手机：

```bash
adb -s 6da31a21 install -r app/build/outputs/apk/debug/app-debug.apk
```

如果修改了 `origininfrerence` 原始模型，需要先重新导出 Android JSON 模型：

```bash
cd /Users/lili/Project/pangxin/blood-check-androidV2
origininfrerence/.venv/bin/python origininfrerence/export_android_origin_models.py
cd blood-check-android
./gradlew :app:assembleDebug
```

## 本地验证

实时推理 Python 可在 macOS 本地用原始服务 venv 做烟测：

```bash
cd blood-check-android
PYTHONPATH=app/src/main/python ../origininfrerence/.venv/bin/python - <<'PY'
import pandas as pd
from origin_live_inference import OriginLiveInferenceService

sample = pd.read_csv('../origininfrerence/receive_data/spectral_data.csv').head(260)
service = OriginLiveInferenceService('app/src/main/assets')
result = service.predict('11', sample['C4'].tolist(), sample['C5'].tolist())
print(result)
PY
```

回归点：

- 实时推理应返回 `心率`、`血氧饱和度`、`氧合血红蛋白分数(FO2Hb)`、`血糖`、`K+`、`Na+`。
- 实时推理当前不应返回默认 `氧分压`。
- Kotlin 传入 Python 的列表在 Chaquopy 中可能是 Java `ArrayList`，`origin_live_inference._to_float_array()` 已做兼容转换。

## UI 展示约定

首页监护卡片目前展示：

- HB：来自 `prediction.hemoglobin`
- LAC：来自 `prediction.lactate`
- PI：来自 `prediction.perfusion_index`
- SpO2：来自 `patient_info["血氧饱和度"]`
- PaO2：来自 `patient_info["氧分压"]`；无数据时空白
- HR：来自 `patient_info["心率"]`
- FO2Hb：来自 `patient_info["氧合血红蛋白分数(FO2Hb)"]`
- 血糖：来自 `patient_info["血糖"]`
- K+：来自 `patient_info["K+"]`
- Na+：来自 `patient_info["Na+"]`

布局文件：

- `activity_main.xml`
- `include_monitor_row_hb_lac.xml`
- `include_monitor_row_pi_spo2.xml`
- `include_monitor_row_pao2_hr.xml`
- `include_monitor_row_fo2hb_glucose.xml`
- `include_monitor_row_k_na.xml`

## 重要注意事项

- 不要把 PaO2 写成固定默认值。实时模型当前没有 PaO2 数据时，UI 保持空白。
- BLE 原始通知可能被分包，`BleMonitorManager` 使用 `pendingText` 缓冲并用正则解析 `array:[...]`。
- 采样不足 200 行时不会推理，会提示实时数据不足。
- BLE 连接可能中途断开，日志中可能出现 `connection state status=8 newState=0`；需要重新开始监测或等待重连逻辑。
- Android assets 更新后要重新构建 APK；仅修改 `origininfrerence` 目录不会自动进入 App。
- `AssetCopyHelper.ensureAppDataReady()` 每次会覆盖拷贝 assets 到 App 私有目录，便于模型资产更新。
- Chaquopy 打包 Python 包时要使用 Android 可用版本；不要直接把桌面 Python 依赖照搬到 `app/build.gradle.kts`。
- `local.properties` 中 SDK 路径是本机配置，不应作为跨机器固定假设。
- 根目录当前不是 Git 仓库；如果要做版本管理，需确认实际 Git 根目录后再提交。

