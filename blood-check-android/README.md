# BloodCheck Android 交付说明

## 1. 项目概况

本项目是血检离线预测 Android 客户端，输入患者编号后，在设备本地完成以下流程：

1. 从本地 CRF 数据中读取患者基础信息。
2. 从本地光谱数据中读取对应患者的红光/红外信号。
3. 在设备本地执行特征工程、模型推理和结果解释。
4. 输出血红蛋白、灌注指数 PI、乳酸三项预测结果。

当前版本为离线版本，不依赖网络服务。模型、CRF 信息和光谱样本数据均随 APK 一起打包。

## 2. 当前交付状态

- Android 工程已可正常同步、编译、打包。
- APK 已验证可安装并运行。
- Python 推理通过 Chaquopy 集成到 Android 应用。
- 设备端推理已改为基于 JSON 模型包加载，不再依赖设备上反序列化原始 pickle 模型。
- 已在 Android 模拟器上完成端到端验证。

验证结果：

- 测试患者编号：`01220028246780`
- 预测结果：
  - Hb：`160.0 g/L`
  - PI：`12.275`
  - 乳酸：`0.7 mmol/L`

## 3. 运行环境

- Android Studio：建议 Giraffe 及以上版本
- JDK：17
- Gradle：随工程 wrapper 提供
- Android SDK：
  - `compileSdk = 34`
  - `targetSdk = 34`
  - `minSdk = 24`
- Python for build：`/usr/local/bin/python3.10`
- ABI：
  - `arm64-v8a`
  - `x86_64`

说明：

- 构建时 Chaquopy 会使用本机 Python 3.10 参与打包。
- 首次启动或首次预测时，应用会解压 Python 运行环境和数据，耗时会明显长于后续运行。

## 4. 工程结构

### 4.1 Android 工程主目录

- `app/build.gradle.kts`
  - Android 配置、Chaquopy 配置、Python 依赖配置
- `app/src/main/AndroidManifest.xml`
  - 应用入口声明
- `app/src/main/res/`
  - 页面布局、颜色、主题、文案资源

### 4.2 Kotlin 代码

- `app/src/main/java/com/bloodcheck/app/MainActivity.kt`
  - 主页面
  - 负责输入患者编号、触发预测、展示结果
- `app/src/main/java/com/bloodcheck/app/PredictViewModel.kt`
  - 页面状态管理
  - 调用 Python 预测服务
- `app/src/main/java/com/bloodcheck/app/PredictResult.kt`
  - Python 返回 JSON 的解析与 Kotlin 数据模型
- `app/src/main/java/com/bloodcheck/app/AssetCopyHelper.kt`
  - 首次运行时将 assets 中的数据复制到应用私有目录

### 4.3 Python 推理代码

- `app/src/main/python/predict_service.py`
  - Android 侧 Python 入口
  - 根据患者编号组织完整预测流程
- `app/src/main/python/spectrum_processor.py`
  - 光谱信号读取、预处理和特征提取
- `app/src/main/python/feature_engineer.py`
  - 业务特征构造
- `app/src/main/python/model_predictor.py`
  - 模型加载与预测执行
  - 当前加载 `android_model_bundle.json` 和 `hb_model.json`

### 4.4 资产数据

- `app/src/main/assets/baseinfo/CRF_info.csv`
  - 患者基础信息表
- `app/src/main/assets/spectral_data/`
  - 光谱样本数据，共 `141` 个文件
- `app/src/main/assets/model/android_model_bundle.json`
  - Android 端主模型包
  - 包含 PI 随机森林、乳酸 Ridge、Scaler、LabelEncoder、特征名等
- `app/src/main/assets/model/hb_model.json`
  - Hb 的 XGBoost JSON 模型

说明：

- `Multi_task_modelV1.pkl` 和 `Multi_task_model_android.pkl` 目前仍保留在 assets 中，主要用于历史对照和调试。
- 应用实际运行时使用的是 `android_model_bundle.json` 和 `hb_model.json`。

### 4.5 根目录下的辅助脚本

以下脚本不参与 Android 运行，但用于模型准备、数据准备和验证：

- `export_android_models.py`
  - 从原始训练产物导出 Android 可用的 JSON 模型包
- `android_python/`
  - 与 Android 端 Python 基本对应的本地调试版本
- `test_predict_service.py`
  - 本地端到端预测验证
- `test_hb_json_predictor.py`
  - Hb JSON 模型与原 XGBoost 输出一致性验证
- `test_cross_validate.py`
  - Android 侧调用流程交叉验证

## 5. 运行流程说明

应用运行流程如下：

1. 打开应用，输入患者编号。
2. Kotlin 层调用 `AssetCopyHelper`，将 `assets/model`、`assets/baseinfo`、`assets/spectral_data` 复制到应用私有目录。
3. `PredictViewModel` 通过 Chaquopy 调用 `predict_service.predict_by_patient_id`。
4. Python 层根据患者编号：
   - 查询 CRF 信息
   - 查找对应光谱文件
   - 生成特征
   - 调用模型预测
   - 返回 JSON 结果
5. Kotlin 层解析 JSON，并展示患者信息与三项指标结果。

## 6. 打包方式

### 6.1 使用 Android Studio

1. 用 Android Studio 打开目录 `blood-check-android`
2. 等待 Gradle Sync 完成
3. 执行：
   - `Build > Make Project`
   - 或 `Build > Build Bundle(s) / APK(s) > Build APK(s)`

### 6.2 使用命令行

在 `blood-check-android` 目录执行：

```bash
./gradlew :app:assembleDebug
```

如需 Release 包，可执行：

```bash
./gradlew :app:assembleRelease
```

当前已验证通过的调试包输出路径：

```text
app/build/outputs/apk/debug/app-debug.apk
```

当前调试 APK 大小约 `136 MB`。

## 7. 安装方式

### 7.1 Android Studio 安装

1. 连接真机或启动模拟器
2. 点击运行按钮安装应用

### 7.2 ADB 安装

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 8. 使用方式

1. 启动应用。
2. 在输入框中填写患者编号。
3. 点击“开始预测”。
4. 等待本地初始化和推理完成。
5. 查看以下结果：
   - 患者信息
   - 血红蛋白
   - 灌注指数 PI
   - 乳酸

使用说明：

- 患者编号必须与本地 CRF 和光谱数据匹配。
- 如果未找到患者编号，会提示“未找到该患者信息，请检查ID”。
- 首次运行较慢属于正常现象。

## 9. 已验证内容

已完成以下验证：

- Android 工程可编译打包
- Python 依赖可在 Android 中加载
- 本地 JSON 模型包预测结果与原始模型结果一致
- 模拟器安装、启动、输入患者编号、显示结果全流程可用

已通过的关键命令：

```bash
./gradlew :app:assembleDebug
```

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 10. 注意事项

- 构建机器需要安装 Python 3.10，并保证路径 `/usr/local/bin/python3.10` 可用。
- 若更换模型，需要重新执行 `export_android_models.py`，并同步更新：
  - `Model/android_model_bundle.json`
  - `Model/hb_model.json`
  - `app/src/main/assets/model/`
- 当前应用是本地离线演示/交付版本，数据范围以 assets 中打包内容为准。
- 如果后续需求方要求新增患者数据、替换模型或接入在线服务，需要单独扩展数据更新和版本管理机制。

## 11. 交付清单

建议交付以下内容：

- `blood-check-android/`
  - 完整 Android 工程
- `Model/android_model_bundle.json`
  - Android 模型导出结果
- `Model/hb_model.json`
  - Hb JSON 模型
- `Baseinfo/CRF_info.csv`
  - 基础信息数据
- `docs/superpowers/reports/2026-03-21-android-blood-check-phase0-results.md`
  - 阶段性验证记录

安装包，可直接提供：

- `app/build/outputs/apk/debug/app-debug.apk`

