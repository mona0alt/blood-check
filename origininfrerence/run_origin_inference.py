from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SITE_PACKAGES = ROOT / ".venv-py314" / "lib" / "python3.14" / "site-packages"
LOG_PATH = ROOT / "origin_inference_service.log"
SCRIPT_PATH = ROOT / "蓝牙设备数据接收及生理指标预测V2.py"

sys.path.insert(0, str(SITE_PACKAGES))
os.chdir(ROOT)

log_file = LOG_PATH.open("a", encoding="utf-8", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print("\n===== starting origin inference service =====", flush=True)
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
