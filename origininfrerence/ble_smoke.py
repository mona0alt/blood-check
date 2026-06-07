from pathlib import Path
import asyncio
import sys
import traceback

ROOT = Path(__file__).resolve().parent
SITE_PACKAGES = ROOT / ".venv-py314" / "lib" / "python3.14" / "site-packages"
LOG_PATH = ROOT / "ble_smoke.log"

sys.path.insert(0, str(SITE_PACKAGES))


async def main():
    from bleak import BleakScanner

    LOG_PATH.write_text("before discover\n", encoding="utf-8")
    devices = await BleakScanner.discover(timeout=0.5, return_adv=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"after discover {len(devices)}\n")


try:
    asyncio.run(main())
except Exception:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    raise
