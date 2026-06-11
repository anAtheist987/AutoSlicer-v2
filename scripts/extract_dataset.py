"""Wait for the zip transfer to finish, then extract with GBK filename handling."""
import os, sys, time, zipfile
from pathlib import Path

ZIP = '/root/AgentGateway/数据集.zip'
OUT = Path('/root/Autoslicer/data/raw')

prev = -1
while True:
    cur = os.path.getsize(ZIP)
    if cur == prev:
        try:
            with zipfile.ZipFile(ZIP) as z:
                bad = z.testzip()
            print('zip complete, testzip ->', bad, flush=True)
            break
        except Exception as e:
            print('not ready:', e, flush=True)
    prev = cur
    time.sleep(30)

OUT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        name = info.filename
        if not (info.flag_bits & 0x800):  # no UTF-8 flag -> raw bytes were decoded as cp437
            try:
                name = name.encode('cp437').decode('gbk')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        target = OUT / name
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(target, 'wb') as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
print('extraction done', flush=True)
