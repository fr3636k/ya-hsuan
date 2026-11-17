"""照片 vs 3D 模型並排比對（開發用）。
用法：python scripts/compare.py <輸出.png> [grip]
需要本機預覽伺服器（node serve.mjs）與 Chrome。
每張照片：裁切框（人從頭頂到腳底）、對應的鏡頭方位角與仰角。
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
# 照片, 裁切框 (x0, y0, x1, y1), 方位角, 仰角, 標籤
SHOTS = [
    ("S__147939358_0.jpg", (300, 315, 830, 1380), 0, 2, "正面"),
    ("S__147939352_0.jpg", (330, 40, 780, 1450), 40, 4, "斜前(人台)"),
    ("S__147939359_0.jpg", (360, 30, 820, 1465), 90, 4, "側面A"),
    ("S__147939360_0.jpg", (270, 20, 750, 1455), 270, 4, "側面B"),
    ("S__147939354_0.jpg", (280, 85, 820, 1405), 180, 4, "背面(人台)"),
]
H = 640


def render(az, el, grip, out):
    """用 puppeteer（真實時間）截圖：Draco 模型在 Chrome 虛擬時間模式下不會解碼。
    需要 PP_DIR 環境變數指向裝了 puppeteer-core 與 shot.mjs 的資料夾。"""
    pp = os.environ.get("PP_DIR")
    q = f"grip={grip}&cam={az},{el},5.4,0.5&light=on&nofx"
    d, name = os.path.split(out)
    name = os.path.splitext(name)[0]
    subprocess.run(["node", "shot.mjs", d, f"{name}|{q}|viewer.html|700|900"], cwd=pp,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return os.path.exists(out) and np.asarray(Image.open(out).convert("L")).mean() > 3


def fit_render(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a > 40)
    y0, y1 = ys.min(), ys.max()
    cx = int(np.median(xs))
    h = y1 - y0
    box = (cx - h * 0.42, y0 - h * 0.02, cx + h * 0.42, y1 + h * 0.02)
    return im.crop(tuple(map(int, box))).resize((int(H * 0.84 / 1.04), H))


def fit_photo(name, box):
    im = Image.open(ROOT / "data" / name).convert("RGB")
    x0, y0, x1, y1 = box
    h = y1 - y0
    cx = (x0 + x1) / 2
    b = (cx - h * 0.42, y0 - h * 0.02, cx + h * 0.42, y1 + h * 0.02)
    return im.crop(tuple(map(int, b))).resize((int(H * 0.84 / 1.04), H))


def main():
    out = sys.argv[1]
    grip = sys.argv[2] if len(sys.argv) > 2 else "0"
    only = set(map(int, sys.argv[3].split(","))) if len(sys.argv) > 3 else None
    tmp = Path(tempfile.mkdtemp())
    cols = []
    for name, box, az, el, label in SHOTS:
        if only and az not in only:
            continue
        r = tmp / f"r{az}.png"
        if not render(az, el, grip, str(r)):
            print("render failed", label)
            continue
        p, m = fit_photo(name, box), fit_render(r)
        pair = Image.new("RGB", (p.width * 2 + 6, H + 28), (40, 40, 40))
        pair.paste(p, (0, 28))
        pair.paste(m, (p.width + 6, 28))
        d = ImageDraw.Draw(pair)
        d.text((6, 6), f"{label}  photo | model  az={az}", fill=(255, 255, 0))
        # 對齊參考線：每 10% 身高一條
        for k in range(1, 10):
            y = 28 + int(H * (0.02 + 0.96 * k / 10) / 1.04 * 1.04)
            d.line([(0, y), (pair.width, y)], fill=(255, 60, 60) if k == 5 else (90, 90, 90), width=1)
        cols.append(pair)
    W = sum(c.width for c in cols) + 4 * (len(cols) - 1)
    sheet = Image.new("RGB", (W, H + 28), (0, 0, 0))
    x = 0
    for c in cols:
        sheet.paste(c, (x, 0))
        x += c.width + 4
    sheet.save(out)
    print("saved", out, sheet.size)


if __name__ == "__main__":
    main()
