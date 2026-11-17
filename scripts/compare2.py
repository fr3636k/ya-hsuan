"""照片 vs 模型：全身 + 細節特寫，一次渲染（開發用）。
用法：python scripts/compare2.py <輸出.png> [grip=0] [light=dim] [set=all|full|detail|grip]
需要 node serve.mjs 與 Chrome。輸出為兩列的並排比對圖。
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
W, H = 400, 640
# (標籤, 照片, 照片框 x0,y0,x1,y1, 鏡頭 az,el,dist,ty, 全身?)
FULL = [
    ("正面 358", "S__147939358_0.jpg", (300, 315, 830, 1380), (0, 2, 5.4, 0.5), True),
    ("側A 359", "S__147939359_0.jpg", (360, 30, 820, 1465), (90, 4, 5.4, 0.5), True),
    ("側B 360", "S__147939360_0.jpg", (270, 20, 750, 1455), (270, 4, 5.4, 0.5), True),
    ("背(人台)354", "S__147939354_0.jpg", (280, 85, 820, 1405), (180, 4, 5.4, 0.5), True),
    ("斜(人台)352", "S__147939352_0.jpg", (330, 40, 780, 1450), (40, 4, 5.4, 0.5), True),
    ("側(戶外)334", "S__345784334_0.jpg", (300, 270, 660, 1420), (300, 2, 5.4, 0.5), True),
]
DETAIL = [
    ("上身正面 361", "S__147939361_0.jpg", (250, 180, 860, 1150), (0, 4, 2.4, 0.74), False),
    ("前襟束結 339", "S__345784339.jpg", (120, 560, 1000, 1478), (0, 10, 1.25, 0.6), False),
    ("帽沿 337", "S__345784337_0.jpg", (60, 200, 1050, 1250), (0, 4, 1.3, 0.86), False),
    ("裙口 358", "S__147939358_0.jpg", (360, 690, 760, 960), (0, 8, 1.2, 0.53), False),
    ("魚尾 358", "S__147939358_0.jpg", (380, 1180, 760, 1390), (0, 10, 1.2, 0.06), False),
    ("蛹側 360", "S__147939360_0.jpg", (330, 20, 760, 720), (270, 6, 2.0, 0.77), False),
    ("背骨架 354", "S__147939354_0.jpg", (280, 80, 820, 700), (180, 6, 2.0, 0.78), False),
]
MORE = [
    ("側(人台)353", "S__147939353_0.jpg", (360, 130, 770, 1430), (90, 4, 5.4, 0.5), True),
    ("斜前(人台)355", "S__147939355_0.jpg", (320, 80, 700, 1420), (340, 4, 5.4, 0.5), True),
    ("正面(人台)357", "S__147939357_0.jpg", (300, 100, 820, 1440), (0, 4, 5.4, 0.5), True),
    ("正面(戶外)335", "S__345784335_0.jpg", (400, 520, 800, 1478), (0, 6, 4.4, 0.42), False),
    ("正面 364", "S__147939364_0.jpg", (300, 410, 800, 1478), (0, 4, 4.4, 0.45), False),
]
GRIP = [
    ("抓殼 392", "S__147939392.jpg", (30, 20, 225, 300), (0, 4, 2.6, 0.75), False),
    ("撥開(影片)", "_grip_frame.png", None, (20, 4, 2.6, 0.75), False),
]


def render(views, grip, light, out):
    v = ",".join(f"{a}:{e}:{d}:{t}" for a, e, d, t in views)
    url = f"http://127.0.0.1:8788/_compare.html?v={v}&grip={grip}&light={light}"
    for _ in range(5):
        prof = tempfile.mkdtemp()
        try:
            subprocess.run([CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                            "--hide-scrollbars", f"--window-size={W * len(views)},{H}", "--virtual-time-budget=150000",
                            f"--user-data-dir={prof}", f"--screenshot={out}", url],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=400)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(prof, ignore_errors=True)
        if os.path.exists(out) and np.asarray(Image.open(out).convert("L")).mean() > 3:
            return Image.open(out).convert("RGB")
    raise SystemExit("render failed")


def photo_tile(name, box, full):
    p = ROOT / "data" / name
    if name == "_grip_frame.png":
        p = ROOT / "qa" / name
    im = Image.open(p).convert("RGB")
    if box is None:
        box = (0, 0, im.width, im.height)
    x0, y0, x1, y1 = box
    if full:
        h = y1 - y0
        cx = (x0 + x1) / 2
        box = (cx - h * 0.3125 * 1.04, y0 - h * 0.02, cx + h * 0.3125 * 1.04, y1 + h * 0.02)
        return im.crop(tuple(map(int, box))).resize((W, H))
    c = im.crop(box)
    s = min(W / c.width, H / c.height)
    c = c.resize((int(c.width * s), int(c.height * s)))
    t = Image.new("RGB", (W, H), (30, 30, 30))
    t.paste(c, ((W - c.width) // 2, (H - c.height) // 2))
    return t


def model_tile(img, i, full):
    t = img.crop((i * W, 0, (i + 1) * W, H))
    if not full:
        return t
    a = np.asarray(t.convert("L"))
    ys, xs = np.where(a > 40)
    if len(ys) == 0:
        return t
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0
    cx = int(np.median(xs))
    box = (cx - h * 0.3125 * 1.04, y0 - h * 0.02, cx + h * 0.3125 * 1.04, y1 + h * 0.02)
    return t.crop(tuple(map(int, box))).resize((W, H))


def main():
    out = str(Path(sys.argv[1]).resolve())
    grip = sys.argv[2] if len(sys.argv) > 2 else "0"
    light = sys.argv[3] if len(sys.argv) > 3 else "dim"
    which = sys.argv[4] if len(sys.argv) > 4 else "all"
    shots = {"full": FULL, "detail": DETAIL, "grip": GRIP, "more": MORE, "all": FULL + DETAIL}[which]
    img = render([s[3] for s in shots], grip, light, out + ".raw.png")
    pairs = []
    for i, (label, name, box, cam, full) in enumerate(shots):
        pair = Image.new("RGB", (W * 2 + 4, H + 22), (45, 45, 45))
        pair.paste(photo_tile(name, box, full), (0, 22))
        pair.paste(model_tile(img, i, full), (W + 4, 22))
        d = ImageDraw.Draw(pair)
        d.text((4, 4), f"{label} | photo  vs  model", fill=(255, 230, 0))
        if full:
            for k in range(1, 10):
                y = 22 + int(H * (0.02 + 0.96 * k / 10) / 1.04)
                d.line([(0, y), (pair.width, y)], fill=(255, 60, 60) if k == 5 else (100, 100, 100))
        pairs.append(pair)
    per_row = 4
    rows = [pairs[i:i + per_row] for i in range(0, len(pairs), per_row)]
    pw, ph = pairs[0].size
    sheet = Image.new("RGB", (pw * per_row + 4 * (per_row - 1), ph * len(rows) + 4 * (len(rows) - 1)))
    for r, row in enumerate(rows):
        for c, p in enumerate(row):
            sheet.paste(p, (c * (pw + 4), r * (ph + 4)))
    sheet.save(out)
    os.remove(out + ".raw.png")
    print("saved", out, sheet.size)


if __name__ == "__main__":
    main()
