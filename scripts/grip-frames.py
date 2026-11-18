"""把 AI 生成的「雙手抓殼」影片轉成網頁用的畫格序列（dist/media/grip/）。

用法（在專案資料夾）：
  python scripts/grip-frames.py <影片1.mp4> [影片2.mp4 ...] [--fps 16] [--drop 28-43]

- --drop a-b：剪掉輸出的第 a～b 格（AI 生成壞掉的片段，例如手臂一開始是白色的）。
  剪接點會插入 4 張「變暗再亮回來」的轉場畫格，蓋住前後兩格的跳動（不用白畫面）。

- 起始畫面：video-src/grip/keyframe-720x1280.png（網頁最後一章同一個構圖），
  影片必須用它當第一格生成（圖生影片），3D 淡入影片時才會對齊。
- 多段影片會依序接起來（第二段用第一段最後一格當起始畫面生成）。
- 影片若不是 9:16，會先縮放到與起始畫面等寬、再置中裁切，與 Wan 的處理方式相同。
- --sharp-eyes：眼神銳利化（逐格找臉 → 眼睛一帶加銳利與對比）。
- --no-chin-line：修掉下巴那條白色細線（帽兜細邊）。
需要：pip install opencv-python numpy
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "media" / "grip"
W = 480  # 輸出寬度
SRC_ASPECT = 720 / 1280  # 起始畫面寬高比


def face_box(img):
    """臉框：畫面上半部最大的膚色連通區塊（閉運算補上眼、嘴），只取臉的高度。"""
    H = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sk = ((hsv[..., 0] < 22) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 40) & (hsv[..., 2] > 70)
    sk = sk.astype(np.uint8) * 255
    sk[int(H * 0.45):] = 0
    sk = cv2.morphologyEx(sk, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(sk)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h = st[i, :4]
    return x, y, w, min(h, int(w * 1.35))


def skin_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sk = ((hsv[..., 0] < 22) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 40) & (hsv[..., 2] > 70)
    return cv2.morphologyEx(sk.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def remove_chin_line(img, debug=False):
    """下巴那條白色細線（帽兜細邊，沿著臉頰繞到下巴）：在下巴（嘴唇以下）與臉頰外緣找比周圍亮的細線（top-hat），用修補補掉。
    鼻子、臉頰中央的高光不在搜尋範圍內，不會被誤補。"""
    b = face_box(img)
    if b is None:
        return img
    x, y, w, h = b
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    m = (top > 25) & (g > 115) & (sat < 120)  # 比周圍亮的細線（米白色也算；紅唇飽和度高不算）
    H, W = img.shape[:2]
    reg = np.zeros_like(m)
    # 下巴：下唇高光以下（從臉框 86% 起）
    reg[min(H, y + int(h * 0.86)):min(H, y + int(h * 1.2)), max(0, x - int(w * 0.1)):min(W, x + w + int(w * 0.1))] = True
    # 臉頰兩側：只取外緣，避開鼻子
    ys, ye = min(H, y + int(h * 0.45)), min(H, y + int(h * 0.85))
    reg[ys:ye, max(0, x - int(w * 0.15)):min(W, x + int(w * 0.22))] = True
    reg[ys:ye, max(0, x + int(w * 0.78)):min(W, x + w + int(w * 0.15))] = True
    m = (m & reg).astype(np.uint8) * 255
    # 只留「細長的線」：外框夠長、填滿比例低（弧線）；實心的亮塊（下唇下方的高光、帽兜一角）不動
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        bw, bh, area = st[i, 2], st[i, 3], st[i, 4]
        if max(bw, bh) >= max(10, w * 0.15) and area / float(bw * bh) < 0.45:
            keep[lab == i] = 255
    m = keep
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    if debug:
        return m
    return cv2.inpaint(img, m, 4, cv2.INPAINT_TELEA)


def sharpen_eyes(img):
    b = face_box(img)
    if b is None:
        return img
    x, y, w, h = b
    # 眼睛一帶：臉框高度 22%～52%
    y0, y1 = y + int(h * 0.22), y + int(h * 0.52)
    x0, x1 = max(0, x - 4), x + w + 4
    roi = img[y0:y1, x0:x1].astype(np.float32)
    blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
    sharp = roi + (roi - blur) * 1.4  # 銳利化
    lum = sharp.mean(2, keepdims=True)
    sharp = lum + (sharp - lum) * 1.05
    sharp = np.where(lum < 90, sharp * 0.82, sharp)  # 瞳孔、睫毛更深，眼神更利
    # 羽化遮罩：橢圓帶狀
    hh, ww = roi.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    m = np.clip(1.3 - np.sqrt(((xx - ww / 2) / (ww * 0.5)) ** 2 + ((yy - hh / 2) / (hh * 0.55)) ** 2), 0, 1)[..., None]
    res = roi * (1 - m) + sharp * m
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(res, 0, 255).astype(np.uint8)
    return out


def main():
    argv = sys.argv[1:]
    drop = set()
    if "--drop" in argv:
        a, b = map(int, argv[argv.index("--drop") + 1].split("-"))
        drop = set(range(a, b + 1))
        del argv[argv.index("--drop"):argv.index("--drop") + 2]
    if "--fps" in argv:
        del argv[argv.index("--fps"):argv.index("--fps") + 2]
    eyes = "--sharp-eyes" in argv
    chin = "--no-chin-line" in argv
    args = [a for a in argv if not a.startswith("--")]
    fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 16
    if not args:
        print(__doc__)
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.jpg"):
        f.unlink()
    n, h_out, k = 0, None, 0
    cut_pending, last, flash_at = False, None, None
    for v in args:
        cap = cv2.VideoCapture(v)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, round(src_fps / fps))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0 and (k := k + 1) - 1 in drop:
                cut_pending = True
                i += 1
                continue
            if i % step == 0:
                h, w = fr.shape[:2]
                nh = round(h * W / w)
                fr = cv2.resize(fr, (W, nh), interpolation=cv2.INTER_AREA)
                h_out = h_out or min(nh, round(W / SRC_ASPECT))
                o = max(0, (nh - h_out) // 2)
                fr = fr[o:o + h_out]
                if cut_pending and last is not None:
                    # 剪接點：短暫變暗再亮回來（呼應「暗」），蓋住前後兩格的跳動；不用白畫面
                    flash_at = flash_at if flash_at is not None else n + 2
                    for t in (0.15, 0.4, 0.6, 0.85):
                        mix = cv2.addWeighted(last, 1 - t, fr, t, 0).astype(np.float32)
                        dim = np.sin(np.pi * (0.1 + 0.8 * t)) * 0.8
                        mix = mix * (1 - dim)
                        cv2.imwrite(str(OUT / f"{n:03d}.jpg"), np.clip(mix, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 84])
                        n += 1
                if chin:
                    fr = remove_chin_line(fr)
                if eyes:
                    fr = sharpen_eyes(fr)
                cut_pending, last = False, fr
                cv2.imwrite(str(OUT / f"{n:03d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 84])
                n += 1
            i += 1
    (OUT / "meta.json").write_text(json.dumps({"count": n, "w": W, "h": h_out, "fps": fps, "srcAspect": SRC_ASPECT, "flashAt": flash_at}))
    print(f"saved {n} frames ({W}x{h_out}) → {OUT}")


if __name__ == "__main__":
    main()
