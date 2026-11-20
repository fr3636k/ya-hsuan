"""把抓殼影片裡清楚的臉，投影成 3D 模型臉部的貼圖（讓 3D 與影片是同一張臉）。

用法：python scripts/face-decal.py <影片.mp4> <格數> dist/textures/face-video.png
- 影片是用最後一章的起始畫面（viewer 鏡頭 cam=0,3,1.7,0.72、720x1280）生成，Wan 輸出 480x832（等寬縮放、上下裁切）。
- 對臉部範圍內每個 3D 點（z 取臉部曲面擬合），用同一個鏡頭投影到影片像素取色 → 正面投影貼圖。
- 輸出 RGBA：A 為臉部羽化遮罩（眼、鼻、嘴、臉頰），範圍與 dist/js/model-config.js 的 face.rect 一致。
"""
import json
import sys

import cv2
import numpy as np

H = 1.75
RECT = (-0.075, 1.43, 0.075, 1.69)  # 公尺：x0, y0, x1, y1
FIT = (0.074, 0.152, 0.225, -14.28, -1.947, -1.553)  # 臉部曲面 z = f(x, y-1.56)（process-model.py 印出的擬合）
TEX_W = 256


def zf(x, y):
    c = FIT
    yc = y - 1.56
    return c[0] + c[1] * x + c[2] * yc + c[3] * x * x + c[4] * yc * yc + c[5] * x * yc


def project(P):
    """viewer.html?cam=0,3,1.7,0.72 在 720x1280（直式：距離 ×1.25、fov 40）的投影 → 影片 480x832 像素"""
    d = 1.7 * 1.25
    tgt = np.array([0, 0.72 * H, 0])
    el = np.radians(3)
    C = tgt + np.array([0, np.sin(el) * d, np.cos(el) * d])
    f = (tgt - C) / np.linalg.norm(tgt - C)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = P - C
    xc, yc, zc = v @ r, v @ u, v @ f
    t = np.tan(np.radians(20))
    asp = 720 / 1280
    xn, yn = xc / (zc * t * asp), yc / (zc * t)
    px, py = (xn + 1) / 2 * 720, (1 - yn) / 2 * 1280
    s = 480 / 720
    return px * s, py * s - (1280 * s - 832) / 2


def main():
    vid, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cap = cv2.VideoCapture(vid)
    cap.set(1, idx)
    ok, fr = cap.read()
    assert ok, "讀不到影片格"
    # 銳利化（Wan 輸出偏軟）
    blur = cv2.GaussianBlur(fr, (0, 0), 1.0)
    fr = cv2.addWeighted(fr, 1.6, blur, -0.6, 0)
    x0, y0, x1, y1 = RECT
    tex_h = int(round(TEX_W * (y1 - y0) / (x1 - x0)))
    xs = np.linspace(x0, x1, TEX_W)
    ys = np.linspace(y1, y0, tex_h)  # 貼圖上方 = 高處
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, zf(X, Y)], -1).reshape(-1, 3)
    vx, vy = project(P)
    mx = vx.reshape(tex_h, TEX_W).astype(np.float32)
    my = vy.reshape(tex_h, TEX_W).astype(np.float32)
    rgb = cv2.remap(fr, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # 羽化遮罩：眼、鼻、嘴、臉頰（橢圓），避開帽沿與下巴邊線
    cx, cy, rx, ry = 0.0, 1.565, 0.05, 0.088
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    a = np.clip((1.0 - d) / 0.25, 0, 1)
    rgba = np.dstack([rgb, (a * 255).astype(np.uint8)])
    cv2.imwrite(out, rgba)
    print(json.dumps({"rect_rel": [round(v / H, 5) for v in RECT], "size": [TEX_W, tex_h]}))
    # 檢查用：影片上的取樣範圍
    corners = np.array([[x0, y0, zf(x0, y0)], [x1, y1, zf(x1, y1)]])
    print("sample px", [tuple(np.round(project(c), 1)) for c in corners])


if __name__ == "__main__":
    main()
