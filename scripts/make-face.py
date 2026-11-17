"""產生 3D 臉部投影貼圖：轉正、對齊五官、自然的白皙精修。
用法：python scripts/make-face.py <照片> <輸出.png>
五官座標（原圖像素）寫在下面 LEFT_EYE / RIGHT_EYE / LIPS。
方向：白皙、有血色、自然的女性臉（不要死白、不要加深眼窩）。
"""
import sys

import cv2
import numpy as np

LEFT_EYE, RIGHT_EYE, LIPS = (457, 500), (537, 487), (513, 583)  # S__345784336
# 3D 模型臉部矩形中，眼睛中點、嘴唇的位置（由上往下的比例）與寬高比
EYE_T, LIP_T, ASPECT = 0.275, 0.64, 0.708
OUT_W, OUT_H = 420, 594
SKIN = np.array([232, 234, 248], np.float32) / 255  # BGR：白皙、微粉、略冷的膚色

src, dst = sys.argv[1], sys.argv[2]
im = cv2.imread(src)
le, re, lp = map(np.array, (LEFT_EYE, RIGHT_EYE, LIPS))
mid = (le + re) / 2
ang = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
M = cv2.getRotationMatrix2D(tuple(mid), ang, 1.0)
im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LANCZOS4)
lp2 = M @ np.array([lp[0], lp[1], 1.0])
H = (lp2[1] - mid[1]) / (LIP_T - EYE_T)
W = H * ASPECT
cx = mid[0] * 0.6 + lp2[0] * 0.4
x0, y0 = cx - W / 2, mid[1] - EYE_T * H
A = np.float32([[OUT_W / W, 0, -x0 * OUT_W / W], [0, OUT_H / H, -y0 * OUT_H / H]])
face = cv2.warpAffine(im, A, (OUT_W, OUT_H), flags=cv2.INTER_LANCZOS4)

# 1. 輕度降噪（保留五官）
den = cv2.fastNlMeansDenoisingColored(face, None, 5, 5, 7, 21)
den = cv2.bilateralFilter(den, 7, 22, 7)
f = den.astype(np.float32) / 255

# 2. 提亮暗部（照片中臉在帽沿陰影下）
lum = f.mean(2, keepdims=True)
lift = np.clip(lum, 1e-3, 1) ** 0.62 / np.clip(lum, 1e-3, 1)
f = np.clip(f * np.minimum(lift, 2.4) * 1.08, 0, 1)

# 3. 膚色：往白皙微粉靠，保留原本的明暗（立體感）
hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
hue, sat, val = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
lips = (((hue < 18) | (hue > 335)) & (sat > 0.32) & (val > 0.15)).astype(np.float32)
lips = cv2.GaussianBlur(lips, (0, 0), 1.5)
skin_w = np.clip((val - 0.25) / 0.25, 0, 1) * (1 - lips)  # 頭髮、瞳孔等暗處不動
L = f.mean(2, keepdims=True)
L2 = np.clip(0.25 + L * 0.85, 0, 1)  # 整體再亮一些
toned = SKIN * (L2 / SKIN.mean()) * 0.8 + f * 0.2
f = f * (1 - skin_w[..., None] * 0.85) + toned * skin_w[..., None] * 0.85

# 4. 淡淡的腮紅（臉頰）
yy, xx = np.mgrid[0:OUT_H, 0:OUT_W].astype(np.float32)
blush = np.zeros((OUT_H, OUT_W), np.float32)
for bx in (0.27, 0.73):
    blush += np.exp(-(((xx - bx * OUT_W) / (OUT_W * 0.1)) ** 2 + ((yy - 0.47 * OUT_H) / (OUT_H * 0.06)) ** 2))
f = f * (1 - blush[..., None] * 0.08) + np.array([0.72, 0.72, 0.95]) * blush[..., None] * 0.08

# 5. 紅唇：自然的正紅，保留唇紋
lipcol = np.array([0.2, 0.12, 0.55], np.float32)  # BGR 酒紅
lip_shade = (L / max(float(L[lips > 0.5].mean()) if (lips > 0.5).any() else 0.4, 0.05))  # 保留唇的明暗與唇紋
f = f * (1 - lips[..., None] * 0.55) + np.clip(lipcol * lip_shade, 0, 1) * lips[..., None] * 0.55

# 6. 輕微銳化
blur = cv2.GaussianBlur(f, (0, 0), 1.6)
f = np.clip(f + (f - blur) * 0.45, 0, 1)

# 柔邊遮罩：只留臉部
mask = np.zeros((OUT_H, OUT_W), np.float32)
cv2.ellipse(mask, (OUT_W // 2, int(OUT_H * 0.54)), (int(OUT_W * 0.38), int(OUT_H * 0.42)), 0, 0, 360, 1, -1)
mask = cv2.GaussianBlur(mask, (0, 0), 22)
cv2.imwrite(dst, np.dstack([f * 255, mask * 255]).astype(np.uint8))
cv2.imwrite(dst.replace(".png", "-preview.jpg"), (f * 255).astype(np.uint8))
print("saved", dst)

# 取樣臉頰膚色（給頭、頸、手臂與 AI 貼圖的皮膚統一用）
cheeks = np.concatenate([f[int(OUT_H * 0.45):int(OUT_H * 0.55), int(OUT_W * 0.2):int(OUT_W * 0.32)].reshape(-1, 3),
                         f[int(OUT_H * 0.45):int(OUT_H * 0.55), int(OUT_W * 0.68):int(OUT_W * 0.8)].reshape(-1, 3)])
b, g_, r = np.median(cheeks, 0)
print("skin_rgb", round(float(r), 3), round(float(g_), 3), round(float(b), 3), "hex #%02x%02x%02x" % (int(r * 255), int(g_ * 255), int(b * 255)))
