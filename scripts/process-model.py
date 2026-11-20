"""把 AI 生成的 GLB 處理成網頁用的模型。兩個版本：

V2（目前使用，Tripo）：形狀與臉保留原貌，只提亮白布、去污、做發光遮罩與 LED
  python scripts/process-model.py models/v2-tripo/raw-tripo.glb <暫存>.glb --tripo
  → Blender 執行 scripts/draco-compress.py <暫存>.glb dist/models/emerge-c.glb（Draco 壓縮）
V1（舊版，TRELLIS）：另外雕塑骨架、蛹形、下擺（SCULPT 參數）
  python scripts/process-model.py models/v1-trellis/raw-trellis2.glb models/v1-trellis/web-emerge-a2.glb [--pink]

處理內容：
  1. 尺寸校正：身高 1.75 m、腳底 y=0、水平置中（直接寫進頂點）
  2. 貼圖：AI 把白布算得偏灰 → 提亮；膚色略提亮；--pink 時把粉紅/洋紅 LED 光轉白
  3. 發光遮罩：只有蛹的白布會發光，存成 emissiveTexture
  4. 材質改成布料（金屬度 0）
  5. LED 光點位置：在發光布面上取樣，輸出 <輸出名>.leds.json
需要：pip install trimesh pillow numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

HEIGHT = 1.75
# 調整參數（與 dist/model-config.js 的說明對應）
FABRIC_MAX_SAT = 0.13  # 布面：飽和度上限（臉、手偏暖色不處理）
FABRIC_MIN_VAL = 0.22  # 布面：亮度下限（排除黑髮）
FABRIC_LIFT = (0.52, 0.95)  # 輸出亮度 = 0.52 + (原亮度 - 0.3) × 0.95
SKIN_LIFT = 1.15
SKIN_RGB = (0.83, 0.72, 0.665)  # 與 dist/model-config.js 的 skin 相同（#d4b8aa）
MASK_MIN_VAL = 0.3
GLOW_BAND = (0.53, 1.0)  # 發光的高度範圍（相對身高）
LED_COUNT = 260


# ===== 共用 =====


def hsv(a):
    mx = a.max(2)
    mn = a.min(2)
    d = mx - mn
    sat = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) * 60
    h = np.where(d > 0, h, 0)
    return h, sat, mx


def cv2_dilate(mask, r=2):
    """布林遮罩擴張 r 像素（不依賴 OpenCV）。"""
    m = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            m |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return m


def clamp01(x):
    return np.clip(x, 0, 1)


# ===== V1（TRELLIS）：貼圖處理與雕塑 =====

HEAD_Y = 0.8  # 相對身高：以上才可能有頭髮
SKIRT_Y = 0.52  # 相對身高：以下是針織魚尾裙


def texel_height(mesh, size):
    """把每個貼圖像素對應到模型高度（相對身高），沒用到的像素為 NaN。"""
    from PIL import ImageDraw

    W = H = size
    img = Image.new("F", (W, H), float("nan"))
    d = ImageDraw.Draw(img)
    uv = mesh.visual.uv
    fy = mesh.vertices[mesh.faces][:, :, 1].mean(1) / HEIGHT
    tri = uv[mesh.faces]
    xs = (tri[:, :, 0] % 1) * (W - 1)
    ys = (1 - (tri[:, :, 1] % 1)) * (H - 1)
    for i in range(len(fy)):
        d.polygon([(xs[i, 0], ys[i, 0]), (xs[i, 1], ys[i, 1]), (xs[i, 2], ys[i, 2])], fill=float(fy[i]))
    return np.asarray(img)


def process_texture(img, pink_to_white, height=None):
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    h, s, v = hsv(a)
    pink = np.zeros_like(v)
    if pink_to_white:
        pink = ((h > 285) & (h < 345) & (s > 0.18) & (v > 0.35)) * clamp01((s - 0.18) / 0.25)
        lum = np.minimum(1, v * 1.08)[..., None]
        a = a + (lum - a) * pink[..., None]
        h, s, v = hsv(a)
    fabric = clamp01((FABRIC_MAX_SAT - s) / 0.05) * clamp01((v - FABRIC_MIN_VAL) / 0.08)
    if height is not None:
        # 頭部以下沒有頭髮：深灰污漬也當成布面提亮；裙子（腰以下）連略帶黃的米白也算
        below_head = height < HEAD_Y
        fabric = np.where(below_head, clamp01((FABRIC_MAX_SAT + 0.04 - s) / 0.05), fabric)
        skirt = height < SKIRT_Y
        fabric = np.where(skirt, clamp01((0.3 - s) / 0.06), fabric)
        v = np.where(below_head, np.maximum(v, 0.3 + (v - 0.3) * 0.5), v)  # 壓縮暗部，污漬變淡
    v0 = hsv(a)[2]
    gain = np.minimum(1, FABRIC_LIFT[0] + (v - 0.3) * FABRIC_LIFT[1]) / np.maximum(v0, 0.05)
    k = 1 + (gain - 1) * fabric
    a = a * k[..., None]
    a[..., 2] *= 1 + 0.01 * fabric  # 一點點冷白
    skin = (h > 8) & (h < 40) & (s > 0.14) & (s < 0.6) & (v > 0.15)
    a[skin] *= np.array([SKIN_LIFT, SKIN_LIFT * 1.03, SKIN_LIFT * 1.06], np.float32)
    a = np.clip(a, 0, 1)

    if height is not None:
        # 去髒污：布面亮度 = 乾淨的底色 + 原本的細皺褶（高頻），大塊色斑被抹平
        from PIL import ImageFilter

        L = a.mean(2)
        Limg = Image.fromarray((L * 255).astype(np.uint8))
        fine = np.asarray(Limg.filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255
        detail = L - fine  # 皺褶細節
        below_head = height < HEAD_Y
        skirt = height < SKIRT_Y
        # 頭部以下的暗色暖斑（AI 殘留的手、污漬）也當成布
        stain = below_head & (h > 5) & (h < 45) & (v0 < 0.4) & ~skirt
        # 灰綠、灰黃的暗斑（裙口內側等），飽和度不高的都清掉
        stain |= below_head & (h >= 45) & (h < 200) & (s < 0.4) & (v0 < 0.6)
        clean = (below_head & ((fabric > 0.3) | stain)) | ((height > 0.749) & (height < 0.751))
        base = np.where(skirt, 0.86, 0.8)  # 蛹的布 0.8（太白會在燈光下過曝、看不到骨架）
        newL = np.clip(base + detail * np.where(skirt, 1.4, 0.0), 0, 1)  # 蛹：均勻白（皺褶交給 shader）
        tint = np.where(skirt[..., None], np.array([1.0, 0.985, 0.95]), np.array([0.985, 0.99, 1.0]))
        cleaned = newL[..., None] * tint
        w = clean.astype(np.float32)[..., None]
        a = a * (1 - w) + cleaned * w
        # 上半身的皮膚：統一成與臉相同的白皙膚色（scripts/make-face.py 取樣）；頭髮保留黑色
        skin_up = (height >= SKIRT_Y) & ~clean & (h > 0) & (h < 45) & (s > 0.12) & (v0 > 0.18)
        skin_col = np.array(SKIN_RGB, np.float32)  # PIL 讀進來是 RGB
        Ls = np.clip(1.0 + detail * 1.5, 0.85, 1.1)[..., None]
        wu = skin_up.astype(np.float32)[..., None]
        a = a * (1 - wu) + np.clip(skin_col * Ls, 0, 1) * wu
        # 頭部附近 AI 殘留的黑髮（頭改成人台後會變成黑斑）→ 改成帽沿內側的白布
        hair = (height >= HEAD_Y) & (v0 < 0.3)
        wh = cv2_dilate(hair).astype(np.float32)[..., None]
        a = a * (1 - wh) + np.clip(0.86 + detail[..., None] * 1.2, 0, 1) * np.array([0.985, 0.99, 1.0]) * wh
    a = np.clip(a, 0, 1)
    mask = np.maximum(fabric * clamp01((v - MASK_MIN_VAL) / 0.15), pink)
    if height is not None:
        mask = np.where(clean & ~skirt, 1.0, mask)  # 清乾淨的蛹布面全部可發光
    color = Image.fromarray((a * 255).astype(np.uint8))
    emissive = Image.fromarray((np.repeat(mask[..., None], 3, 2) * 255).astype(np.uint8))
    return color, emissive, mask


SCULPT = {
    # 蛹往後延伸：在 back_y（相對身高）附近、背面（z<0）往後推 back_extend 公尺，
    # 讓殼比臀部更凸（照片側面）
    "back_extend": 0.16,
    "back_y": 0.8,
    "back_spread": 0.11,
    # 裙口：側面看應該窄（橢圓，左右寬、前後窄）
    "rim_z_scale": 0.75,
    "rim_y": (0.47, 0.63),
    # 蛹的上半部往前傾（側面照片中花苞低頭的姿態）：頂端往前移 lean 公尺
    "lean": 0.06,
    "lean_from": 0.55,
    # 蛹的背面與側面撫平的次數（像布繃在骨架上的燈籠殼）
    "smooth": 0,
    # 帽沿往下折：前方中央的帽沿往下拉（照片帽沿都折到眉毛）
    "brim_drop": 0.03,
    # 區域平滑：(相對身高 起, 迄, 次數)。魚尾荷葉褶的尖縫、裙口的皺團
    "smooth_regions": [(0.54, 0.62, 120)],
    "post_smooth_regions": [(0.55, 0.74, 120)],  # 腰部交界與蛹下緣的小撕裂片
    # 下擺用平滑殼（相對身高範圍）與方位角平滑（格，1 格 3.75°；2 格可保留約 8 個大褶）
    "hem_hull": (0.005, 0.14),
    "hem_sigma_t": 2.0,
    # 下擺大褶：(褶數, 振幅, 起始高度[相對身高])
    "hem_flutes": (8, 0.09, 0.12),
    # 繃緊的殼：把蛹的外層投到平滑殼上
    "hull": True,
    "hull_from": 0.57,      # 從腰（相對身高）開始
    "hull_sigma_t": 5.0,    # 方位角方向的平滑（格，1 格 = 3.75°）
    "hull_sigma_y": 2.5,    # 高度方向的平滑（格）
    "hull_scale": 0.985,    # 殼略收一點，避免比原本膨脹
    "hull_front": 0.0,      # 正面開口（±40°）完全不動：避免把開口後的身體拉成碎片
    "hull_round": 0.32,     # 上半部（肩到頂）飽滿成洋蔥形
    "hull_round_y": 0.8,    # 飽滿中心（殼高度比例）
    "hull_round_w": 0.16,
    "hull_top_taper": 0.6,  # 頂端收尖的程度（背面人台照是尖頂、正面真人照帽頂較圓 → 折衷）
    "hull_front_clamp": 1.04, # 正面開口一帶只收回超出殼 4% 以上的凸塊
    "hull_cone": 1.0,       # 下半部填成錐形的程度
    "hull_cone_peak": 0.55,  # 錐形到殼高度的哪裡達到最寬
    "hull_cone_base": 0.32, # 腰部半徑 = 最寬半徑 × 此值
    # 正面看蛹左右要更寬（照片 358 最寬處比模型寬約 24%）：只放大 x，深度不變
    "widen_x": 0.0,  # 已改用 cocoon_widen 曲線
    # 蛹正面寬度曲線（只放大 x、離中線 4 cm 外才作用）：照片 358 在 30% 身高(相對 0.7) 前襟仍寬 155px，
    # 模型只有 75px；最寬處(0.83) 照片 230、模型 215。[(相對身高, 倍率)]
    "cocoon_widen": [(0.59, 1.0), (0.63, 1.15), (0.68, 1.35), (0.72, 1.42), (0.78, 1.4),
                     (0.83, 1.38), (0.9, 1.18), (0.97, 1.0)],
    "widen_y": 0.83,
    "widen_w": 0.10,
    # 裙子正面要更寬（照片 358：裙口 195px、50% 處 175px；模型 125、95px）。
    # 側面寬度已相符 → 只放大 x。[(相對身高, 倍率)]，中間線性內插
    "skirt_widen": [(0.0, 1.25), (0.08, 1.2), (0.15, 1.12), (0.22, 1.2), (0.35, 1.38),
                    (0.45, 1.4), (0.53, 1.45), (0.58, 1.35), (0.61, 1.2), (0.64, 1.0)],  # 依多張正面照（358/361/364/357/335）取中間值  # 裙口上緣收圓，不要平台直角
}


def smooth_region(mesh, y0, y1, iters):
    """在高度範圍 y0~y1（相對身高）內做 Taubin 平滑（柔化尖銳褶縫，不縮小）。"""
    v = mesh.vertices
    key = np.round(v / 1e-5).astype(np.int64)
    uniq, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.ravel()
    n = len(uniq)
    P = np.zeros((n, 3))
    P[weld] = v
    F = weld[mesh.faces]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(np.float64)
    y = P[:, 1] / HEIGHT
    w = (np.clip((y - y0) / 0.02, 0, 1) * np.clip((y1 - y) / 0.02, 0, 1))[:, None]
    for _ in range(iters):
        for lam in (0.55, -0.58):
            acc = np.zeros_like(P)
            np.add.at(acc, e[:, 0], P[e[:, 1]])
            P = P + lam * w * (acc / np.maximum(deg, 1)[:, None] - P)
    mesh.vertices = P[weld]


def smooth_shell(mesh, iters=40):
    """把蛹的背面與側面撫平成繃在骨架上的殼（Taubin 平滑，不會縮小）。
    正面開口、前襟、臉部附近保留原本細節。"""
    v = mesh.vertices
    key = np.round(v / 1e-5).astype(np.int64)
    _, weld, inv_first = None, None, None
    uniq, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.ravel()
    n = len(uniq)
    P = np.zeros((n, 3))
    P[weld] = v
    F = weld[mesh.faces]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(np.float64)
    y = P[:, 1] / HEIGHT
    # 平滑權重：蛹（腰以上）全平滑；正面開口一帶（|x-0.03|<0.14 且 z>0）漸弱
    w = np.clip((y - 0.57) / 0.05, 0, 1)
    front = np.clip(1 - (np.abs(P[:, 0] - 0.03) - 0.1) / 0.06, 0, 1) * np.clip(P[:, 2] / 0.06, 0, 1)
    w *= 1 - 0.9 * front
    w = w[:, None]
    for it in range(iters):
        for lam in (0.5, -0.53):
            acc = np.zeros_like(P)
            np.add.at(acc, e[:, 0], P[e[:, 1]])
            avg = acc / np.maximum(deg, 1)[:, None]
            P = P + lam * w * (avg - P)
    mesh.vertices = P[weld]


def _gauss1d(a, sigma, axis, wrap):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.zeros_like(a)
    for i, w in zip(range(-r, r + 1), k):
        if wrap:
            out += w * np.roll(a, i, axis)
        else:
            idx = np.clip(np.arange(a.shape[axis]) + i, 0, a.shape[axis] - 1)
            out += w * np.take(a, idx, axis)
    return out


def hull_smooth(mesh, cfg, y_range=None):
    """把蛹的外層拉成繃緊的殼（像布繃在骨架上）：
    以高度×方位角分格，取每格最外側半徑並大幅平滑，外層頂點投到這個平滑殼上。
    內層（靠近身體）不動；正面開口一帶保留部分細節。"""
    v = mesh.vertices.copy()
    y = v[:, 1] / HEIGHT
    y0, y1 = y_range if y_range else (cfg["hull_from"], 1.0)
    sel = (y > y0 - 0.02) & (y < y1 + 0.02)
    NY, NT = 48, 96
    yf = np.clip((y - y0) / (y1 - y0) * NY - 0.5, 0, NY - 1.001)  # 連續座標（雙線性內插用）
    yb = np.clip(((y - y0) / (y1 - y0) * NY).astype(int), 0, NY - 1)
    # 每個高度切片的中心（用外層點的中點）
    cx = np.zeros(NY)
    cz = np.zeros(NY)
    for b in range(NY):
        m = sel & (yb == b)
        if m.sum() > 10:
            cx[b] = (v[m, 0].min() + v[m, 0].max()) / 2
            cz[b] = (v[m, 2].min() + v[m, 2].max()) / 2
    cx = _gauss1d(cx, 3, 0, False)
    cz = _gauss1d(cz, 3, 0, False)
    y_lo = np.floor(yf).astype(int)
    fy = yf - y_lo
    CX = cx[y_lo] * (1 - fy) + cx[y_lo + 1] * fy
    CZ = cz[y_lo] * (1 - fy) + cz[y_lo + 1] * fy
    dx = v[:, 0] - CX
    dz = v[:, 2] - CZ
    r = np.hypot(dx, dz)
    th = np.arctan2(dx, dz)  # 0 = 正面
    tb = ((th + np.pi) / (2 * np.pi) * NT).astype(int) % NT
    R = np.zeros((NY, NT))
    np.maximum.at(R, (yb[sel], tb[sel]), r[sel])
    # 補空格：沿角度用鄰近值
    for _ in range(NT):
        empty = R == 0
        if not empty.any():
            break
        R = np.where(empty, np.maximum(np.roll(R, 1, 1), np.roll(R, -1, 1)), R)
    Rraw = R.copy()  # 未平滑的每格最外側半徑（判斷外層用）
    R = _gauss1d(R, cfg["hull_sigma_t"], 1, True)
    R = _gauss1d(R, cfg["hull_sigma_y"], 0, False)
    # 雙線性內插（避免鋸齒）
    tf = (th + np.pi) / (2 * np.pi) * NT - 0.5
    t_lo = np.floor(tf).astype(int)
    ft = tf - t_lo
    t0, t1 = t_lo % NT, (t_lo + 1) % NT
    Rv = (R[y_lo, t0] * (1 - fy) * (1 - ft) + R[y_lo + 1, t0] * fy * (1 - ft)
          + R[y_lo, t1] * (1 - fy) * ft + R[y_lo + 1, t1] * fy * ft) * cfg["hull_scale"]
    Rbase = (Rraw[y_lo, t0] * (1 - fy) * (1 - ft) + Rraw[y_lo + 1, t0] * fy * (1 - ft)
             + Rraw[y_lo, t1] * (1 - fy) * ft + Rraw[y_lo + 1, t1] * fy * ft)  # 判斷「外層」用未平滑半徑
    # 輪廓圓潤度：讓殼的上半部像洋蔥一樣飽滿（0 = 不改）
    tt = np.clip((y - y0) / (y1 - y0), 0, 1)
    Rv *= 1 + cfg.get("hull_round", 0) * np.exp(-(((tt - cfg.get("hull_round_y", 0.7)) / cfg.get("hull_round_w", 0.18)) ** 2))
    # 頂端收成洋蔥尖頂（不要瓶口般的直邊平頂）
    tp = cfg.get("hull_top_taper", 0)
    if tp:
        Rv *= 1 - tp * (1 - np.sqrt(np.clip(1 - ((tt - 0.86) / 0.16) ** 2, 0, 1))) * (tt > 0.86)
    # 下半部填成錐形（從腰直線張開到最寬處），不要細脖子
    cone = cfg.get("hull_cone", 0)
    if cone:
        Rpk = R.max(0)  # 每個方位角的最大半徑
        peak_t = cfg.get("hull_cone_peak", 0.5)
        frac = cfg.get("hull_cone_base", 0.3) + (1 - cfg.get("hull_cone_base", 0.3)) * np.clip(tt / peak_t, 0, 1)
        Rc = (Rpk[t0] * (1 - ft) + Rpk[t1] * ft) * frac * cfg["hull_scale"]
        Rv = np.where(tt < peak_t, np.maximum(Rv, Rv + (Rc - Rv) * cone * (Rc > Rv)), Rv)
    # 只動外層：r 接近該格最外側的頂點
    outer = np.clip((r / np.maximum(Rbase, 1e-4) - 0.55) / 0.15, 0, 1)
    region = np.clip((y - y0) / 0.05, 0, 1) if not y_range else np.clip((y - y0) / 0.01, 0, 1) * np.clip((y1 - y) / 0.05, 0, 1)
    # 正面開口（方位角 ±55° 內、且在胸口以上）只部分撫平，保留前襟與開口
    front = np.clip(1 - (np.abs(th) - np.radians(40)) / np.radians(25), 0, 1) if not y_range else 0 * th
    w = outer * region * (1 - front * (1 - cfg["hull_front"]))
    # 用法線分辨布的兩面：朝外的面（外布）→ 正好在平滑殼上；朝內的面（內襯）→ 殼往內一個固定厚度。
    # 這樣殼是平滑的、布有一致的厚度，兩層不會穿插（避免 z-fighting 斑點）。
    fn = np.cross(v[mesh.faces[:, 1]] - v[mesh.faces[:, 0]], v[mesh.faces[:, 2]] - v[mesh.faces[:, 0]])
    vn = np.zeros_like(v)
    for c in range(3):
        np.add.at(vn, mesh.faces[:, c], fn)
    facing = (vn[:, 0] * dx + vn[:, 2] * dz) / np.maximum(np.linalg.norm(vn, axis=1) * r, 1e-9)
    target = np.where(facing > 0, Rv, Rv - cfg.get("hull_thick", 0.012))
    newr = r + (target - r) * w
    # 正面開口一帶：不推出、只把超出殼的凸塊收回（側面看前緣要是平順斜線）
    fc = cfg.get("hull_front_clamp", 0)
    if fc:
        lim = np.where(facing > 0, Rv * fc, Rv * fc - cfg.get("hull_thick", 0.012))
        wf = front * outer * region
        newr = np.where((wf > 0) & (newr > lim), newr + (lim - newr) * wf, newr)
    k = newr / np.maximum(r, 1e-6)
    v[:, 0] = np.where(sel, CX + dx * k, v[:, 0])
    v[:, 2] = np.where(sel, CZ + dz * k, v[:, 2])
    mesh.vertices = v


def sculpt(mesh):
    for y0, y1, it in SCULPT.get("smooth_regions", []):
        smooth_region(mesh, y0, y1, it)
    if SCULPT.get("hem_hull"):
        # 下擺：外層拉成平滑殼，蓋住荷葉褶之間的深縫，保留約 8 個大褶
        hcfg = dict(SCULPT, hull_sigma_t=SCULPT["hem_sigma_t"], hull_sigma_y=1.5, hull_scale=0.99,
                    hull_round=0, hull_cone=0)
        hull_smooth(mesh, hcfg, SCULPT["hem_hull"])
    if SCULPT.get("hull"):
        hull_smooth(mesh, SCULPT)
    if SCULPT.get("smooth"):
        smooth_shell(mesh, SCULPT["smooth"])
    v = mesh.vertices.copy()
    y = v[:, 1] / HEIGHT
    z = v[:, 2]
    k = SCULPT
    if k["back_extend"]:
        band = np.exp(-(((y - k["back_y"]) / k["back_spread"]) ** 2))
        back = np.clip(-z / 0.2, 0, 1) ** 1.2  # 越靠背面推越多
        v[:, 2] = z - k["back_extend"] * band * back
    if k.get("brim_drop"):
        wx = np.exp(-(((v[:, 0] - 0.015) / 0.09) ** 2))
        wy = np.clip((y - 0.915) / 0.02, 0, 1) * np.clip((1.0 - y) / 0.02, 0, 1)
        wz = np.clip((v[:, 2] - 0.03) / 0.04, 0, 1)
        wb = wx * wy * wz
        v[:, 1] -= k["brim_drop"] * wb
        v[:, 2] += 0.012 * wb
    if k.get("hem_flutes"):
        # 魚尾下擺：約 8 個圓潤大褶（照片 358），抽褶線 0、往地面最大
        n, amp, ytop = k["hem_flutes"]
        sel = y < ytop
        cx0 = np.median(v[sel, 0]); cz0 = np.median(v[sel, 2])
        dx = v[:, 0] - cx0; dz = v[:, 2] - cz0
        th = np.arctan2(dx, dz)
        w = np.clip((ytop - y) / (ytop * 0.7), 0, 1) ** 1.3 * sel
        f = 1 + amp * w * np.cos(n * th + 0.4)
        v[:, 0] = cx0 + dx * f
        v[:, 2] = cz0 + dz * f
    if k.get("skirt_widen"):
        ys, fs = zip(*k["skirt_widen"])
        f = np.interp(y, ys, fs)
        v[:, 0] = 0.01 + (v[:, 0] - 0.01) * f
    if k.get("cocoon_widen"):
        ys, fs = zip(*k["cocoon_widen"])
        f = np.interp(y, ys, fs, left=1.0, right=1.0)
        side = np.clip((np.abs(v[:, 0] - 0.01) - 0.04) / 0.08, 0, 1)
        v[:, 0] = 0.01 + (v[:, 0] - 0.01) * (1 + (f - 1) * side)
    if k.get("widen_x"):
        wgt = k["widen_x"] * np.exp(-(((y - k["widen_y"]) / k["widen_w"]) ** 2))
        cxm = 0.01  # 蛹的中線
        # 只加寬外側的殼：離中線 8 cm 以內（頭、身體、開口）不動，往外漸進
        side = np.clip((np.abs(v[:, 0] - cxm) - 0.08) / 0.1, 0, 1)
        v[:, 0] = cxm + (v[:, 0] - cxm) * (1 + wgt * side)
    if k["lean"]:
        t = np.clip((y - k["lean_from"]) / (1 - k["lean_from"]), 0, 1) ** 2
        v[:, 2] = v[:, 2] + k["lean"] * t
    if k["rim_z_scale"] != 1:
        y0, y1 = k["rim_y"]
        t = np.clip((y - y0) / 0.04, 0, 1) * np.clip((y1 - y) / 0.04, 0, 1)
        v[:, 2] = v[:, 2] * (1 - t * (1 - k["rim_z_scale"]))
    mesh.vertices = v
    # 收尾平滑：各步驟交界處（腰部）留下的小撕裂片
    for y0, y1, it in SCULPT.get("post_smooth_regions", []):
        smooth_region(mesh, y0, y1, it)


# ===== V2（Tripo）：貼圖處理 =====


def texel_xyz(mesh, size):
    """每個貼圖像素對應的 3D 位置 (H, W, 3)，沒用到的像素為 NaN。"""
    from PIL import ImageDraw

    out = []
    uv = mesh.visual.uv
    tri = uv[mesh.faces]
    xs = (tri[:, :, 0] % 1) * (size - 1)
    ys = (1 - (tri[:, :, 1] % 1)) * (size - 1)
    cen = mesh.vertices[mesh.faces].mean(1)
    for c in range(3):
        img = Image.new("F", (size, size), float("nan"))
        d = ImageDraw.Draw(img)
        for i in range(len(cen)):
            d.polygon([(xs[i, 0], ys[i, 0]), (xs[i, 1], ys[i, 1]), (xs[i, 2], ys[i, 2])], fill=float(cen[i, c]), outline=float(cen[i, c]))  # 連邊線一起畫：細長三角形（眼皮、鼻孔）才不會漏掉
        out.append(np.asarray(img))
    return np.stack(out, -1)


# Tripo 模型：臉保留原貌（帽兜內的臉、頸、頭髮），其餘布面提亮去污
FACE_BOX = {"x": 0.085, "y": (1.2, 1.7), "z": (-0.02, 0.15)}  # 眼睛約 y 1.61、眉毛 1.63


def pad_islands(img, used, iters=16):
    """把已用像素的顏色往未用像素擴張 iters 次（每次 1 像素，取鄰居平均）。"""
    img = img * used[..., None]
    w = used.copy()
    for _ in range(iters):
        acc = np.zeros_like(img)
        cnt = np.zeros_like(w)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            acc += np.roll(np.roll(img * w[..., None], dy, 0), dx, 1)
            cnt += np.roll(np.roll(w, dy, 0), dx, 1)
        grow = (w == 0) & (cnt > 0)
        img = np.where(grow[..., None], acc / np.maximum(cnt, 1e-6)[..., None], img)
        w = np.where(grow, 1.0, w)
    # 剩下的空白填成布的白
    return np.where(w[..., None] > 0, img, 0.82)


FACE_LIFT = 1.7  # 臉的提亮曲線 1-(1-x)^k：k 越大越亮（AI 影片畫格用同一條曲線，見 scripts/grip-frames.py）


def face_lift(a, k=FACE_LIFT):
    lifted = 1 - (1 - np.clip(a, 0, 1)) ** k
    # 稍微保留飽和度：亮度提升後顏色會變淡，按原本色差補回一點
    mean = lifted.mean(-1, keepdims=True)
    return np.clip(mean + (lifted - mean) * 1.15, 0, 1)


def blend_face_seams(out, xyz, region, voxel=0.003, ring=4):
    """臉部貼圖接縫：兩個區塊在臉上的交界顏色接不起來（中線、脖子出現細線）。
    找出區塊交界（相鄰像素的 3D 位置跳動 > 5 mm），交界附近 ring 像素改成「同一個 3D 位置（3 mm 體素）所有像素的平均色」，
    交界兩側用同一個顏色 → 接縫消失。"""
    P = np.nan_to_num(xyz, nan=1e3)
    jump = np.zeros(P.shape[:2], bool)
    for ax in (0, 1):
        d = np.linalg.norm(np.diff(P, axis=ax), axis=-1) > 0.005
        if ax == 0:
            jump[:-1] |= d
            jump[1:] |= d
        else:
            jump[:, :-1] |= d
            jump[:, 1:] |= d
    near = cv2_dilate(jump & region, ring) & region & ~np.isnan(xyz[..., 0])
    sel = region & ~np.isnan(xyz[..., 0])
    vk = np.floor(P[sel] / voxel).astype(np.int64)
    _, inv = np.unique(vk, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv).astype(np.float64)
    avg = np.stack([np.bincount(inv, weights=out[sel][:, c]) for c in range(3)], 1) / cnt[:, None]
    col = np.zeros_like(out)
    col[sel] = avg[inv]
    w = near.astype(np.float32)[..., None]
    print("face seam texels blended", int(near.sum()))
    return out * (1 - w) + col * w


# V2 造型修正（作者回饋）：裙子垂直往下、腰部交接處整理成微露的腰、整體皺褶撫平
RESHAPE = {
    "skirt": (0.12, 0.90),     # 裙子拉直的範圍（公尺）：魚尾上方到臀部骨架下緣（臀部骨架與魚尾荷葉邊不動）
    "skirt_ref": (0.80, 0.86), # 以臀部骨架正下方這一段的半徑為準（0.86 以上會量到骨架往下垂的邊），往下維持同樣寬度（垂直往下）
    "waist": (1.095, 1.178),   # 露腰的高度範圍（蛹下緣與裙子上緣之間；上緣不能壓到蛹的下緣）
    "waist_r": (0.125, 0.105), # 腰身半徑（左右 x、前後 z）
    "smooth_iters": 0,         # 整體布料平滑：使用者要求模型改回原本 → 不做
    # 2026-11-19 使用者：下半身與腰身改回原本（不拉直裙子、不露腰）；交界處精細化；裙子上緣左右同高
    "straighten_skirt": False,
    "bare_waist": True,        # 2026-11-19 使用者：要露腰
    "rim_level": 1.09,          # 裙子上緣（臀部骨架外翻邊）一圈對齊的高度（公尺）；None = 不調
    "rim_band": (0.93, 1.13),   # 調平作用的範圍：從 0.93 往上漸強到上緣
    "junction": (1.17, 1.27),   # 蛹（上衣）下緣一圈：整理平滑（露腰後交界就是這圈邊）
    "junction_iters": 30,
    "rim_balance": True,        # （舊）裙子臀部骨架凸出量左右平衡；有 hip_sym 時不用
    # 2026-11-19 第二輪使用者回饋：上半身撫平皺褶、骨架不要太突出；殼內沒穿衣服；交界乾淨；
    # 裙子左右凸出同高、皺褶撫平；白布去污、皮膚與布交界分明、殼內壁不能是膚色
    "hip_sym": (0.84, 1.10),    # 臀部骨架左右鏡像對稱的高度範圍
    "chest_y": (1.22, 1.535),   # 胸口（開口的底）判斷範圍
    "chest_nz": 0.35,           # 法線朝前的門檻
    "chest_depth": 0.045,       # 比殼前緣深多少才算胸口（公尺）
    "chest_zmax": 0.14,         # 胸口最前不超過這個 z（殼外側的前緣不會被誤判成胸口）
    "chest_deep": 0.03,         # 比胸口最深處淺 3 cm 以內的（領子摺邊）也算皮膚
    "hip_front": 40,            # 裙子上緣正前方 ±40°：扭結垂布鼓出的袋狀 → 改用兩側外形內插
    "chest_membrane": 700,      # 胸口內部撫平輪數（邊界固定）
    "smooth_s": 45.0,           # 隱式平滑強度（越大越平）
    "shell_smooth": 0.85,       # 上半身殼的撫平比例
    "skirt_smooth": 0.97,       # 裙子的撫平比例
    "skirt_s": 260.0,           # 裙子用的隱式平滑強度（比上半身強）
    "rim_smooth": 0.75,         # 臀部骨架附近的撫平比例（保留外翻邊的形狀）
    "shelf_depth": 0.035,       # 裙子上緣平台：往下多深的布團要壓平（太深會多出重疊層 → 斑駁）
    "shelf_lip": 0.012,         # 外緣保留的翻邊寬度
    "shelf_rise": 0.008,        # 平台從外緣到腰升高多少
    "back_slope": None,         # （停用）裙子後半圈硬收進腰 → 使用者：要把身體往後移，不是硬接
    "waist_shift": -0.055,      # 2026-11-19 使用者：身體（腰）往後移，腰後側自然對齊臀部上方與蛹下緣後側
    "waist_taper": 0.035,       # 蛹下緣往上這段漸漸收進腰身（取代水平切口）
    "detail_s": 3.0,            # 細紋（絎縫、針織，約 1 cm 以下）的分離平滑強度
    "detail_keep": 0.85,        # 撫平後把細紋加回的比例（保留材質感）
    "chest_hw": ((1.22, 0.06), (1.30, 0.10), (1.37, 0.095), (1.42, 0.065), (1.46, 0.047), (1.50, 0.045), (1.54, 0.04)),
    "neck_hw": 0.045,           # 下巴以下（< 1.51 m）只有這個半寬內是脖子皮膚，兩側是帽兜白色內襯往下延續  # 胸口皮膚的半寬（脖子處窄，兩側是白色內襯）
    "back_sector": (105, 140),  # 後半圈的方位角（度，0 = 正前方）：105° 起漸強、140° 以後完全作用
    "junction_cull": None,  # 停用（原 0.012）：腰與平台已整段重做，刪拉長三角形只會在蛹下緣留下破洞     # 交界範圍內邊長超過這個（公尺）的拉長三角形刪除
    "upper_shift": -0.065,      # 2026-11-19 使用者：整個上半身（腰、殼、頭）往後移，坐在臀部正上方（剛體平移，不變形）
    "skin_hw": ((1.262, 0.0), (1.28, 0.03), (1.31, 0.058), (1.37, 0.062), (1.42, 0.052), (1.46, 0.045), (1.50, 0.043), (1.54, 0.038)),  # 胸口皮膚半寬（兩側的布往內多蓋）
    "piping": 0.005,            # 布與皮膚交界的滾邊寬度
    "tail": (0.24, 0.25, 0.25, 1.7),  # 2026-11-20 作者：魚尾以俐落的斜角張開（起點高度、落地半徑、落地荷葉邊保留比例）
    "neck_smooth": 200,         # 脖子局部撫平輪數（領子的起伏）
    "normal_smooth": 3,         # 法線平滑輪數（柔化銳利摺痕的陰影線）
    "chest_fit": 1.0,           # 胸口平滑曲面擬合的比例
    "cuff": (0.055, 0.15),      # 裙子上緣翻折布的深度：正前方、兩側與後面（公尺）
    "cuff_thick": 0.006,        # 翻折布離裙身的厚度
    "paint_old_waist": False,   # 原始貼圖上的腰不塗膚色（腰面是另外做的）
    "tail_liner": False,        # 魚尾內襯：停用（棕線是貼圖重疊造成的，不是裂縫）
    "hoop": 0.012,              # 裙子上緣：像布套在骨架圈上的圓弧半徑（不是銳利的布邊）
    "top_env": (0.86, 7.0, 4.0),  # 裙子上段包絡面整理：起點高度、方位平滑（格，1 格 2.5°）、高度平滑（格，1 格 5 mm）
    "hem": 0.0022,              # 收邊滾條的半徑（裙子上緣、蛹下緣）
    "skirt_back": (0.97, 0.03),  # 裙子上緣後側收到離腰 3 cm（自此高度起漸收）
    "butt": (0.92, 0.075, 0.028),  # 2026-11-19 使用者：屁股再翹一點（臀峰高度、上下範圍、往後鼓出量）
    "skirt_front": (0.94, 0.075, 0.095),  # 裙子上段前側不再往前鼓：自此高度起前緣上限由 0.075 漸增到上緣 0.095（寬體只往左右）
}
CHEST_V = None   # 每個頂點是否屬於胸口（皮膚）
CHEST_TEX = None  # 對應的貼圖遮罩
YMIN_TEX = None   # 每個貼圖像素所屬三角形的最低頂點高度
FACE_TEX = None   # 臉部保留區（貼圖），法線貼圖只在這裡保留
FACEM_TEX = None    # 臉的保留區（貼圖）
SKIN_RGB = None     # 身體膚色（0~1）
CHEST_EDGE = None   # 胸口皮膚左右邊界：(高度, 左, 右)
CHEST_POLY = None   # 胸口／脖子曲面 z = f(x, y) 的多項式係數
FACE_X0 = 0.0       # 臉中線接縫的 x 位置
COLORED_TEX = None  # 貼圖上有顏色的像素（皮膚、臉）
SKIRT_ENV = None  # 裙子上段的包絡面（smooth_skirt_top 算出，cap_shelf 用它重做外牆）
WAIST_ELL = None  # 腰身橢圓 (cx, cz, rx, rz, 下緣, 上緣)


def _weld(mesh):
    v = mesh.vertices.copy()
    key = np.round(v / 1e-5).astype(np.int64)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    e = inv[mesh.edges_unique]
    return v[uid].copy(), inv, e[e[:, 0] != e[:, 1]]


def _nbr_sum(X, e0, e1):
    """每個頂點所有鄰居的 X 總和（用 bincount，比 np.add.at 快很多）"""
    n = len(X)
    out = np.empty_like(X)
    for c in range(X.shape[1]):
        out[:, c] = np.bincount(e0, X[e1, c], n) + np.bincount(e1, X[e0, c], n)
    return out


def _taubin(P, e, w, iters, lam=0.5, mu=-0.53):
    deg = np.bincount(e.ravel(), minlength=len(P)).astype(np.float64)
    e0, e1 = e[:, 0], e[:, 1]
    W = w[:, None]
    for _ in range(iters):
        for f in (lam, mu):
            if f == 0:
                continue
            P = P + f * W * (_nbr_sum(P, e0, e1) / np.maximum(deg, 1)[:, None] - P)
    return P


def _implicit_smooth(P, e, s, iters=300, tol=1e-7):
    """隱式拉普拉斯平滑：解 (I + s·L) X = P（L = 度數 − 鄰接，共軛梯度法）。
    一次就能抹平好幾公分寬的皺褶與骨架凸起（1 mm 網格用顯式平滑要上千輪）；s 越大越平。"""
    e0, e1 = e[:, 0], e[:, 1]
    deg = np.bincount(e.ravel(), minlength=len(P)).astype(np.float64)[:, None]
    A = lambda X: X + s * (deg * X - _nbr_sum(X, e0, e1))
    X = P.copy()
    R = P - A(X)
    Dg = R.copy()
    rr = (R * R).sum(0)
    r0 = rr.copy()
    for it in range(iters):
        Ad = A(Dg)
        al = rr / np.maximum((Dg * Ad).sum(0), 1e-30)
        X += al * Dg
        R -= al * Ad
        rn = (R * R).sum(0)
        if (rn / r0).max() < tol:
            break
        Dg = R + (rn / rr) * Dg
        rr = rn
    return X


def _vnormals(P, F):
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n = np.zeros_like(P)
    for c in range(3):
        n[:, c] = sum(np.bincount(F[:, k], fn[:, c], len(P)) for k in range(3))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _grow(mask, e, n=1):
    for _ in range(n):
        g = mask.copy()
        g[e[:, 0][mask[e[:, 1]]]] = True
        g[e[:, 1][mask[e[:, 0]]]] = True
        mask = g
    return mask


def _components(mask, e, min_frac=0.15):
    """網格上連在一起的區塊：只留大小 ≥ 最大區塊 min_frac 的（去掉零星小島）"""
    ee = e[mask[e[:, 0]] & mask[e[:, 1]]]
    lab = np.arange(len(mask))
    while True:
        a = np.minimum(lab[ee[:, 0]], lab[ee[:, 1]])
        new = lab.copy()
        np.minimum.at(new, ee[:, 0], a)
        np.minimum.at(new, ee[:, 1], a)
        new = new[new]
        if (new == lab).all():
            break
        lab = new
    vals, cnt = np.unique(lab[mask], return_counts=True)
    big = vals[cnt >= cnt.max() * min_frac]
    return mask & np.isin(lab, big)


def level_rim(P, cfg):
    """裙子上緣（臀部骨架外翻邊）左右高度不一：量出每個方位角的上緣高度，
    把整圈對齊到 rim_level（低的往上拉、高的往下壓），從 rim_band 下緣往上漸強；上緣以上（蛹）不動。"""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    lo_, hi_ = cfg["rim_band"]
    band = (y > 0.95) & (y < 1.12)
    cx, cz = np.median(x[band]), np.median(z[band])
    r = np.hypot(x - cx, z - cz)
    th = np.arctan2(x - cx, z - cz)
    nb = 48
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    top = np.full(nb, np.nan)
    for i in range(nb):
        s = band & (th >= edges[i]) & (th < edges[i + 1])
        if s.sum() < 30:
            continue
        rmax = np.percentile(r[s], 99)
        lip = s & (r > rmax - 0.015)
        top[i] = np.percentile(y[lip], 95)
    # 正前方（±35°）被交界的扭結擋住，量到的是扭結不是裙子上緣 → 用兩側推算
    cen0 = (edges[:-1] + edges[1:]) / 2
    top[np.abs(cen0) < np.radians(35)] = np.nan
    ok = ~np.isnan(top)
    idx = np.arange(nb)
    top = np.interp(idx, idx[ok], top[ok], period=nb)
    # 方位角方向平滑（環狀）
    k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2); k /= k.sum()
    top = np.convolve(np.r_[top[-4:], top, top[:4]], k, "valid")
    cen = (edges[:-1] + edges[1:]) / 2
    h = np.interp(th, cen, top, period=2 * np.pi)
    delta = cfg["rim_level"] - h
    # 作用權重：0.93 往上漸強到上緣；上緣上方 2 cm 內漸弱為 0（蛹不動）
    w = np.clip((y - lo_) / np.maximum(h - lo_, 1e-3), 0, 1)
    w = np.where(y > h, np.clip(1 - (y - h) / 0.02, 0, 1), w)
    w = w * (r > 0.08)  # 身體中心（腰內側）不動
    P = P.copy()
    P[:, 1] = y + delta * w
    print(f"rim level: 上緣高度 {top.min():.3f}～{top.max():.3f} → 全部 {cfg['rim_level']:.3f}")
    return P


SHIRT_V = None   # 每個頂點是否屬於殼內的襯衫（shirt_vertices 算）
SHIRT_TEX = None  # 對應的貼圖遮罩（texel_flag 算）


def shirt_vertices(mesh):
    """殼內的襯衫：胸口開口一帶、原始貼圖顏色偏暖（米色）的頂點，再沿網格往外擴兩圈。"""
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    Ht, Wt = tex.shape[:2]
    uv = mesh.visual.uv
    px = np.clip(((uv[:, 0] % 1) * (Wt - 1)).astype(int), 0, Wt - 1)
    py = np.clip(((1 - uv[:, 1] % 1) * (Ht - 1)).astype(int), 0, Ht - 1)
    h, s, v = hsv(tex[py, px][:, None, :])
    h, s, v = h[:, 0], s[:, 0], v[:, 0]
    P = mesh.vertices
    region = (np.abs(P[:, 0]) < 0.15) & (P[:, 1] > 1.24) & (P[:, 1] < 1.47) & (P[:, 2] > -0.06)  # 只限胸口開口（腰部由露腰處理）
    flag = region & (h > 8) & (h < 55) & (s > 0.08) & (v > 0.2)
    # 位置相同的重複頂點（貼圖接縫）一起算；沿邊往外擴兩圈
    key = np.round(P / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    f = np.zeros(inv.max() + 1, bool)
    f[inv[flag]] = True
    e = inv[mesh.edges_unique]
    for _ in range(2):
        g = f.copy()
        g[e[:, 0][f[e[:, 1]]]] = True
        g[e[:, 1][f[e[:, 0]]]] = True
        f = g & np.isin(np.arange(len(f)), inv[region])
    out = f[inv]
    print("shirt vertices", int(out.sum()))
    return out


def texel_ymin(mesh, size):
    """每個貼圖像素所屬三角形的最低頂點高度（腰的膚色只給整個三角形都在腰上的）"""
    from PIL import ImageDraw
    img = Image.new("F", (size, size), float("nan"))
    d = ImageDraw.Draw(img)
    uv = mesh.visual.uv
    tri = uv[mesh.faces]
    xs = (tri[:, :, 0] % 1) * (size - 1)
    ys = (1 - (tri[:, :, 1] % 1)) * (size - 1)
    ymin = mesh.vertices[mesh.faces][:, :, 1].min(1)
    for i in np.where((ymin > 1.0) & (ymin < 1.25))[0]:
        d.polygon([(xs[i, 0], ys[i, 0]), (xs[i, 1], ys[i, 1]), (xs[i, 2], ys[i, 2])], fill=float(ymin[i]))
    return np.asarray(img)


def texel_flag(mesh, size, vflag):
    """每個貼圖像素是否屬於旗標頂點組成的三角形（三個頂點中至少兩個）"""
    from PIL import ImageDraw
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    uv = mesh.visual.uv
    tri = uv[mesh.faces]
    xs = (tri[:, :, 0] % 1) * (size - 1)
    ys = (1 - (tri[:, :, 1] % 1)) * (size - 1)
    fsel = np.where(vflag[mesh.faces].sum(1) >= 2)[0]
    for i in fsel:
        d.polygon([(xs[i, 0], ys[i, 0]), (xs[i, 1], ys[i, 1]), (xs[i, 2], ys[i, 2])], fill=255, outline=255)
    return np.asarray(img).astype(np.float32) / 255


def balance_rim(P, cfg):
    """裙子臀部骨架凸出量左右平衡：每個方位角的外緣半徑，改成和左右對稱位置的平均（鏡像平均）"""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    lo_, hi_ = cfg["rim_band"]
    band = (y > 0.95) & (y < 1.12)
    cx, cz = np.median(x[band]), np.median(z[band])
    r = np.hypot(x - cx, z - cz)
    th = np.arctan2(x - cx, z - cz)  # 0 = 正前方；左右對稱 = th ↔ -th
    nb = 48
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    cen = (edges[:-1] + edges[1:]) / 2
    rr = np.full(nb, np.nan)
    for i in range(nb):
        s = band & (th >= edges[i]) & (th < edges[i + 1])
        if s.sum() >= 30:
            rr[i] = np.percentile(r[s], 97)
    ok = ~np.isnan(rr)
    idx = np.arange(nb)
    rr = np.interp(idx, idx[ok], rr[ok], period=nb)
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2); k /= k.sum()
    rr = np.convolve(np.r_[rr[-3:], rr, rr[:3]], k, "valid")
    mirror = np.interp(-cen, cen, rr, period=2 * np.pi)
    target = (rr + mirror) / 2
    fac_b = target / np.maximum(rr, 1e-3)
    fac = np.interp(th, cen, fac_b, period=2 * np.pi)
    w = np.clip((y - lo_) / 0.06, 0, 1) * np.clip((hi_ + 0.02 - y) / 0.04, 0, 1) * (r > 0.08)
    P = P.copy()
    f = 1 + (fac - 1) * w
    P[:, 0] = cx + (x - cx) * f
    P[:, 2] = cz + (z - cz) * f
    print(f"rim balance: 左右凸出差 最大 {np.abs(rr - mirror).max():.3f} m → 0")
    return P


def chest_floor(P, F, e, cfg):
    """殼前方 V 形開口的「底」＝她的胸口（身體）；開口兩側的牆＝殼的內裡。
    以幾何判斷：法線朝前、且比殼的前緣深 4.5 cm 以上的頂點 → 胸口。牆一律不算（不會被染成膚色）。"""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    e0, e1 = e[:, 0], e[:, 1]
    deg = np.bincount(e.ravel(), minlength=len(P)).astype(np.float64)[:, None]
    n = _vnormals(P, F)
    for _ in range(8):  # 摺痕上的法線很亂 → 先和鄰居平均
        n = n + _nbr_sum(n, e0, e1) / np.maximum(deg, 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    ya, yb = cfg["chest_y"]
    region = (np.abs(x) < 0.14) & (y > ya) & (y < yb) & (z > -0.05)
    ys = np.arange(ya - 0.01, yb + 0.02, 0.01)
    zf = []
    for yy in ys:
        s_ = (np.abs(y - yy) < 0.006) & (np.abs(x) < 0.2)
        zf.append(np.percentile(z[s_], 97) if s_.sum() > 20 else np.nan)
    zf = np.array(zf)
    zf = np.interp(ys, ys[~np.isnan(zf)], zf[~np.isnan(zf)])
    zfv = np.interp(y, ys, zf)
    deep_ok = region & (z < zfv - cfg["chest_depth"]) & (z < cfg["chest_zmax"])
    if cfg.get("chest_hw"):
        hy, hw = zip(*cfg["chest_hw"])
        deep_ok &= np.abs(x) < np.interp(y, hy, hw)  # 脖子附近窄：兩側殼內壁（帽兜白色內襯往下延續）不算皮膚
    cand = deep_ok & (n[:, 2] > cfg["chest_nz"])
    cand = _components(cand, e)
    # 領子的摺邊（法線朝側面）也在胸口深處：每個高度比胸口最深處淺 chest_deep 以內的都算皮膚（不然會留白色碎片）
    zmin = []
    for yy in ys:
        s_ = cand & (np.abs(y - yy) < 0.006)
        zmin.append(np.percentile(z[s_], 10) if s_.sum() > 20 else np.nan)
    zmin = np.array(zmin)
    if (~np.isnan(zmin)).sum() > 2:
        zmin = np.interp(ys, ys[~np.isnan(zmin)], zmin[~np.isnan(zmin)])
        cand = _components(cand | (deep_ok & (z < np.interp(y, ys, zmin) + cfg["chest_deep"])), e)
    cl = _grow(cand, e, 3) & region  # 填小洞：擴 3 圈再縮 3 圈
    cl = ~_grow(~cl, e, 3)
    cl = _components(cl, e)
    print("chest floor vertices", int(cl.sum()), "(candidates", int(cand.sum()), ")")
    return cl


def symmetrize_hips(P, cfg):
    """裙子臀部骨架（左右凸出）左右對稱：每個高度、每個方位角的外緣半徑，改成和鏡像方位的平均。
    凸出的高度、寬度、往下垂的形狀左右一致。"""
    ya, yb = cfg["hip_sym"]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ref = (y > 0.95) & (y < 1.08)
    cx, cz = np.median(x[ref]), np.median(z[ref])
    sub = np.where((y > ya - 0.04) & (y < yb + 0.02))[0]
    r = np.hypot(x[sub] - cx, z[sub] - cz)
    th = np.arctan2(x[sub] - cx, z[sub] - cz)
    nb = 72
    cen = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
    yg = np.arange(ya - 0.04, yb + 0.025, 0.01)
    bi = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    R = np.full((len(yg), nb), np.nan)
    for i, yy in enumerate(yg):
        s_ = np.abs(y[sub] - yy) < 0.007
        for j in range(nb):
            t_ = s_ & (bi == j)
            if t_.sum() >= 8:
                R[i, j] = np.percentile(r[t_], 97)
    for i in range(len(yg)):  # 缺的方位用左右插值補；沿方位角平滑
        ok = ~np.isnan(R[i])
        if ok.sum() < 4:
            R[i] = np.nanmedian(R) if np.isnan(R[i]).all() else np.nanmean(R[i])
            continue
        R[i] = np.interp(np.arange(nb), np.arange(nb)[ok], R[i][ok], period=nb)
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2); k /= k.sum()
    k2 = np.array([0.25, 0.5, 0.25])
    def _sm2(M):
        M = np.stack([np.convolve(np.r_[row[-3:], row, row[:3]], k, "valid") for row in M])
        return np.vstack([M[:1], np.stack([k2 @ M[i - 1:i + 2] for i in range(1, len(M) - 1)]), M[-1:]])
    R = _sm2(R)  # 實際量到的外緣半徑（平滑）
    Rt = R.copy()  # 目標
    if cfg.get("hip_front"):
        # 正前方：扭結的垂布在裙子上緣鼓成袋狀 → 用 ±hip_front 外側的半徑內插（上緣附近完全取代，往下漸弱；只縮不放）
        fs_ = np.abs(np.degrees(cen)) < cfg["hip_front"]
        idx = np.arange(nb)
        wrow = np.clip((yg - (ya + 0.02)) / 0.08, 0, 1)
        for i in range(len(yg)):
            interp = np.interp(idx, idx[~fs_], R[i][~fs_], period=nb)
            Rt[i] = np.where(fs_, np.minimum(R[i], R[i] * (1 - wrow[i]) + interp * wrow[i]), R[i])
    target = (Rt + Rt[:, ::-1]) / 2  # 方位角 θ 與 −θ 的鏡像
    fac = target / np.maximum(R, 1e-3)
    print(f"hip symmetry: 左右凸出差 最大 {np.abs(R - R[:, ::-1]).max():.3f} m → 0")
    # 每個頂點：沿高度與方位角內插
    yi = np.clip((y[sub] - yg[0]) / 0.01, 0, len(yg) - 1.001)
    i0 = yi.astype(int); fy = yi - i0
    tj = (th + np.pi) / (2 * np.pi) * nb - 0.5
    j0 = np.floor(tj).astype(int); fj = tj - j0
    j0m, j1m = j0 % nb, (j0 + 1) % nb
    f = (fac[i0, j0m] * (1 - fj) + fac[i0, j1m] * fj) * (1 - fy) + (fac[i0 + 1, j0m] * (1 - fj) + fac[i0 + 1, j1m] * fj) * fy
    ys_ = y[sub]
    w = np.clip((ys_ - (ya - 0.04)) / 0.04, 0, 1) * np.clip((yb + 0.02 - ys_) / 0.02, 0, 1) * (r > 0.08)
    f = 1 + (f - 1) * w
    P = P.copy()
    P[sub, 0] = cx + (x[sub] - cx) * f
    P[sub, 2] = cz + (z[sub] - cz) * f
    return P


def flatten_shelf(P, cfg, ell):
    """裙子上緣往內到腰之間的「平台」：上面卡著的布團（扭結殘留）壓平到一個乾淨的斜面（外緣 → 腰微微升高）"""
    lvl = cfg["rim_level"]
    wcx, wcz, rx, rz = ell
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    sub = np.where((y > lvl - cfg["shelf_depth"]) & (y < cfg["waist"][0] + 0.055))[0]
    xs, ys_, zs = x[sub], y[sub], z[sub]
    r = np.hypot(xs - wcx, zs - wcz)
    th = np.arctan2(xs - wcx, zs - wcz)
    nb = 72
    bi = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    rim = (ys_ > lvl - 0.03) & (ys_ < lvl + 0.003)
    Rr = np.full(nb, np.nan)
    for j in range(nb):
        t_ = rim & (bi == j)
        if t_.sum() >= 8:
            Rr[j] = np.percentile(r[t_], 98)
    ok = ~np.isnan(Rr)
    Rr = np.interp(np.arange(nb), np.arange(nb)[ok], Rr[ok], period=nb)
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2); k /= k.sum()
    Rr = np.convolve(np.r_[Rr[-3:], Rr, Rr[:3]], k, "valid")
    tj = (th + np.pi) / (2 * np.pi) * nb - 0.5
    rv = np.interp(tj, np.arange(nb), Rr, period=nb)
    # 腰身橢圓在這個方位的半徑
    re = 1 / np.sqrt((np.sin(th) / rx) ** 2 + (np.cos(th) / rz) ** 2)
    # 平台下方的布團也壓上來（平台有小縫時不會從縫裡看到）：外牆半徑逐高度量，只壓外牆以內的
    yb_ = np.arange(lvl - cfg["shelf_depth"], lvl + 0.006, 0.01)
    Rw = np.full((len(yb_), nb), np.nan)
    for i, yy in enumerate(yb_):
        s_ = np.abs(ys_ - yy) < 0.006
        for j in range(nb):
            t_ = s_ & (bi == j)
            if t_.sum() >= 8:
                Rw[i, j] = np.percentile(r[t_], 98)
        ok_ = ~np.isnan(Rw[i])
        Rw[i] = np.interp(np.arange(nb), np.arange(nb)[ok_], Rw[i][ok_], period=nb) if ok_.sum() > 3 else Rr
    yi = np.clip((ys_ - yb_[0]) / 0.01, 0, len(yb_) - 1).round().astype(int)
    rw = np.empty(len(sub))
    for i in range(len(yb_)):
        m_ = yi == i
        rw[m_] = np.interp(tj[m_], np.arange(nb), Rw[i], period=nb)
    rw = np.minimum(rw, rv)
    # 平台高度以上只壓離腰身明顯較遠的布團（腰身本身的下緣不能被壓下去，否則腰後面會破洞）
    inner = (r < np.where(ys_ < lvl, rw, rv) - cfg["shelf_lip"]) & (r > re * 0.98) & (ys_ < cfg["waist"][0] + 0.03)         & ((ys_ < cfg["waist"][0]) | (r > re * 1.15))
    t = np.clip((rv - cfg["shelf_lip"] - r) / np.maximum(rv - cfg["shelf_lip"] - re, 1e-3), 0, 1)
    ynew = lvl - 0.002 + cfg["shelf_rise"] * t
    # 外緣翻邊上方（正前方扭結殘留高到 1.14）：高出上緣的一律壓到上緣高度
    # 只壓上緣到腰下緣之間；更高的交給腰身投影收進腰裡（整段壓下會把三角形拉成直條紋）
    lip = (r >= rv - cfg["shelf_lip"]) & (r > re * 1.05) & (ys_ > lvl) & (ys_ < cfg["waist"][0])
    P = P.copy()
    idx = sub[inner]
    P[idx, 1] = ynew[inner]
    P[sub[lip], 1] = lvl
    # 腰下緣下方、腰身橢圓內側垂出來的布（舌狀布團）：往內收到 0.8 倍腰身半徑，藏進裙子裡
    tongue = (r <= re * 0.98) & (r > re * 0.5) & (ys_ < cfg["waist"][0] - 0.003)
    kk = np.where(tongue, re * 0.8 / np.maximum(r, 1e-6), 1.0)
    kk = np.minimum(kk, 1.0)
    P[sub, 0] = wcx + (P[sub, 0] - wcx) * kk
    P[sub, 2] = wcz + (P[sub, 2] - wcz) * kk
    print("shelf tongue tucked vertices", int((tongue & (kk < 1)).sum()))
    print("shelf lip lowered vertices", int(lip.sum()))
    print("shelf flattened vertices", len(idx))
    return P


def back_slope(P, cfg, ell):
    """後腰接臀部：裙子後半圈從 back_slope[0] 往上，外緣半徑以 S 形曲線收進腰身橢圓（到 back_slope[1] 剛好接上）。
    後面不再有平台，從側面看腰的後面順順地接到屁股上方。側面的臀部骨架凸出與前面不動。"""
    cx, cz, rx, rz, a, b2 = ell
    y0, y1 = cfg["back_slope"]
    s0, s1 = np.radians(cfg["back_sector"][0]), np.radians(cfg["back_sector"][1])
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    sub = np.where((y > y0 - 0.01) & (y < y1 + 0.01))[0]
    dx, dz = x[sub] - cx, z[sub] - cz
    r = np.hypot(dx, dz)
    th = np.arctan2(dx, dz)
    tb = np.clip((np.abs(th) - s0) / (s1 - s0), 0, 1)
    wb = tb * tb * (3 - 2 * tb)
    nb = 72
    bi = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    base = np.abs(y[sub] - y0) < 0.006
    R0 = np.full(nb, np.nan)
    for j in range(nb):
        t_ = base & (bi == j)
        if t_.sum() >= 8:
            R0[j] = np.percentile(r[t_], 97)
    ok = ~np.isnan(R0)
    R0 = np.interp(np.arange(nb), np.arange(nb)[ok], R0[ok], period=nb)
    tj = (th + np.pi) / (2 * np.pi) * nb - 0.5
    r0 = np.interp(tj, np.arange(nb), R0, period=nb)
    re = 1 / np.sqrt((np.sin(th) / rx) ** 2 + (np.cos(th) / rz) ** 2)
    re1 = re * (1 - 0.07 * np.sin(np.pi * np.clip((y1 - a) / (b2 - a), 0, 1)))
    t = np.clip((y[sub] - y0) / (y1 - y0), 0, 1)
    Rt = r0 + (re1 - r0) * (t * t * (3 - 2 * t))
    ys_ = y[sub]
    # 腰身範圍內（已貼在橢圓上）全部推到斜面；平台高度以下只收外面的（裡面藏著的布不動）
    move = (wb > 0) & (((ys_ >= a) & (r > re * 0.85)) | ((ys_ < a) & (r > Rt - 0.01)))
    k = np.where(move, 1 + (Rt / np.maximum(r, 1e-6) - 1) * wb, 1.0)
    P = P.copy()
    P[sub, 0] = cx + dx * k
    P[sub, 2] = cz + dz * k
    print("back slope moved vertices", int(move.sum()))
    return P


def shift_body(P, cfg, ell):
    """身體（腰）往後移：用平滑的位移場，腰身內全部移動；裙子平台由腰往外緣漸弱（外牆不動）、
    蛹下緣由腰往外漸弱、往上漸弱 → 網格不被拉長（不會破洞、不會出現碎片）"""
    cx, cz, rx, rz, a, b2 = ell
    sh = cfg["waist_shift"]
    lvl = cfg["rim_level"]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    sub = np.where((y > lvl - 0.12) & (y < b2 + 0.08))[0]
    xs, ys_, zs = x[sub], y[sub], z[sub]
    r = np.hypot(xs - cx, zs - cz)
    th = np.arctan2(xs - cx, zs - cz)
    re = 1 / np.sqrt((np.sin(th) / rx) ** 2 + (np.cos(th) / rz) ** 2)
    nb = 72
    bi = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    rim = (ys_ > lvl - 0.03) & (ys_ < lvl + 0.003)
    Rr = np.full(nb, np.nan)
    for j in range(nb):
        t_ = rim & (bi == j)
        if t_.sum() >= 8:
            Rr[j] = np.percentile(r[t_], 98)
    ok = ~np.isnan(Rr)
    Rr = np.interp(np.arange(nb), np.arange(nb)[ok], Rr[ok], period=nb)
    rv = np.interp((th + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), Rr, period=nb)
    ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    # 裙子部分（腰下緣以下）：腰身內 1 → 外緣翻邊 0；更下面（藏在裙子裡的身體）往下漸弱
    w_low = ss((rv - cfg["shelf_lip"] - r) / np.maximum(rv - cfg["shelf_lip"] - re, 1e-3)) * ss((ys_ - (lvl - 0.12)) / 0.09)
    # 蛹下緣（腰上緣以上）：腰身附近 1 → 半徑 1.8 倍處 0；往上 8 cm 漸弱
    w_up = ss((re * 1.8 - r) / (re * 0.6)) * (1 - ss((ys_ - b2) / 0.08))
    w = np.where(ys_ < a, w_low, np.where(ys_ > b2, w_up, 1.0))
    P = P.copy()
    P[sub, 2] += sh * w
    print(f"body shift: {sh * 100:.1f} cm, moved vertices {int((w > 0.01).sum())}")
    return P


def move_upper(mesh, cfg=RESHAPE):
    """最後一步（貼圖算完後）：整個上半身剛體往後移，裙子上段前側往後收。
    平台（腰與裙子外緣之間）用線性權重過渡 → 位移單調、不會摺疊穿模。"""
    if WAIST_ELL is None or not cfg.get("upper_shift"):
        return
    cx, cz, rx, rz, a, b2 = WAIST_ELL
    D = cfg["upper_shift"]
    lvl = cfg["rim_level"]
    V = mesh.vertices.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    r = np.hypot(x - cx, z - cz)
    th = np.arctan2(x - cx, z - cz)
    re = 1 / np.sqrt((np.sin(th) / rx) ** 2 + (np.cos(th) / rz) ** 2)
    nb = 72
    bi = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    rim = (y > lvl - 0.03) & (y < lvl + 0.003)
    Rr = np.full(nb, np.nan)
    for j in range(nb):
        t_ = rim & (bi == j)
        if t_.sum() >= 8:
            Rr[j] = np.percentile(r[t_], 98)
    ok = ~np.isnan(Rr)
    Rr = np.interp(np.arange(nb), np.arange(nb)[ok], Rr[ok], period=nb)
    rv = np.interp((th + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), Rr, period=nb) - cfg["shelf_lip"]
    # 位移權重：腰身內 1 → 裙子外緣 0（線性）；往下 9 cm 漸弱；腰下緣往上 3 cm 全部變 1（殼、頭整個移）
    w_rad = np.clip((rv - r) / np.maximum(rv - re, 0.01), 0, 1)
    w_rad = np.where(r <= re, 1.0, w_rad)
    w = w_rad * ss((y - (lvl - 0.12)) / 0.09)
    w = w + (1 - w) * ss((y - a) / 0.03)
    # 裙子前側收回：z0 為支點，前半往後等比壓縮到 T(y)
    y0, f0, f1 = cfg["skirt_front"]
    sk = (w < 0.999) & (y > y0 - 0.02) & (y < a + 0.03)
    yb = np.arange(y0 - 0.02, a + 0.03, 0.005)
    zmo = np.array([np.percentile(z[sk & (np.abs(y - t) < 0.004)], 99.5) if (sk & (np.abs(y - t) < 0.004)).sum() > 20 else np.nan for t in yb])
    okb = ~np.isnan(zmo)
    zmo = np.interp(yb, yb[okb], zmo[okb])
    zmo = np.convolve(np.pad(zmo, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    T = f0 + (f1 - f0) * ss((yb - y0) / (lvl - y0))
    z0 = -0.02
    k = np.minimum(1.0, (T - z0) / np.maximum(zmo - z0, 1e-3))
    kv = np.interp(y, yb, k)
    kv = 1 - (1 - kv) * ss((y - (y0 - 0.02)) / 0.02)  # 下端平順接回原形
    zf = np.where(z > z0, z0 + (z - z0) * kv, z)
    # 裙子上段後側往腰收：臀峰往上到裙子上緣漸漸收進來（腰→臀是一條順的弧線，上緣不再和屁股一樣凸）
    if cfg.get("skirt_back"):
        yb0, lipb = cfg["skirt_back"]
        zmb = np.array([np.percentile(z[sk & (np.abs(y - t) < 0.004)], 0.5) if (sk & (np.abs(y - t) < 0.004)).sum() > 20 else np.nan for t in yb])
        okb = ~np.isnan(zmb)
        zmb = np.interp(yb, yb[okb], zmb[okb])
        zmb = np.convolve(np.pad(zmb, 3, mode="edge"), np.ones(7) / 7, mode="valid")
        b0 = float(np.interp(yb0, yb, zmb))
        b1 = max(b0, (cz - rz) + D - lipb)
        TB = b0 + (b1 - b0) * ss((yb - yb0) / (lvl - yb0))
        kb = np.where(yb > yb0, np.minimum(1.0, (z0 - TB) / np.maximum(z0 - zmb, 1e-3)), 1.0)
        kbv = np.interp(y, yb, kb)
        zf = np.where(z < z0, z0 + (zf - z0) * kbv, zf)
        print(f"skirt back tuck: rim back {zmb[-8]:+.3f} -> {b1:+.3f}, k min {kb.min():.2f}")
    # 臀部翹度：後側在臀峰高度往後鼓出（高斯分佈，左右往側面漸弱），上下自然收回
    by, bs, ba = cfg.get("butt", (0.91, 0.075, 0.0))
    if ba:
        tb = np.arctan2(x, z + 0.045)
        zf = zf - ba * np.exp(-((y - by) / bs) ** 2) * np.clip(-np.cos(tb), 0, 1) ** 1.5
    V[:, 2] = w * (z + D) + (1 - w) * zf
    V[:, 2] -= D  # 整體再往前平移回來：上半身留在原位（臉、燈光、發光中心的設定都不用動），等於下半身往前移
    print(f"upper shift {D * 100:.1f} cm, skirt front k min {k.min():.2f}, moved {int((w > 0.01).sum())}")
    mesh.vertices = V


def shape_tail(mesh, cfg=RESHAPE):
    """魚尾：原本是直筒到地面才攤平成一圈 → 改成從 y1 起以固定斜角一路張開到地面（直線輪廓的錐形），
    地上攤開的那圈縮成錐底外的一小圈荷葉邊。"""
    if not cfg.get("tail"):
        return
    y1, R0, brim, pw_ = (tuple(cfg["tail"]) + (1.0,))[:4]
    V = mesh.vertices.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ys = np.arange(0.05, y1 + 0.03, 0.01)
    cxs, czs, rs = [], [], []
    for yy in ys:
        s_ = np.abs(y - yy) < 0.008
        cx_, cz_ = np.median(x[s_]), np.median(z[s_])
        cxs.append(cx_); czs.append(cz_)
        rs.append(np.percentile(np.hypot(x[s_] - cx_, z[s_] - cz_), 95))
    k5 = np.ones(5) / 5
    smo = lambda A: np.convolve(np.pad(np.array(A), 2, mode="edge"), k5, mode="valid")
    cxs, czs, rs = smo(cxs), smo(czs), smo(rs)
    sel = y < y1
    ysel = y[sel]
    cx = np.interp(ysel, ys, cxs); cz = np.interp(ysel, ys, czs); rt = np.interp(ysel, ys, rs)
    R1 = float(np.interp(y1, ys, rs))
    t = np.clip(1 - ysel / y1, 0, 1)
    target = R1 + (R0 - R1) * t ** pw_   # 指數 > 1：從裙身平順地彎出去，越往下越斜（不是突然折一個角）
    g = target / rt
    g = 1 + (g - 1) * np.clip((y1 - ysel) / 0.08, 0, 1) ** 2 * (3 - 2 * np.clip((y1 - ysel) / 0.08, 0, 1))  # 起點平順接上
    dx, dz = x[sel] - cx, z[sel] - cz
    r = np.maximum(np.hypot(dx, dz), 1e-6)
    r2 = np.where(r <= rt, r * g, rt * g + (r - rt) * brim)
    V[sel, 0] = cx + dx / r * r2
    V[sel, 2] = cz + dz / r * r2
    mesh.vertices = V
    print(f"tail cone: from y {y1} r {R1:.3f} to floor r {R0}, slope {np.degrees(np.arctan2(R0 - R1, y1)):.0f} deg from vertical")


def protect_white_uv(mesh, colored, used, radius=None):
    """白布的頂點如果 UV 落在膚色／臉色像素旁邊（貼圖區塊邊緣），縮小取樣時會混到膚色 → 白布上出現棕線。
    把這些頂點的 UV 移到最近的「安全白布像素」（離有色像素 radius 以上）。只動不屬於皮膚三角形的頂點。"""
    import cv2
    H_, W_ = colored.shape
    radius = radius or int(18 * W_ / 1024)
    uv = np.array(mesh.visual.uv, dtype=np.float64)
    px = np.clip(np.round((uv[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
    py = np.clip(np.round((1 - uv[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
    danger = cv2.dilate(colored.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    Fc = mesh.faces
    cu = uv[Fc].mean(1)
    cpx = np.clip(np.round((cu[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
    cpy = np.clip(np.round((1 - cu[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
    tcol = colored[cpy, cpx] | colored[py[Fc], px[Fc]].any(axis=1)
    touches = np.zeros(len(uv), bool)
    touches[Fc[tcol].ravel()] = True
    move = danger[py, px] & ~colored[py, px] & ~touches
    if not move.any():
        return
    unsafe = (danger | ~used).astype(np.uint8)          # 0 = 安全的白布像素
    _, labels = cv2.distanceTransformWithLabels(unsafe, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(unsafe == 0)
    lut = np.zeros((labels.max() + 1, 2), np.int64)
    lut[labels[zy, zx]] = np.c_[zy, zx]
    tgt = lut[labels[py[move], px[move]]]
    uv[move, 0] = tgt[:, 1] / (W_ - 1)
    uv[move, 1] = 1 - tgt[:, 0] / (H_ - 1)
    mesh.visual.uv = uv
    print(f"white uv protected: {int(move.sum())} vertices moved away from skin texels")


def front_face_skin(mesh, colored):
    """胸口、脖子、下巴的皮膚三角形如果朝後（繞序相反），從前面看到的是它的背面；
    檢視程式把背面畫成布的內側（白色）→ 下巴下方出現一圈白。把這些面翻成朝前。"""
    H_, W_ = colored.shape
    uv = np.asarray(mesh.visual.uv)
    Fc = np.array(mesh.faces)
    V = mesh.vertices
    cu = uv[Fc].mean(1)
    cpx = np.clip(np.round((cu[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
    cpy = np.clip(np.round((1 - cu[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
    T = V[Fc]
    nz = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])[:, 2]
    cen = T.mean(1)
    flip = colored[cpy, cpx] & (nz < 0) & (cen[:, 1] > 1.2)
    Fc[flip] = Fc[flip][:, ::-1]
    mesh.faces = Fc
    print(f"skin faces turned to front: {int(flip.sum())}")


def remove_chin_collar(mesh, facem, cfg=RESHAPE):
    """下巴下方有一條原本襯衫領子的布帶，立在脖子表面前方數公分（不是臉、也不是脖子）。
    刪掉這些三角形：下巴以下直接看到後面內收的脖子。臉（貼圖在臉保留區內）的三角形不動。"""
    if CHEST_POLY is None:
        return
    H_, W_ = facem.shape
    uv = np.asarray(mesh.visual.uv)
    Fc = np.array(mesh.faces)
    V = mesh.vertices
    cen = V[Fc].mean(1)
    box = (np.abs(cen[:, 0]) < 0.033) & (cen[:, 1] > 1.452) & (cen[:, 1] < 1.526)
    idx = np.where(box)[0]
    xs_, ys_ = cen[idx, 0] / 0.1, (cen[idx, 1] - 1.36) / 0.1
    zfit = np.zeros(len(idx))
    k_ = 0
    for i in range(5):
        for j in range(5 - i):
            zfit += CHEST_POLY[k_] * xs_ ** i * ys_ ** j
            k_ += 1
    front = cen[idx, 2] > zfit + 0.022
    # 三個頂點有任何一個落在臉的保留區 → 是臉，不刪
    px = np.clip(np.round((uv[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
    py = np.clip(np.round((1 - uv[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
    vface = facem[py, px]
    cu_ = uv[Fc[idx]].mean(1)
    cface = facem[np.clip(np.round((1 - cu_[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1), np.clip(np.round((cu_[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)]
    isface = cface & (vface[Fc[idx]].sum(axis=1) >= 2)
    kill = np.zeros(len(Fc), bool)
    kill[idx[front & ~isface]] = True
    mesh.update_faces(~kill)
    print(f"chin collar faces removed: {int(kill.sum())}")


def bridge_jaw(mesh, colored):
    """脖子頂端（y≈1.488）和下顎底面（y≈1.496）之間沒有面：從前面會看穿到帽兜後壁（一道白月牙）。
    每 2 mm 一欄，找脖子最上緣與下顎最後緣，補一片皮膚接起來。"""
    if CHEST_POLY is None:
        return mesh
    H_, W_ = colored.shape
    V = np.asarray(mesh.vertices); UV = np.asarray(mesh.visual.uv); Fc = np.asarray(mesh.faces)
    px = np.clip(np.round((UV[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
    py = np.clip(np.round((1 - UV[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
    used_v = np.zeros(len(V), bool); used_v[Fc.ravel()] = True
    sk = colored[py, px] & used_v & (np.abs(V[:, 0]) < 0.05) & (V[:, 1] > 1.40) & (V[:, 1] < 1.54)
    ii = np.where(sk)[0]
    x, y, z = V[ii, 0], V[ii, 1], V[ii, 2]
    xs_, ys_ = x / 0.1, (y - 1.36) / 0.1
    zf = np.zeros(len(ii)); k_ = 0
    for i in range(5):
        for j in range(5 - i):
            zf += CHEST_POLY[k_] * xs_ ** i * ys_ ** j
            k_ += 1
    cols = np.arange(-0.030, 0.0301, 0.002)
    N, J = np.full((len(cols), 2), np.nan), np.full((len(cols), 2), np.nan)
    for c, xc in enumerate(cols):
        m = np.abs(x - xc) < 0.0025
        nk = m & (z < zf + 0.010) & (z > zf - 0.02) & (y > 1.46) & (y < 1.50)
        if nk.sum() < 4:
            continue
        yn = np.percentile(y[nk], 96)
        zn = np.median(z[nk & (y > yn - 0.004)])
        jw = m & (z > zn + 0.012) & (y > yn - 0.004) & (y < yn + 0.03)
        if jw.sum() < 4:
            continue
        ylo = np.percentile(y[jw], 4)
        low = jw & (y < ylo + 0.005)
        N[c] = yn, zn
        J[c] = np.median(y[low]), np.percentile(z[low], 10)
    ok = ~np.isnan(N[:, 0])
    if ok.sum() < 6:
        print("jaw bridge: not enough columns", int(ok.sum()))
        return mesh
    for A in (N, J):                                     # 沿左右方向平滑
        for d in range(2):
            v = np.interp(np.arange(len(cols)), np.where(ok)[0], A[ok, d])
            A[:, d] = np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    lo_, hi_ = np.where(ok)[0][[0, -1]]
    cs = np.arange(lo_, hi_ + 1)
    # 膚色：胸口正中、遠離邊界的一個頂點的貼圖位置
    mid = (np.abs(x) < 0.012) & (np.abs(y - 1.445) < 0.012) & (np.abs(z - zf) < 0.006)
    if not mid.any():
        return mesh
    uv0 = UV[ii[np.where(mid)[0][np.argmin(np.abs(x[mid]) + np.abs(y[mid] - 1.445))]]]
    rows = 6
    newV = []
    for r_ in range(rows):
        f = r_ / (rows - 1)
        yy = (N[cs, 0] - 0.005) + (J[cs, 0] + 0.004 - (N[cs, 0] - 0.005)) * f
        zz = (N[cs, 1] - 0.004) + (J[cs, 1] + 0.008 - (N[cs, 1] - 0.004)) * f
        newV.append(np.c_[cols[cs], yy, zz])
    nc = len(cs)
    base = len(V)
    newF = []
    for r_ in range(rows - 1):
        for c in range(nc - 1):
            a0, a1 = base + r_ * nc + c, base + r_ * nc + c + 1
            b0, b1 = a0 + nc, a1 + nc
            newF += [[a0, a1, b1], [a0, b1, b0]]
    newV = np.vstack(newV); newF = np.array(newF)
    T = newV[newF - base]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = nrm[:, 2] - 0.5 * nrm[:, 1] < 0                # 朝前、略朝下
    newF[flip] = newF[flip][:, ::-1]
    out = trimesh.Trimesh(np.vstack([V, newV]), np.vstack([Fc, newF]), process=False,
                          visual=trimesh.visual.TextureVisuals(uv=np.vstack([UV, np.tile(uv0, (len(newV), 1))]), material=mesh.visual.material))
    print(f"jaw bridge: {nc} columns, neck top y {np.nanmean(N[cs, 0]):.3f} z {np.nanmean(N[cs, 1]):.3f} -> jaw y {np.nanmean(J[cs, 0]):.3f} z {np.nanmean(J[cs, 1]):.3f}")
    return out


def clear_chest_front(mesh, colored, cfg=RESHAPE):
    """胸口皮膚的範圍內，如果有白色內襯的碎片貼在皮膚前面（2 cm 內），皮膚和布的交界就會跟著碎片的破邊參差。
    刪掉這些碎片：交界只由貼圖上的收邊線決定，是一條乾淨的線。"""
    if CHEST_POLY is None or CHEST_EDGE is None:
        return
    import cv2
    H_, W_ = colored.shape
    cdil = cv2.dilate(colored.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    uv = np.asarray(mesh.visual.uv)
    Fc = np.array(mesh.faces)
    V = mesh.vertices
    T = V[Fc]; cen = T.mean(1)
    yb_, le_, ri_ = CHEST_EDGE
    y0_, y1_ = cfg["chest_y"][0] + 0.02, min(cfg["chest_y"][1], 1.50)
    idx = np.where((cen[:, 1] > y0_) & (cen[:, 1] < y1_) & (np.abs(cen[:, 0]) < 0.12))[0]
    cx_, cy_ = cen[idx, 0], cen[idx, 1]
    inside = (T[idx][:, :, 0].min(1) > np.interp(cy_, yb_, le_) + 0.0005) & (T[idx][:, :, 0].max(1) < np.interp(cy_, yb_, ri_) - 0.0005)
    xs_, ys_ = cx_ / 0.1, (cy_ - 1.36) / 0.1
    zfit = np.zeros(len(idx)); k_ = 0
    for i in range(5):
        for j in range(5 - i):
            zfit += CHEST_POLY[k_] * xs_ ** i * ys_ ** j
            k_ += 1
    dz = cen[idx, 2] - zfit
    px = np.clip(np.round((uv[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
    py = np.clip(np.round((1 - uv[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
    vcol = cdil[py, px]
    white = ~vcol[Fc[idx]].any(axis=1)
    kill = np.zeros(len(Fc), bool)
    kill[idx[inside & white & (dz > -0.004) & (dz < 0.02)]] = True
    mesh.update_faces(~kill)
    print(f"chest front lining fragments removed: {int(kill.sum())}")


def fix_stray_uv(mesh, color, cfg=RESHAPE):
    """白布的三角形如果取樣到膚色、別種白（裙子暖白／蛹冷白）或很暗的像素（貼圖最下面一排擠在一起的小三角形、
    或 UV 被拉長跨過別的區塊）→ 布面上出現棕線、淡藍色塊。把這些三角形獨立出來，整個改貼到一個乾淨的同色白布像素。"""
    import cv2
    tex = np.asarray(color.convert("RGB")).astype(np.float32) / 255
    H_, W_ = tex.shape[:2]
    V = np.asarray(mesh.vertices); UV = np.asarray(mesh.visual.uv, dtype=np.float64); Fc = np.asarray(mesh.faces)
    T = V[Fc]; cen = T.mean(1)
    lvl = cfg["rim_level"]
    skirt_f = T[:, :, 1].max(1) < lvl - 0.08
    coc_f = (T[:, :, 1].min(1) > 1.26) & ((cen[:, 2] < -0.03) | (np.abs(cen[:, 0]) > 0.11) | (cen[:, 1] > 1.74))
    rb = tex[..., 0] - tex[..., 2]
    Lt = tex.mean(2)
    brown = cv2.dilate((rb > 0.09).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cool = rb < -0.004
    dark = Lt < 0.66
    warm_safe = cv2.erode(((rb > 0.012) & (rb < 0.075) & (Lt > 0.78)).astype(np.uint8), np.ones((25, 25), np.uint8))
    cool_safe = cv2.erode(((rb < 0.0) & (rb > -0.04) & (Lt > 0.74)).astype(np.uint8), np.ones((25, 25), np.uint8))
    uvt = UV[Fc]                                             # (F, 3, 2)
    bw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                   [.7, .15, .15], [.15, .7, .15], [.15, .15, .7]])
    bad_s = np.zeros(len(Fc), bool); bad_c = np.zeros(len(Fc), bool)
    for w_ in bw:
        u_ = (uvt * w_[None, :, None]).sum(1)
        qx = np.clip(np.round((u_[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
        qy = np.clip(np.round((1 - u_[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
        bad_s |= brown[qy, qx] | cool[qy, qx] | dark[qy, qx]
        bad_c |= brown[qy, qx]
    e_uv = np.linalg.norm(uvt - np.roll(uvt, 1, axis=1), axis=2).max(1) * W_
    e_3d = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).max(1)
    stretched = (e_uv > 30) & (e_3d < 0.006)
    fs = skirt_f & (bad_s | stretched)
    fc_ = coc_f & (bad_c | stretched)
    if COLORED_TEX is not None and COLORED_TEX.shape == rb.shape:
        cdil = cv2.dilate(COLORED_TEX.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        vx = np.clip(np.round((UV[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
        vy = np.clip(np.round((1 - UV[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
        vwhite = ~cdil[vy, vx]
        cu0 = uvt.mean(1)
        cwhite = ~cdil[np.clip(np.round((1 - cu0[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1), np.clip(np.round((cu0[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)]
        front_f = (T[:, :, 1].min(1) > 1.26) & (T[:, :, 1].max(1) < 1.50) & (np.abs(cen[:, 0]) < 0.14) & (cen[:, 2] > -0.03)
        n_before = int(fc_.sum())
        fc_ = fc_ | (front_f & cwhite & (vwhite[Fc].sum(axis=1) >= 2) & (bad_c | stretched))
        print(f"stray uv (chest sides): {int(fc_.sum()) - n_before} faces")
    newV, newUV = [], []
    F2 = Fc.copy()
    base = len(V)
    for fsel, safe, name in ((fs, warm_safe, "skirt"), (fc_, cool_safe, "cocoon")):
        idx = np.where(fsel)[0]
        if len(idx) == 0 or not safe.any():
            print(f"stray uv ({name}): {len(idx)} faces, no fix")
            continue
        _, labels = cv2.distanceTransformWithLabels((safe == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        zy, zx = np.nonzero(safe)
        lut = np.zeros((labels.max() + 1, 2), np.int64)
        lut[labels[zy, zx]] = np.c_[zy, zx]
        cu = uvt[idx].mean(1)
        qx = np.clip(np.round((cu[:, 0] % 1) * (W_ - 1)).astype(int), 0, W_ - 1)
        qy = np.clip(np.round((1 - cu[:, 1] % 1) * (H_ - 1)).astype(int), 0, H_ - 1)
        tgt = lut[labels[qy, qx]]
        tuv = np.c_[tgt[:, 1] / (W_ - 1), 1 - tgt[:, 0] / (H_ - 1)]
        newV.append(V[Fc[idx]].reshape(-1, 3))
        newUV.append(np.repeat(tuv, 3, axis=0))
        F2[idx] = base + np.arange(len(idx) * 3).reshape(-1, 3)
        base += len(idx) * 3
        print(f"stray uv ({name}): {len(idx)} faces re-mapped to clean white")
    if not newV:
        return mesh
    out = trimesh.Trimesh(np.vstack([V] + newV), F2, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=np.vstack([UV] + newUV), material=mesh.visual.material))
    out.remove_unreferenced_vertices()
    return out


def _env_r(env, th_s, yy):
    """包絡面半徑（雙線性內插）"""
    G, nt_, ny_ = env["sm"], env["nt"], env["ny"]
    tf = (th_s + np.pi) / (2 * np.pi) * nt_ - 0.5
    yf = np.clip((yy - env["ybase"]) / env["dy"] - 0.5, 0, ny_ - 1)
    t0 = np.floor(tf).astype(int); ft = tf - t0
    j0 = np.clip(np.floor(yf).astype(int), 0, ny_ - 2); fy = yf - j0
    g = lambda jj, tt: G[jj, tt % nt_]
    return (g(j0, t0) * (1 - ft) + g(j0, t0 + 1) * ft) * (1 - fy) + (g(j0 + 1, t0) * (1 - ft) + g(j0 + 1, t0 + 1) * ft) * fy


def smooth_skirt_top(mesh, cfg=RESHAPE):
    """裙子上段（臀部骨架那圈）原始網格是好幾層布片疊在一起，有裂縫、翻起的布角。
    以每個方位、每個高度的最外層半徑為「包絡面」，平滑後把外層貼上去、內層壓到它下面 3 mm →
    外觀是一張乾淨連續的布，大形狀（往外鼓的弧度）不變。"""
    if WAIST_ELL is None or not cfg.get("top_env"):
        return
    y0, sig_t, sig_y = cfg["top_env"]
    a = WAIST_ELL[4]
    lvl = cfg["rim_level"]
    V = mesh.vertices.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    y1 = lvl + 0.004
    zone = (y > y0 - 0.03) & (y < y1)
    ref = (y > y0 - 0.03) & (y < lvl)
    cx, cz = float(np.median(x[ref])), float(np.median(z[ref]))
    idx = np.where(zone)[0]
    dx, dz = x[idx] - cx, z[idx] - cz
    r = np.hypot(dx, dz)
    th = np.arctan2(dx, dz)
    nt, dy = 144, 0.005
    ny = int(np.ceil((y1 - (y0 - 0.03)) / dy))
    ti = np.clip(((th + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)
    yi = np.clip(((y[idx] - (y0 - 0.03)) / dy).astype(int), 0, ny - 1)
    cell = yi * nt + ti
    env = np.full(ny * nt, np.nan)
    order = np.argsort(cell)
    cs, rs_ = cell[order], r[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    for seg_c, seg_r in zip(np.split(cs, bounds), np.split(rs_, bounds)):
        if len(seg_r) >= 4:
            env[seg_c[0]] = np.percentile(seg_r, 97)
    env = env.reshape(ny, nt)
    # 補空格：沿方位內插，再沿高度往上延伸
    for j in range(ny):
        ok = ~np.isnan(env[j])
        if ok.sum() >= 8:
            env[j] = np.interp(np.arange(nt), np.arange(nt)[ok], env[j][ok], period=nt)
    for j in range(1, ny):
        if np.isnan(env[j]).any():
            env[j] = env[j - 1]
    for j in range(ny - 2, -1, -1):
        if np.isnan(env[j]).any():
            env[j] = env[j + 1]
    # 個別格子只量到裡面的東西（外層在那裡剛好沒有頂點）→ 半徑遠小於同高度的鄰居：視為空格，用鄰居補
    for j in range(ny):
        row = env[j]
        med = np.median(np.stack([np.roll(row, k) for k in range(-8, 9)]), axis=0)
        bad = row < med * 0.85
        if bad.any() and (~bad).sum() >= 8:
            env[j] = np.interp(np.arange(nt), np.arange(nt)[~bad], row[~bad], period=nt)
    raw = env.copy()
    import cv2
    pad = int(3 * sig_t) + 1
    e2 = np.concatenate([env[:, -pad:], env, env[:, :pad]], axis=1).astype(np.float32)
    e2 = cv2.medianBlur(e2, 5)
    e2 = cv2.GaussianBlur(e2, (0, 0), sigmaX=sig_t, sigmaY=sig_y, borderType=cv2.BORDER_REPLICATE)
    sm_env = e2[:, pad:-pad].astype(np.float64)
    # 每一圈只留低頻（8 階以下）：輪廓是平順的橢圓，不留局部的凹窩；最下面 5 cm 漸漸回到實際表面（和下面的裙子接得上）
    lowp = sm_env.copy()
    for j in range(ny):
        Fq = np.fft.rfft(sm_env[j]); Fq[9:] = 0
        lowp[j] = np.fft.irfft(Fq, nt)
    wrow = np.clip((np.arange(ny) * dy - 0.03) / 0.05, 0, 1)[:, None]
    sm_env = sm_env * (1 - wrow) + lowp * wrow
    def samp(G):
        tf = (th + np.pi) / (2 * np.pi) * nt - 0.5
        yf = np.clip((y[idx] - (y0 - 0.03)) / dy - 0.5, 0, ny - 1)
        t0 = np.floor(tf).astype(int); ft = tf - t0
        j0 = np.clip(np.floor(yf).astype(int), 0, ny - 2); fy = yf - j0
        g = lambda jj, tt: G[jj, tt % nt]
        return (g(j0, t0) * (1 - ft) + g(j0, t0 + 1) * ft) * (1 - fy) + (g(j0 + 1, t0) * (1 - ft) + g(j0 + 1, t0 + 1) * ft) * fy
    r_raw, r_sm = samp(raw), samp(sm_env)
    d = np.maximum(r_raw - r, 0)                      # 在最外層底下多深
    outer = d < 0.008
    r_new = np.where(outer, r_sm - d * 0.3, r_sm - 0.004 - np.minimum(d - 0.008, 0.06) * 0.15)
    ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    w = ss((y[idx] - (y0 - 0.03)) / 0.04) * (d < 0.07)   # 下端漸入；太深的（藏在裡面的身體）不動
    r2 = r + np.clip(r_new - r, -0.03, 0.03) * w   # 單點最多移 3 cm（整理皺褶，不改大形狀）
    V[idx, 0] = cx + dx / np.maximum(r, 1e-6) * r2
    V[idx, 2] = cz + dz / np.maximum(r, 1e-6) * r2
    mesh.vertices = V
    global SKIRT_ENV
    SKIRT_ENV = dict(cx=cx, cz=cz, ybase=y0 - 0.03, dy=dy, nt=nt, ny=ny, sm=sm_env)
    print(f"skirt top envelope: {int((w > 0.01).sum())} vertices, mean move {np.abs(r2 - r)[w > 0.01].mean() * 1000:.1f} mm, max {np.abs(r2 - r).max() * 1000:.0f} mm")


def cap_shelf(mesh, color, xyz_tex=None, glow=None, emissive=None, cfg=RESHAPE):
    """裙子上緣內側（腰與裙子翻邊之間）原本是一團衣服碎片 → 整片刪掉，換成乾淨的白色平台（略往腰升高），
    腰身底部補一圈皮膚帶接到平台。平台外緣藏在翻邊頂端下方，從外面看是俐落的一圈。"""
    if WAIST_ELL is None:
        return mesh
    cx, cz, rx, rz, a, b2 = WAIST_ELL
    lvl = cfg["rim_level"]
    V, Fc, UV = mesh.vertices, mesh.faces, mesh.visual.uv
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    r = np.hypot(x - cx, z - cz)
    th = np.arctan2(x - cx, z - cz)
    re = 1 / np.sqrt((np.sin(th) / rx) ** 2 + (np.cos(th) / rz) ** 2)
    nb = 120
    bi = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    near = (y > lvl - 0.05) & (y < lvl + 0.003) & (r > re * 1.08)
    Rr, Yt = np.full(nb, np.nan), np.full(nb, np.nan)
    for j in range(nb):
        t_ = near & (bi == j)
        if t_.sum() >= 8:
            Rr[j] = np.percentile(r[t_], 98)
            lip = t_ & (r > Rr[j] - 0.015)
            Yt[j] = np.percentile(y[lip], 97) if lip.sum() >= 4 else np.nan
    for A in (Rr, Yt):
        ok = ~np.isnan(A)
        A[:] = np.interp(np.arange(nb), np.arange(nb)[ok], A[ok], period=nb)
        A[:] = np.median(np.stack([np.roll(A, k) for k in range(-3, 4)]), axis=0)  # 去掉個別方位的離群值
        if A is Yt:
            A[:] = np.min(np.stack([np.roll(A, k) for k in range(-2, 3)]), axis=0)  # 上緣修齊到每一處都到得了的高度
        A[:] = np.convolve(np.r_[A[-3:], A, A[:3]], np.ones(7) / 7, mode="valid")
    Yt[:] = np.median(Yt)                       # 骨架圈是平的：整圈同一個高度
    Fr = np.fft.rfft(Rr); Fr[6:] = 0
    Rr[:] = np.fft.irfft(Fr, nb)                # 輪廓只留低頻（平順的橢圓），不跟著布邊的凹凸走
    if SKIRT_ENV is not None and cfg.get("top_wall", True):
        te = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        Re = _env_r(SKIRT_ENV, te, np.full(360, float(Yt[0]) - 0.03)) + cfg.get("cuff_thick", 0.006)
        ex, ez = SKIRT_ENV["cx"] + Re * np.sin(te), SKIRT_ENV["cz"] + Re * np.cos(te)
        tw, rw = np.arctan2(ex - cx, ez - cz), np.hypot(ex - cx, ez - cz)
        od_ = np.argsort(tw)
        Rr[:] = np.interp((np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi, tw[od_], rw[od_], period=2 * np.pi)
    ang = lambda A: np.interp((th + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), A, period=nb)
    rv, yt = ang(Rr), ang(Yt)
    y_out = yt - 0.005                      # 平台外緣：翻邊頂端下方
    y_in = np.minimum(y_out + cfg["shelf_rise"], a - 0.002)
    tt = np.clip((rv - r) / np.maximum(rv - re, 0.01), 0, 1)
    y_cap = y_out + (y_in - y_out) * tt
    # 刪除：翻邊內側、平台以上到腰下緣（含腰身最下面一小段）；腰下緣往上 3 cm 內不在腰面上的碎片
    inside = r < rv - 0.008
    # 腰最下面 3 cm 整段重做（原本這段不在橢圓上，後腰會缺一塊）
    kill_v = inside & (y > y_cap - 0.002) & (y < a + 0.03)
    # 裙子上緣參差的布邊：修齊到平順的高度（之後包上一圈滾邊）
    # 蛹的下緣（每個方位最低的那一圈）：之後用一片底面把腰和蛹下緣之間封起來，下緣包滾邊
    selS = (y > b2 - 0.01) & (y < b2 + 0.09) & (r > re * 1.12) & (r < re * 3.4)
    Rl, Yl = np.full(nb, np.nan), np.full(nb, np.nan)
    for j in range(nb):
        t_ = np.where(selS & (bi == j))[0]
        if len(t_) >= 12:
            low = t_[y[t_] <= np.percentile(y[t_], 12)]
            Rl[j], Yl[j] = np.median(r[low]), np.median(y[low])
    for A in (Rl, Yl):
        ok = ~np.isnan(A)
        A[:] = np.interp(np.arange(nb), np.arange(nb)[ok], A[ok], period=nb)
        A[:] = np.median(np.stack([np.roll(A, k) for k in range(-3, 4)]), axis=0)
        A[:] = np.convolve(np.r_[A[-3:], A, A[:3]], np.ones(7) / 7, mode="valid")
    rl, yl = ang(Rl), ang(Yl)
    # 底面以下、腰以外的碎邊全部清掉
    tt_d = np.clip((r - re) / np.maximum(rl + 0.01 - re, 0.01), 0, 1)
    y_dia = (b2 - 0.002) + (yl + 0.012 - (b2 - 0.002)) * tt_d
    kill_v = kill_v | ((r > re * 1.03) & (r < rl - 0.004) & (y > b2 - 0.02) & (y < y_dia - 0.003))
    # 蛹下緣垂下來的毛邊修齊（之後折一道收邊）
    kill_v = kill_v | ((np.abs(r - rl) < 0.06) & (y > b2 - 0.05) & (y < yl - 0.0015))
    hoop = cfg.get("hoop", 0.012)
    ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    env = SKIRT_ENV if cfg.get("top_wall", True) else None
    if env is not None:
        def env_r(th_s, yy):
            G, nt_, ny_ = env["sm"], env["nt"], env["ny"]
            tf = (th_s + np.pi) / (2 * np.pi) * nt_ - 0.5
            yf = np.clip((yy - env["ybase"]) / env["dy"] - 0.5, 0, ny_ - 1)
            t0 = np.floor(tf).astype(int); ft = tf - t0
            j0 = np.clip(np.floor(yf).astype(int), 0, ny_ - 2); fy = yf - j0
            g = lambda jj, tt: G[jj, tt % nt_]
            return (g(j0, t0) * (1 - ft) + g(j0, t0 + 1) * ft) * (1 - fy) + (g(j0 + 1, t0) * (1 - ft) + g(j0 + 1, t0 + 1) * ft) * fy
        y0w = cfg["top_env"][0]
        r_s = np.hypot(x - env["cx"], z - env["cz"])
        th_s = np.arctan2(x - env["cx"], z - env["cz"])
        kill_v |= (y > y0w) & (y < a + 0.03) & (r_s > env_r(th_s, y) - 0.03)
    kill_v |= ~inside & (r < rv + 0.04) & (y > yt - hoop) & (y < a + 0.03)
    # 腰：整段重做（cap 到蛹下緣），不留原本投影出來的面 → 沒有補過的接縫
    kill_v |= (r < re * 1.08 + 0.003) & (y >= a + 0.03) & (y < b2 - 0.001)
    # 腰外面、裙子上緣與蛹下緣之間的空檔：不該有任何東西（殘留的布片、拉長的三角形）
    kill_v |= (y > a + 0.03) & (y < b2 - 0.02) & (r < rv + 0.02)
    kill_f = kill_v[Fc].any(axis=1)
    # 取色：腰身皮膚、裙子白布各取一個代表性的 UV
    W_, H_ = color.size
    col = np.asarray(color.convert("RGB")).astype(np.float32)
    def pick_uv(sel):
        ids = np.where(sel)[0]
        ids = ids[np.linspace(0, len(ids) - 1, min(400, len(ids))).astype(int)]
        u = UV[ids] % 1
        c = col[np.clip(((1 - u[:, 1]) * (H_ - 1)).astype(int), 0, H_ - 1), np.clip((u[:, 0] * (W_ - 1)).astype(int), 0, W_ - 1)]
        med = np.median(c, axis=0)
        return u[np.argmin(np.abs(c - med).sum(axis=1))], med
    import cv2
    wcurve = lambda yy: 1 - 0.07 * np.sin(np.pi * np.clip((yy - a) / (b2 - a), 0, 1))  # 腰身內收的弧度（和 reshape_v2 相同）
    c_skin = SKIN_RGB * 255 * 0.98 if SKIN_RGB is not None else pick_uv((y > a + 0.035) & (y < b2 - 0.02) & (np.abs(r - re * wcurve(y)) < 0.004))[1]
    # 皮膚：取貼圖上腰部膚色區塊最中心的點（離邊界最遠 → 縮小取樣不會混到旁邊的白布）
    sm_ = (np.abs(col - c_skin).max(axis=2) < 5).astype(np.uint8)
    dt = cv2.distanceTransform(sm_, cv2.DIST_L2, 3)
    py_, px_ = np.unravel_index(np.argmax(dt), dt.shape)
    uv_skin = np.array([px_ / (W_ - 1), 1 - py_ / (H_ - 1)])
    # 白布：取裙子針織布最大的一塊，平台用它鋪（材質和下面的裙子一致）
    patch = None
    if xyz_tex is not None:
        Yt_ = np.nan_to_num(xyz_tex[..., 1], nan=-1.0)
        km = ((Yt_ > 0.35) & (Yt_ < 0.85)).astype(np.uint8)
        dk = cv2.distanceTransform(km, cv2.DIST_L2, 3)
        ky, kx = np.unravel_index(np.argmax(dk), dk.shape)
        half = int(dk[ky, kx] / 1.5)
        # 候選：夠大的區塊裡，挑「大範圍明暗均勻（沒有摺痕、陰影）、細紋理明顯（針織的顆粒）」的位置
        Lc = col.mean(2) / 255.0
        sc2 = W_ / 1024.0
        kq = int(61 * sc2) | 1
        Lb = cv2.GaussianBlur(Lc, (0, 0), 6 * sc2)
        m1, m2 = cv2.blur(Lb, (kq, kq)), cv2.blur(Lb * Lb, (kq, kq))
        lf = np.sqrt(np.maximum(m2 - m1 * m1, 0))
        hf = cv2.blur(np.abs(Lc - cv2.GaussianBlur(Lc, (0, 0), 2 * sc2)), (kq, kq))
        candm = dk >= 44 * sc2
        if candm.any():
            score = np.where(candm, lf - 0.6 * hf, 1e9)
            ky, kx = np.unravel_index(np.argmin(score), score.shape)
            half = int(30 * sc2)
        if half >= 12:
            mpp = np.linalg.norm(xyz_tex[ky, kx + 4] - xyz_tex[ky, kx - 4]) / 8  # 每像素幾公尺
            gh = np.array([Yt_[ky, kx + 4] - Yt_[ky, kx - 4], Yt_[ky + 4, kx] - Yt_[ky - 4, kx]])  # 貼圖上「往上」的方向
            gh = gh / max(np.linalg.norm(gh), 1e-9) if np.linalg.norm(gh) > 1e-6 else np.array([0.0, -1.0])
            patch = (kx, ky, half, mpp, gh)
            print(f"knit patch at ({kx},{ky}) half {half}px, {mpp * 1000:.2f} mm/px, up dir {np.round(gh, 2)}")
    uv_white, c_white = pick_uv((y > 0.5) & (y < 0.8) & (z > 0.03))
    # 蛹的白布（會發光的那種）：蛹下緣滾邊用
    uv_shell = uv_white
    if glow is not None:
        _, c_shell = pick_uv((y > 1.3) & (y < 1.42) & (z < cz - 0.08))
        gl = np.asarray(Image.fromarray((glow * 255).astype(np.uint8)).resize((W_, H_))) > 128 if glow.shape != (H_, W_) else glow > 0.5
        sh_ = ((np.abs(col - c_shell).max(axis=2) < 8) & gl).astype(np.uint8)
        dsh = cv2.distanceTransform(sh_, cv2.DIST_L2, 3)
        py2, px2 = np.unravel_index(np.argmax(dsh), dsh.shape)
        uv_shell = np.array([px2 / (W_ - 1), 1 - py2 / (H_ - 1)])
    print(f"cap shelf: removed faces {int(kill_f.sum())}, skin {c_skin.round()}, white {c_white.round()}")
    # 新幾何：平台（環狀，內緣在腰裡 0.6 倍處）＋ 腰底皮膚帶
    na = 240
    ta = np.linspace(-np.pi, np.pi, na, endpoint=False)
    rea = 1 / np.sqrt((np.sin(ta) / rx) ** 2 + (np.cos(ta) / rz) ** 2)
    rva = np.interp((ta + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), Rr, period=nb)
    yta = np.interp((ta + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), Yt, period=nb)
    yo = yta - 0.005
    yi = np.minimum(yo + cfg["shelf_rise"], a - 0.002)
    rings = []  # (半徑, 高度, uv)
    for t in np.linspace(0, 1, 9):          # 外緣 → 腰
        rr = (rva - 0.006) + (rea - (rva - 0.006)) * t
        rings.append((rr, yo + (yi - yo) * (3 * t * t - 2 * t ** 3), uv_white))
    rings.append((rea * 0.6, yi, uv_white))
    # 腰的膚色：在貼圖的空白處畫一小塊漸層（上緣在蛹的陰影裡較暗、下緣略暗、前後中線有淡淡的凹影）
    waist_uv = None
    if xyz_tex is not None:
        free = np.isnan(xyz_tex[..., 1]).astype(np.uint8)
        dfree = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        fy, fx = np.unravel_index(np.argmax(dfree), dfree.shape)
        hp = min(int(dfree[fy, fx] / 1.5), int(48 * W_ / 1024))
        if hp >= 12:
            n_ = 2 * hp
            uu, vv = np.meshgrid(np.linspace(0, 1, n_), np.linspace(0, 1, n_))   # uu: 0 正前 → 1 正後；vv: 0 上緣 → 1 下緣
            shd = 1 - 0.24 * np.exp(-vv / 0.22) - 0.07 * np.exp(-(1 - vv) / 0.15)
            tile = np.clip(c_skin[None, None, :] / 0.98 * shd[..., None], 0, 255).astype(np.uint8)
            color.paste(Image.fromarray(tile), (fx - hp, fy - hp))
            if emissive is not None:
                ke = emissive.size[0] / W_
                emissive.paste(0, (int((fx - hp) * ke), int((fy - hp) * ke), int((fx + hp) * ke) + 1, int((fy + hp) * ke) + 1))
            def waist_uv(yk):
                u_ = np.abs(ta) / np.pi
                v_ = 1 - np.clip((yk - a) / (b2 - a), 0, 1)
                px_w = fx - hp + 12 + u_ * (n_ - 25)
                py_w = fy - hp + 8 + v_ * (n_ - 17)
                return np.c_[px_w / (W_ - 1), 1 - py_w / (H_ - 1)]
            print(f"waist skin tile at ({fx},{fy}) half {hp}px")
    skin = []
    nk = 28
    for k in range(nk + 1):
        yk = yi - 0.001 + (b2 + 0.008 - yi + 0.001) * k / nk   # 往上伸進蛹下緣裡面墊著（蛹下緣若有細縫，看到的是皮膚不是黑洞）
        uvk = waist_uv(yk) if waist_uv is not None else uv_skin
        skin.append((rea * wcurve(yk) * np.where(yk < b2 - 0.003, 1.0, 0.985), yk, uvk))
    newV, newUV, newF = [], [], []
    base = len(V)
    def strip(rs, want, closed=False, vcoord=None):
        """rs: [(半徑, 高度, uv)]；want(面中心) → 希望法線朝的方向；closed: 斷面也繞一圈（管狀）"""
        nonlocal base
        idx0 = base
        pts = []
        for k, (rr, yy, uv) in enumerate(rs):
            pv = rr if (isinstance(rr, np.ndarray) and rr.ndim == 2) else np.c_[cx + rr * np.sin(ta), yy * np.ones(na), cz + rr * np.cos(ta)]
            pts.append(pv)
            newV.append(pv)
            if uv.ndim == 2:
                newUV.append(uv)                  # 逐點指定的 UV（腰的漸層）
            elif uv is uv_white and patch is not None:
                # 針織布：以實際尺寸鋪貼，超出區塊就鏡射折返（接縫連續）
                kx, ky, half, mpp, gh = patch
                tri_ = lambda t: 0.7 * half * (2 * np.abs((t / (1.4 * half) + 0.5) % 2 - 1) - 1)
                if vcoord is None:
                    ca, cb = tri_(pv[:, 0] / mpp), tri_(pv[:, 2] / mpp)
                else:
                    ca, cb = tri_(ta * 0.2 / mpp), tri_(np.full(na, vcoord[k]) / mpp)   # ca 沿圓周、cb 沿高度
                du, dv = ca * (-gh[1]) + cb * gh[0], ca * gh[0] + cb * gh[1]            # 高度方向對齊貼圖上針織紋的「往上」
                newUV.append(np.c_[(kx + du) / (W_ - 1), 1 - (ky + dv) / (H_ - 1)])
            else:
                newUV.append(np.tile(uv, (na, 1)))
        fs = []
        nr = len(rs)
        for k in range(nr if closed else nr - 1):
            i0, i1 = idx0 + k * na, idx0 + ((k + 1) % nr) * na
            j = np.arange(na); j2 = (j + 1) % na
            fs.append(np.c_[i0 + j, i1 + j, i0 + j2])
            fs.append(np.c_[i0 + j2, i1 + j, i1 + j2])
        fs = np.vstack(fs)
        allp = np.vstack(pts)
        tri = allp[fs - idx0]
        fn_ = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if (fn_ * want(tri.mean(1))).sum() < 0:
            fs = fs[:, ::-1]                      # 整條翻面：正面朝外（背面會被當成布的內側畫成白色）
        newF.append(fs)
        base += nr * na
    axis_out = lambda c: np.c_[c[:, 0] - cx, np.zeros(len(c)), c[:, 2] - cz]
    strip(rings, lambda c: np.tile([0.0, 1.0, 0.0], (len(c), 1)))
    strip(skin, axis_out)
    # 滾邊（圓管）：斷面 10 邊
    def tube(rc, yc, rad, uv):
        ph = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        flat = 0.45 if rad < 0.008 else 1.0  # 一般衣服的收邊：薄薄一道折邊（不是圓鼓的滾邊）
        rs_ = [(rc + rad * np.cos(q), yc + rad * flat * np.sin(q), uv) for q in ph]
        def want(c):
            tc = np.arctan2(c[:, 0] - cx, c[:, 2] - cz)
            rcc = np.interp(tc, np.r_[ta, np.pi], np.r_[rc, rc[0]])
            ycc = np.interp(tc, np.r_[ta, np.pi], np.r_[yc * np.ones(na), (yc * np.ones(na))[0]])
            rr_ = np.hypot(c[:, 0] - cx, c[:, 2] - cz)
            k_ = (rr_ - rcc) / np.maximum(rr_, 1e-6)
            return np.c_[(c[:, 0] - cx) * k_, c[:, 1] - ycc, (c[:, 2] - cz) * k_]
        strip(rs_, want, closed=True, vcoord=ph * rad)
    if env is not None:
        # 裙子上段外牆：沿包絡面一圈一圈往上，最上面接到骨架圈（圓管）的外緣；最下面 2 cm 略往內收，藏在原本的布底下
        cxs, czs = env["cx"], env["cz"]
        tcx = cx + (rva - hoop + 0.001) * np.sin(ta)
        tcz = cz + (rva - hoop + 0.001) * np.cos(ta)
        tcy = yta - hoop + 0.001
        th_t = np.arctan2(tcx - cxs, tcz - czs)
        r_t = np.hypot(tcx - cxs, tcz - czs)
        od = np.argsort(th_t)
        Rtop = np.interp(ta, th_t[od], r_t[od], period=2 * np.pi) + hoop
        Ytop = np.interp(ta, th_t[od], tcy[od], period=2 * np.pi)
        ybot = y0w - 0.04
        ntl, y_l0, dyl = 72, ybot - 0.02, 0.01
        nyl = int(np.ceil((y0w + 0.02 - y_l0) / dyl))
        rb_ = np.hypot(x - cxs, z - czs); tb_ = np.arctan2(x - cxs, z - czs)
        sl_ = (y > y_l0) & (y < y_l0 + nyl * dyl) & (rb_ < 0.4)
        cl_ = np.clip(((y[sl_] - y_l0) / dyl).astype(int), 0, nyl - 1) * ntl + np.clip(((tb_[sl_] + np.pi) / (2 * np.pi) * ntl).astype(int), 0, ntl - 1)
        gl_ = np.full(nyl * ntl, np.nan)
        odl = np.argsort(cl_); csl, rsl = cl_[odl], rb_[sl_][odl]
        cutl = np.flatnonzero(np.diff(csl)) + 1
        for sc_, sr_ in zip(np.split(csl, cutl), np.split(rsl, cutl)):
            if len(sr_) >= 4:
                gl_[sc_[0]] = np.percentile(sr_, 96)
        gl_ = gl_.reshape(nyl, ntl)
        for j in range(nyl):
            okl = ~np.isnan(gl_[j])
            gl_[j] = np.interp(np.arange(ntl), np.arange(ntl)[okl], gl_[j][okl], period=ntl) if okl.sum() >= 6 else np.nan
        for j in range(1, nyl):
            if np.isnan(gl_[j]).any():
                gl_[j] = gl_[j - 1]
        for j in range(nyl - 2, -1, -1):
            if np.isnan(gl_[j]).any():
                gl_[j] = gl_[j + 1]
        gl2 = np.concatenate([gl_[:, -6:], gl_, gl_[:, :6]], axis=1).astype(np.float32)
        gl2 = cv2.GaussianBlur(gl2, (0, 0), sigmaX=2.0, sigmaY=0.8)[:, 6:-6]
        tfl = (ta + np.pi) / (2 * np.pi) * ntl - 0.5
        gla = np.stack([np.interp(tfl, np.arange(ntl), gl2[j], period=ntl) for j in range(nyl)])   # (nyl, na)

        def loc_r(yy):
            jf = np.clip((yy - y_l0) / dyl - 0.5, 0, nyl - 1.001)
            j0 = np.floor(jf).astype(int); fj = jf - j0
            qa = np.arange(na)
            return gla[j0, qa] * (1 - fj) + gla[np.minimum(j0 + 1, nyl - 1), qa] * fj
        radial = lambda c: np.c_[c[:, 0] - cxs, np.zeros(len(c)), c[:, 2] - czs]
        ring = lambda R, yy: np.c_[cxs + R * np.sin(ta), yy * np.ones(na), czs + R * np.cos(ta)]
        # 裙身：一路延伸到翻折布的底下（最下面 3 cm 略蓋在原本裙子的外面，遮住切口）
        body, vcb = [], []
        Kb = int(np.ceil((Ytop.max() - 0.004 - ybot) / 0.004))
        for k in range(Kb + 1):
            f = k / Kb
            yk = ybot + (Ytop - 0.004 - ybot) * f
            b_ = ss((f - 0.75) / 0.25)
            R = env_r(ta, yk) * (1 - b_) + (Rtop - 0.004) * b_
            if np.min(yk) < y0w + 0.03:
                Rl_ = loc_r(yk * np.ones(na)) + 0.0025        # 下面裙子實際的表面，外加 2.5 mm
                wb_ = 1 - ss((yk - ybot) / 0.07)
                R = np.maximum(R * (1 - wb_) + Rl_ * wb_, np.where(yk < y0w + 0.01, Rl_, 0))
            body.append((ring(R, yk), None, uv_white))
            vcb.append(float(np.mean(yk)))
        strip(body, radial, vcoord=vcb)
        # 2026-11-20 使用者照片：裙子上緣是一片往外翻折下來的布，下緣在正前方較高，往兩側以弧線下降
        thk = cfg.get("cuff_thick", 0.006)
        d0, d1 = cfg.get("cuff", (0.055, 0.15))
        dcuff = d0 + (d1 - d0) * np.sin(np.clip(np.abs(ta) / np.radians(85), 0, 1) * np.pi / 2)
        ycb = Ytop - dcuff
        Rb_ = env_r(ta, ycb)
        cuff, vcc = [], []
        cuff.append((ring(Rb_ + 0.0005, ycb + 0.004), None, uv_white))   # 布邊往內折：下緣有厚度
        vcc.append(-0.008)
        cuff.append((ring(Rb_ + thk * 0.6, ycb + 0.0005), None, uv_white))
        vcc.append(-0.003)
        Kc = 40
        for k in range(Kc + 1):
            f = k / Kc
            yk = ycb + 0.002 + (Ytop - ycb - 0.002) * f
            b_ = ss((f - 0.7) / 0.3)
            R = (env_r(ta, yk) + thk) * (1 - b_) + Rtop * b_
            cuff.append((ring(R, yk), None, uv_white))
            vcc.append(float(np.mean(yk - ycb)))
        # 上緣：布翻過骨架圈（朝外 → 朝上 → 朝內），和翻折布是同一張
        for q in np.linspace(0, np.pi, 11)[1:]:
            cuff.append((ring((Rtop - hoop) + hoop * np.cos(q), Ytop + hoop * np.sin(q)), None, uv_white))
            vcc.append(vcc[-1] + hoop * np.pi / 10)
        strip(cuff, radial, vcoord=vcc)
        print(f"skirt top rebuilt: body {Kb + 1} rings from y {ybot:.3f}; cuff depth {d0}~{d1} m")
    if cfg.get("tail_liner", True):
        selT = np.where((y > 0.06) & (y < 0.50))[0]
        cxt, czt = float(np.median(x[selT])), float(np.median(z[selT]))
        rT = np.hypot(x[selT] - cxt, z[selT] - czt)
        tT = np.arctan2(x[selT] - cxt, z[selT] - czt)
        ntT, dyT = 72, 0.01
        nyT = int(round((0.50 - 0.06) / dyT))
        cellT = np.clip(((y[selT] - 0.06) / dyT).astype(int), 0, nyT - 1) * ntT + np.clip(((tT + np.pi) / (2 * np.pi) * ntT).astype(int), 0, ntT - 1)
        gT = np.full(nyT * ntT, np.nan)
        odT = np.argsort(cellT)
        csT, rsT = cellT[odT], rT[odT]
        bT = np.flatnonzero(np.diff(csT)) + 1
        for sc_, sr_ in zip(np.split(csT, bT), np.split(rsT, bT)):
            if len(sr_) >= 4:
                gT[sc_[0]] = np.percentile(sr_, 8)
        gT = gT.reshape(nyT, ntT)
        for j in range(nyT):
            ok = ~np.isnan(gT[j])
            if ok.sum() >= 6:
                gT[j] = np.interp(np.arange(ntT), np.arange(ntT)[ok], gT[j][ok], period=ntT)
        for j in range(1, nyT):
            if np.isnan(gT[j]).any():
                gT[j] = gT[j - 1]
        for j in range(nyT - 2, -1, -1):
            if np.isnan(gT[j]).any():
                gT[j] = gT[j + 1]
        g2 = np.concatenate([gT[:, -4:], gT, gT[:, :4]], axis=1).astype(np.float32)
        g2 = cv2.GaussianBlur(cv2.medianBlur(g2, 3), (0, 0), 1.5)[:, 4:-4] - 0.008
        liner, vcl = [], []
        for j in range(nyT):
            Rj = np.interp((ta + np.pi) / (2 * np.pi) * ntT - 0.5, np.arange(ntT), g2[j], period=ntT)
            yj = 0.06 + (j + 0.5) * dyT
            liner.append((np.c_[cxt + Rj * np.sin(ta), np.full(na, yj), czt + Rj * np.cos(ta)], None, uv_white))
            vcl.append(yj)
        strip(liner, lambda c: np.c_[c[:, 0] - cxt, np.zeros(len(c)), c[:, 2] - czt], vcoord=vcl)
        print(f"tail liner: {nyT} rings")
    hem = cfg.get("hem", 0)
    if hem:
        if env is None:
            tube(rva - hoop + 0.001, yta - hoop + 0.001, hoop, uv_white)           # （沒有重做外牆時才用圓管）
        rla = np.interp((ta + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), Rl, period=nb)
        yla = np.interp((ta + np.pi) / (2 * np.pi) * nb - 0.5, np.arange(nb), Yl, period=nb)
        # 底面：腰的上緣 → 蛹下緣（略伸進蛹裡），把兩者之間的空隙封起來
        dia = []
        for k in range(7):
            f = k / 6
            dia.append((rea * 0.995 + (rla + 0.01 - rea * 0.995) * f, (b2 - 0.002) + (yla + 0.012 - (b2 - 0.002)) * f, uv_shell))
        strip(dia, lambda c: np.tile([0.0, -1.0, 0.0], (len(c), 1)))
        tube(rla, yla, hem, uv_shell)                                              # 蛹下緣的滾邊
        print(f"shell bottom: edge r {Rl.min():.3f}~{Rl.max():.3f}, y {Yl.min():.3f}~{Yl.max():.3f}")
    V2 = np.vstack([V] + newV)
    UV2 = np.vstack([UV] + newUV)
    F2 = np.vstack([Fc[~kill_f]] + newF)
    out = trimesh.Trimesh(V2, F2, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=UV2, material=mesh.visual.material))
    out.remove_unreferenced_vertices()
    return out


def reshape_v2(mesh, cfg=RESHAPE):
    global CHEST_V, WAIST_ELL
    P, inv, e = _weld(mesh)
    F = inv[mesh.faces]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    sm = lambda a, b, t: np.clip((t - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((t - a) / (b - a), 0, 1))
    if cfg.get("rim_level") is not None:
        P = level_rim(P, cfg)
        P = level_rim(P, cfg)  # 第二次修掉第一次的殘差（沿一圈平滑時左右互相拉扯）
    if cfg.get("hip_sym"):
        P = symmetrize_hips(P, cfg)
    elif cfg.get("rim_balance"):
        P = balance_rim(P, cfg)
    # 胸口（V 形開口的底）：幾何判斷
    floor = chest_floor(P, F, e, cfg) if cfg.get("chest_y") else np.zeros(len(P), bool)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # 整體撫平：隱式平滑一次算出平順的版本，再依區域混合（臉不動、上半身殼與裙子撫平、胸口整片磨平）
    if cfg.get("smooth_s"):
        Ps = _implicit_smooth(P, e, cfg["smooth_s"])
        box = lambda v, a, b, ramp: np.clip((v - a) / ramp + 1, 0, 1) * np.clip((b - v) / ramp + 1, 0, 1)
        facew = box(x, -0.10, 0.10, 0.025) * box(y, 1.47, 1.80, 0.03) * np.clip((z + 0.03) / 0.03, 0, 1)
        w_shell = sm(1.16, 1.20, y) * cfg["shell_smooth"]
        w_skirt = (sm(0.10, 0.18, y) * (1 - sm(1.06, 1.12, y))) * (cfg["skirt_smooth"] - (cfg["skirt_smooth"] - cfg["rim_smooth"]) * sm(0.88, 1.0, y))
        w_shell = w_shell * (1 - facew)
        w_shell = np.where(floor, 1.0, w_shell)
        if cfg.get("skirt_s"):
            # 裙子另外用更強的平滑（作者：裙子皺褶太多）；上半身維持原本強度
            Pk = _implicit_smooth(P, e, cfg["skirt_s"])
            wk = np.clip(w_skirt * (1 - (w_shell > 0)), 0, 1)
            P0 = P
            P = P + np.clip(w_shell, 0, 1)[:, None] * (Ps - P) + wk[:, None] * (Pk - P)
            del Pk
            if cfg.get("detail_keep"):
                # 只撫平大的皺褶：原本的細紋（材質感）加回來；胸口皮膚不加
                Pf = _implicit_smooth(P0, e, cfg["detail_s"])
                wd = np.clip(np.maximum(np.where(floor, 0.0, w_shell), wk), 0, 1) * cfg["detail_keep"]
                P = P + wd[:, None] * (P0 - Pf)
                del Pf
            # 平滑會讓裙子變細 → 每 1 cm 一層，把寬度（x）與深度（z）分別放回原本的大小（比例不變，只去皺褶）
            ys = np.arange(0.08, 1.12, 0.01)
            fx, fz, cxs, czs = [], [], [], []
            for yy in ys:
                s0 = np.abs(P0[:, 1] - yy) < 0.006
                s1 = np.abs(P[:, 1] - yy) < 0.006
                if s0.sum() < 50 or s1.sum() < 50:
                    fx.append(1); fz.append(1); cxs.append(0); czs.append(0); continue
                c0 = np.median(P0[s0][:, [0, 2]], 0); c1 = np.median(P[s1][:, [0, 2]], 0)
                ext = lambda Q, c, k: np.percentile(np.abs(Q[:, k] - c), 97)
                fx.append(ext(P0[s0], c0[0], 0) / max(ext(P[s1], c1[0], 0), 1e-3))
                fz.append(ext(P0[s0], c0[1], 2) / max(ext(P[s1], c1[1], 2), 1e-3))
                cxs.append(c1[0]); czs.append(c1[1])
            # 比例沿高度大幅平滑（σ 約 5 cm）並限制在 1.00～1.12：層與層之間連續，不會出現一圈圈的橫紋
            kg = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2); kg /= kg.sum()
            smg = lambda a: np.convolve(np.pad(np.array(a, float), 12, mode="edge"), kg, "valid")
            fx = smg(np.clip(fx, 1.0, 1.12)); fz = smg(np.clip(fz, 1.0, 1.12))
            cxs, czs = smg(cxs), smg(czs)
            yy_ = P[:, 1]
            gx = 1 + (np.interp(yy_, ys, fx) - 1) * wk
            gz = 1 + (np.interp(yy_, ys, fz) - 1) * wk
            ccx, ccz = np.interp(yy_, ys, cxs), np.interp(yy_, ys, czs)
            P[:, 0] = ccx + (P[:, 0] - ccx) * gx
            P[:, 2] = ccz + (P[:, 2] - ccz) * gz
            print(f"skirt size restore: x ×{fx.min():.3f}～{fx.max():.3f}, z ×{fz.min():.3f}～{fz.max():.3f}")
            del P0
        else:
            w = np.clip(np.maximum(w_shell, w_skirt), 0, 1)
            P = P + w[:, None] * (Ps - P)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        print(f"smooth: s={cfg['smooth_s']}, 平均位移 {np.linalg.norm(Ps - P, axis=1).mean() * 1000:.1f} mm")
    if floor.any() and cfg.get("chest_membrane"):
        # 胸口：邊界（和殼內壁的交界）固定，內部反覆取鄰居平均 → 像一片繃緊的皮膚，領子、摺痕全部消失
        bd = floor & _grow(~floor, e, 1)
        P = _taubin(P, e, (floor & ~bd).astype(np.float64), cfg["chest_membrane"], lam=0.7, mu=0.0)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if floor.any() and cfg.get("chest_fit"):
        # 胸口再用一張平滑曲面（四次多項式）擬合：殘留的領子、門襟、扣子的起伏全部消失；靠邊界 8 圈漸漸接回原形
        depth = np.zeros(len(P))
        er = floor.copy()
        for k_ in range(1, 9):
            er = er & ~_grow(~er, e, 1)
            depth[er] = k_
        fi = np.where(floor)[0]
        xs_, ys_ = P[fi, 0] / 0.1, (P[fi, 1] - 1.36) / 0.1
        A_ = np.stack([xs_ ** i * ys_ ** j for i in range(5) for j in range(5 - i)], 1)
        coef_, *_ = np.linalg.lstsq(A_, P[fi, 2], rcond=None)
        dz_ = np.clip(A_ @ coef_ - P[fi, 2], -0.012, 0.012)
        wf = np.clip(depth[fi] / 8, 0, 1)
        wf = wf * wf * (3 - 2 * wf)
        P[fi, 2] += dz_ * wf * cfg["chest_fit"]
        P = _taubin(P, e, (floor & ~bd).astype(np.float64), 120, lam=0.7, mu=0.0)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        print(f"chest fit: rms residual {np.sqrt((dz_ ** 2).mean()) * 1000:.1f} mm")
    global CHEST_POLY
    if floor.any():
        fi2 = np.where(floor)[0]
        A2 = np.stack([(P[fi2, 0] / 0.1) ** i * ((P[fi2, 1] - 1.36) / 0.1) ** j for i in range(5) for j in range(5 - i)], 1)
        CHEST_POLY, *_ = np.linalg.lstsq(A2, P[fi2, 2], rcond=None)
    chin = np.zeros(len(P), bool)
    if floor.any():
        xb_ = np.round(x / 0.004).astype(int)
        off_ = xb_.min()
        top_ = np.full(xb_.max() - off_ + 1, -1.0)
        fl_ = np.where(floor)[0]
        np.maximum.at(top_, xb_[fl_] - off_, y[fl_])
        ytop = top_[xb_ - off_]
        cand_ = ~floor & (np.abs(x) < 0.045) & (ytop > 1.44) & (y > ytop - 0.002) & (y < ytop + 0.022) & (y < 1.56) & (z > -0.03)
        chin = _grow(floor, e, 24) & cand_
        print("chin band vertices", int(chin.sum()))
    CHEST_V = (floor | chin)[inv]
    if cfg.get("junction"):
        ja, jb = cfg["junction"]
        wj = sm(ja - 0.02, ja + 0.02, y) * (1 - sm(jb - 0.02, jb + 0.02, y)) * 0.85
        P = _taubin(P, e, wj, cfg["junction_iters"])
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # 1. 裙子垂直往下：每 2 cm 一層，半徑小於 skirt_r 的層往外放大到 skirt_r（兩端漸變）
    if cfg.get("straighten_skirt", True):
        y0, y1 = cfg["skirt"]
        ra, rb = cfg["skirt_ref"]
        ref = (y > ra) & (y < rb)
        rcx, rcz = np.median(x[ref]), np.median(z[ref])
        skirt_r = np.percentile(np.hypot(x[ref] - rcx, z[ref] - rcz), 90)
        wy = sm(y0 - 0.05, y0 + 0.04, y) * (1 - sm(y1 - 0.03, y1, y))
        # 每 1 cm 量一次寬度與中心，沿高度平滑後逐點連續套用（不會出現一圈圈的階梯紋）
        ys = np.arange(y0 - 0.06, y1 + 0.005, 0.01)
        cxs, czs, rs = [], [], []
        for yy in ys:
            s_ = np.abs(y - yy) < 0.012
            cx_, cz_ = np.median(x[s_]), np.median(z[s_])
            cxs.append(cx_); czs.append(cz_)
            rs.append(np.percentile(np.hypot(x[s_] - cx_, z[s_] - cz_), 90))
        k5 = np.ones(7) / 7
        smo = lambda arr: np.convolve(np.pad(np.array(arr), 3, mode="edge"), k5, "valid")
        cxs, czs, rs = smo(cxs), smo(czs), smo(rs)
        fs = np.maximum(1.0, skirt_r / np.maximum(rs, 1e-3))
        f_v = 1 + (np.interp(y, ys, fs) - 1) * wy
        ccx, ccz = np.interp(y, ys, cxs), np.interp(y, ys, czs)
        fac = np.where(wy > 0, f_v, 1.0)
        P[:, 0] = ccx + (x - ccx) * fac
        P[:, 2] = ccz + (z - ccz) * fac
    # 2. 腰部：交接處的扭結與垂布收進腰身橢圓（只把超出的收進來），再大量平滑 → 平順的腰
    s = np.zeros(len(P), bool)
    cx = cz = 0.0
    if cfg.get("bare_waist", True):
        a, b2 = cfg["waist"]
        rx, rz = cfg["waist_r"]
        cx, cz = np.median(x[(y > a) & (y < b2)]), np.median(z[(y > a) & (y < b2)])
        if cfg.get("shelf_depth"):
            # 裙子上緣的平台先壓平（布團消失），腰身的作用範圍只從平台上方開始 → 平台不被拉進腰裡
            P = flatten_shelf(P, cfg, (cx, cz, rx, rz))
            x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if cfg.get("waist_shift"):
            P = shift_body(P, cfg, (cx, cz, rx, rz, a, b2))
            cz = cz + cfg["waist_shift"]
            x, y, z = P[:, 0], P[:, 1], P[:, 2]
        WAIST_ELL = (cx, cz, rx, rz, a, b2)
        if cfg.get("shelf_depth"):
            ww = sm(a - 0.003, a + 0.004, y) * (1 - sm(b2 - 0.004, b2 + cfg.get("waist_taper", 0.012), y))
        else:
            ww = sm(a - 0.012, a + 0.006, y) * (1 - sm(b2 - 0.006, b2 + 0.012, y))
        s = ww > 0
        dx, dz = P[:, 0] - cx, P[:, 2] - cz
        q = np.sqrt((dx / rx) ** 2 + (dz / rz) ** 2)
        # 腰身範圍內所有點都貼到平滑的腰身表面（扭結、布片皺摺完全消失）；中間稍微收細，做出自然的腰線
        tw = np.clip((P[:, 1] - a) / (b2 - a), 0, 1)
        curve = 1 - 0.07 * np.sin(np.pi * tw)
        k = curve / np.maximum(q, 1e-6)
        P[:, 0] = cx + dx * (1 + (k - 1) * ww)
        P[:, 2] = cz + dz * (1 + (k - 1) * ww)
        P = _taubin(P, e, np.clip(ww * 1.0, 0, 1), 60)
        if cfg.get("back_slope"):
            P = back_slope(P, cfg, WAIST_ELL)
    # 脖子（下巴以下到胸口之間）：原本是襯衫領子的形狀，起伏明顯 → 局部強力撫平，往上到下巴漸弱、往下接胸口
    if cfg.get("neck_smooth"):
        xn, yn, zn = P[:, 0], P[:, 1], P[:, 2]
        wn = (1 - sm(1.49, 1.52, yn)) * sm(1.40, 1.43, yn) * (1 - sm(0.05, 0.075, np.abs(xn))) * (zn > -0.02)
        P = _taubin(P, e, wn.astype(np.float64), cfg["neck_smooth"], lam=0.7, mu=0.0)
        print(f"neck smooth: {int((wn > 0.5).sum())} vertices")
    # 3. 整體皺褶撫平（臉與脖子不動）
    face = (np.abs(P[:, 0]) < 0.09) & (P[:, 1] > 1.35) & (P[:, 1] < 1.7) & (P[:, 2] > 0.0)
    P = _taubin(P, e, np.where(face, 0.0, 1.0), cfg["smooth_iters"])
    mesh.vertices = P[inv]
    if cfg.get("bare_waist", True) and cfg.get("junction_cull"):
        # 交界清理：壓平／投影後留下的「拉長三角形」（平台拉到腰上的帳篷狀條紋、細縫）
        # 與腰上朝內的三角形（和外層皮膚重疊，背面打光不同 → 白斑）直接刪掉
        a, b2 = cfg["waist"]
        T = P[F]
        ty = T[..., 1]
        zone = (ty.min(1) > cfg["rim_level"] - 0.012) & (ty.max(1) < b2 + 0.02)
        el = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
        long_ = zone & (el > cfg["junction_cull"])
        if cfg.get("back_slope"):
            # 後半圈的斜面本來就被拉長（腰接臀部），不刪，否則會破洞
            thc = np.abs(np.arctan2(T.mean(1)[:, 0] - cx, T.mean(1)[:, 2] - cz))
            long_ &= thc < np.radians(cfg["back_sector"][0] - 10)
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        cen = T.mean(1)
        rad = np.stack([(cen[:, 0] - cx) / rx ** 2, np.zeros(len(cen)), (cen[:, 2] - cz) / rz ** 2], 1)
        band = (ty.min(1) > a + 0.003) & (ty.max(1) < b2 - 0.006)
        inward = band & ((fn * rad).sum(1) < 0)
        # 朝內的翻面（不刪，刪了會破洞）；拉長的刪掉
        fc = mesh.faces.copy()
        fc[inward] = fc[inward][:, ::-1]
        mesh.faces = fc
        mesh.update_faces(~long_)
        print(f"junction cull: long {int(long_.sum())} removed, inward {int(inward.sum())} flipped")
    print("reshape:", "skirt straightened" if cfg.get("straighten_skirt", True) else "skirt original", "|", "bare waist" if cfg.get("bare_waist", True) else "waist original", "| any NaN:", bool(np.isnan(P).any()))


def smooth_face(mesh, iters=12, lam=0.5, mu=-0.53):
    """V2：臉部網格平滑（Taubin，不縮小），只作用在臉與脖子，邊緣漸弱。
    貼圖接縫上的重複頂點依位置合併後一起移動，避免裂開。"""
    v = mesh.vertices.copy()
    key = np.round(v / 1e-5).astype(np.int64)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    P = v[uid].copy()
    e = inv[mesh.edges_unique]
    e = e[e[:, 0] != e[:, 1]]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    w = (np.clip((0.085 - np.abs(x)) / 0.025, 0, 1) * np.clip((y - 1.35) / 0.05, 0, 1)
         * np.clip((1.69 - y) / 0.03, 0, 1) * np.clip((z - 0.0) / 0.03, 0, 1))
    deg = np.bincount(e.ravel(), minlength=len(P)).astype(np.float64)
    for _ in range(iters):
        for f in (lam, mu):
            acc = np.zeros_like(P)
            np.add.at(acc, e[:, 0], P[e[:, 1]])
            np.add.at(acc, e[:, 1], P[e[:, 0]])
            avg = acc / np.maximum(deg, 1)[:, None]
            P = P + (f * w)[:, None] * (avg - P)
    mesh.vertices = P[inv]
    print("face smoothed vertices", int((w > 0).sum()))


def process_texture_tripo(img, xyz):
    from PIL import ImageFilter

    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    sc = a.shape[0] / 1024.0          # 貼圖放大後，以像素為單位的半徑要跟著放大（實際尺寸不變）
    si = max(1, int(round(sc)))
    h, s, v = hsv(a)
    unused = np.isnan(xyz[..., 1])
    X, Y, Z = (np.nan_to_num(xyz[..., i], nan=0.0) for i in range(3))
    Y = np.where(unused, 0.75 * HEIGHT, Y)
    rel = Y / HEIGHT
    L = a.mean(2)
    Limg = Image.fromarray((L * 255).astype(np.uint8))
    low = np.asarray(Limg.filter(ImageFilter.GaussianBlur(3 * sc))).astype(np.float32) / 255
    detail = L - low  # 細皺褶、針織紋
    skirt = rel < SKIRT_Y
    skirt_look = Y < RESHAPE["waist"][0] + 0.01  # 裙子上段（臀部骨架那圈）和下面是同一塊針織布 → 同樣的底色、紋理強度
    in_face = (np.abs(X) < FACE_BOX["x"]) & (Y > FACE_BOX["y"][0]) & (Y < FACE_BOX["y"][1]) \
        & (Z > FACE_BOX["z"][0]) & (Z < FACE_BOX["z"][1])
    skinish = (h > 3) & (h < 45) & (s > 0.15) & (v > 0.12)
    lips = ((h < 15) | (h > 340)) & (s > 0.3)
    # 臉框內：除了明顯的白布（低飽和且亮），其餘（膚色、嘴唇、頭髮、陰影）全部保留原貌
    white_cloth = (s < 0.14) & (v > 0.42)
    keep = in_face & (skinish | lips | ~white_cloth) & ~unused
    # 胸口以下（< 1.5 m）不是臉：只保留膚色，前襟內側的灰色布料陰影不保留（會留下灰色污點），清成乾淨白布
    keep &= (Y > 1.5) | skinish
    # 1.47 以下（胸口開口）不保留原貼圖：開口的底另外塗成均勻膚色，開口兩側的殼內壁是白布（不能留米色／膚色）
    keep &= Y > 1.47
    # 脖子兩側（下巴以下）是帽兜內襯往下延續 → 不保留原本膚色，清成白布
    keep &= ~((Y < 1.51) & (np.abs(X) > RESHAPE.get("neck_hw", 0.2)))
    # 臉的中央（眼睛、眼皮、鼻子、嘴）不論顏色全部保留：眼白、眼皮反光不能被當成白布塗掉
    core = (np.abs(X) < 0.06) & (Y > 1.47) & (Y < 1.67) & (Z > 0.03) & ~unused
    keep |= core
    # 帽兜的布（帽沿內側、臉前的翻片）也落在框內、顏色偏暖 → 用深度分開：
    # 以確定是皮膚的像素擬合臉的曲面 z = f(x, y)，離臉面超過 2.5 cm 的不算臉
    sure = core & skinish & (s > 0.22)
    zf = None
    if sure.sum() > 200:
        xs_, ys_, zs_ = X[sure], Y[sure] - 1.56, Z[sure]
        A = np.stack([np.ones_like(xs_), xs_, ys_, xs_ ** 2, ys_ ** 2, xs_ * ys_], 1)
        coef, *_ = np.linalg.lstsq(A, zs_, rcond=None)
        Yc = Y - 1.56
        zf = coef[0] + coef[1] * X + coef[2] * Yc + coef[3] * X ** 2 + coef[4] * Yc ** 2 + coef[5] * X * Yc
        off_face = (np.abs(Z - zf) > 0.025) & (Y > 1.47)  # 頭與脖子：離臉面太遠的是帽兜／殼內壁
        keep &= ~off_face
        # 2026-11-20 使用者：臉旁邊的布映到臉的顏色 → 臉中央以外（臉頰外緣、額頭邊）條件收緊：
        # 要夠飽和的膚色（被臉色染到的布顏色淡、飽和度低）且貼著臉面 1.4 cm 內，其餘一律是白布
        rim_zone = keep & ~core
        true_skin = (skinish & (s > 0.2)) | lips
        keep &= ~(rim_zone & (~true_skin | (np.abs(Z - zf) > 0.014)))
        print("face surface fit", np.round(coef, 3), "off-face texels removed", int((in_face & off_face).sum()))
    keep = cv2_dilate(keep, 1) & in_face
    # 保留區邊緣柔化
    import cv2 as _ck
    _u = (~unused).astype(np.float32)
    kf = _ck.GaussianBlur(keep.astype(np.float32) * _u, (0, 0), 1.5 * sc) / np.maximum(_ck.GaussianBlur(_u, (0, 0), 1.5 * sc), 1e-4)
    kf = np.where(unused, _ck.GaussianBlur(kf * _u, (0, 0), 3 * sc) / np.maximum(_ck.GaussianBlur(_u, (0, 0), 3 * sc), 1e-4), kf).astype(np.float32)
    base = np.where(skirt_look, 0.86, 0.82)
    gain = np.where(skirt, 1.5, 1.1)  # 布料細節恢復原本
    # 細節的暗部限深：小黑點（原始貼圖的污漬）不保留，正常皺褶的陰影不受影響
    # 2026-11-19：白布上的髒污一律蓋白 → 暗部只留很淺的紋理（-0.035），亮部針織紋保留
    # 2026-11-19 使用者：材質感是評分重點 → 布紋強度回到接近原本；髒污＝成塊的暗斑 → 只扣掉低頻的暗部，細的紋路保留
    import cv2 as _cv2
    blob = _cv2.GaussianBlur(np.minimum(detail, 0).astype(np.float32), (0, 0), 4 * sc)
    detail_c = detail - blob * 1.2  # 布用（臉的明暗仍用原本的 detail）
    gain = np.where(skirt_look, 1.3, 1.0)
    dclip = np.where(skirt_look, np.clip(detail_c, -0.08, 0.14), np.clip(detail_c, -0.06, 0.1))
    newL = np.clip(base + dclip * gain, 0, 1)
    tint = np.where(skirt[..., None], np.array([1.0, 0.985, 0.95]), np.array([0.985, 0.99, 1.0]))
    tint = np.where(skirt_look[..., None], np.array([1.0, 0.985, 0.95]), np.array([0.985, 0.99, 1.0]))
    cleaned = newL[..., None] * tint
    # 臉中央被 AI 烤上帽兜白布色的地方（眼皮上的白月牙）→ 換成她的膚色，保留明暗細節
    skin_px = core & skinish & (v > 0.25)
    if skin_px.any():
        skin_col = np.median(a[skin_px], axis=0)
        wc = core & white_cloth
        wc = np.asarray(Image.fromarray((cv2_dilate(wc, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255
        shade = np.clip(0.85 + detail * 2.0, 0.6, 1.1)[..., None]
        a = a * (1 - wc[..., None]) + skin_col * shade * wc[..., None]
    # 磨皮：保留輪廓的平滑（雙邊濾波），眼、眉、唇的邊緣保留，膚色的顆粒與斑駁抹平（貼近影片裡的臉）
    import cv2
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    sm = cv2.bilateralFilter(a8, 9, 40, 7).astype(np.float32) / 255
    sm = cv2.bilateralFilter((sm * 255).astype(np.uint8), 9, 32, 6).astype(np.float32) / 255
    sm = cv2.bilateralFilter((sm * 255).astype(np.uint8), 7, 24, 4).astype(np.float32) / 255
    skin_soft = (keep & skinish & ~lips).astype(np.float32)
    skin_soft = cv2.GaussianBlur(skin_soft, (0, 0), 1.2)[..., None] * 0.95
    a = a * (1 - skin_soft) + sm * skin_soft
    # 左右半臉來自不同貼圖區塊、亮度不一：以中線兩側 2 cm 的膚色比例，校正較暗那一半（越靠中線校正越完整）
    # 接縫不一定剛好在 x = 0：在 ±1.2 cm 內找左右色差最大的位置當中線
    fsk = keep & skinish & (Y > 1.47) & (Y < 1.62)
    best, x0s = -1.0, 0.0
    for xc_ in np.linspace(-0.012, 0.012, 25):
        l_ = fsk & (X > xc_ - 0.004) & (X < xc_ - 0.0005)
        r_ = fsk & (X > xc_ + 0.0005) & (X < xc_ + 0.004)
        if l_.sum() > 40 and r_.sum() > 40:
            d_ = float(np.abs(a[l_].mean(0) - a[r_].mean(0)).sum())
            if d_ > best:
                best, x0s = d_, float(xc_)
    global FACE_X0
    FACE_X0 = x0s
    Xs = X - x0s
    print(f"face seam at x = {x0s * 1000:.1f} mm (step {best:.3f})")
    strip = keep & skinish & (np.abs(Xs) < 0.02) & (Y > 1.45) & (Y < 1.62)
    left, right = strip & (Xs < -0.003), strip & (Xs > 0.003)
    if left.sum() > 50 and right.sum() > 50:
        ml, mr = a[left].mean(0), a[right].mean(0)
        side = np.where(Xs < 0, 1.0, 0.0)  # 左半臉
        ratio_l = np.clip(mr / np.maximum(ml, 1e-3), 0.8, 1.25)  # 把左半調成右半的色
        # 每 1 cm 高度各自量一次比例（下巴、嘴唇、鼻子的落差不同），沿高度平滑；量不到的高度用整體比例
        yb2 = np.arange(1.45, 1.63, 0.01)
        rat = np.tile(ratio_l, (len(yb2), 1)).astype(np.float64)
        for i_, t_ in enumerate(yb2):
            l_, r_ = left & (np.abs(Y - t_) < 0.008), right & (np.abs(Y - t_) < 0.008)
            if l_.sum() > 25 and r_.sum() > 25:
                rat[i_] = np.clip(a[r_].mean(0) / np.maximum(a[l_].mean(0), 1e-3), 0.8, 1.25)
        k3 = np.ones(3) / 3
        rat = np.stack([np.convolve(np.pad(rat[:, c], 1, mode="edge"), k3, mode="valid") for c in range(3)], 1)
        ratio_map = np.stack([np.interp(Y, yb2, rat[:, c]) for c in range(3)], -1)
        wgt = (keep & (np.abs(Xs) < 0.09)).astype(np.float32) * side
        a = a * (1 + (ratio_l - 1) * wgt[..., None])
        print("face side balance (L→R ratio)", np.round(ratio_l, 3))
    face = face_lift(a)  # 臉提亮（暗部拉起來、亮部不爆），保留原本五官與膚色
    w = kf[..., None]
    out = cleaned * (1 - w) + face * w
    skin_w = np.zeros(out.shape[:2], np.float32)
    skin_paint = None
    # 作者回饋：裡面沒有穿衣服（只有外殼）、上下兩段之間微露腰 → 襯衫與腰部改成她的膚色（保留明暗）
    skin_src = keep & skinish & (v > 0.25) & (Y > 1.47)
    if skin_src.any():
        skin_rgb = np.clip(face_lift(np.median(a[skin_src], axis=0)[None, None, :])[0, 0] * 1.18, 0, 1)  # 身上在殼的陰影裡，提亮才和臉接近
        shade = np.clip(1.0 + (L - low) * 0.6, 0.92, 1.05)[..., None]  # 只留一點明暗，原本布料的皺褶紋路淡掉
        skin_tex = np.clip(skin_rgb * shade, 0, 1)
        # 襯衫：殼的開口裡、胸口一帶偏暖色（米色）的布
        # 襯衫：只用逐頂點判斷出的襯衫網格（SHIRT_TEX），外殼上偏暖的污點不會被誤染成膚色
        # 胸口：只有幾何判斷出的「開口的底」（CHEST_TEX）是皮膚；殼內壁不論原本顏色都是白布
        if CHEST_TEX is not None:
            # 再用 3D 位置確認：遮罩畫三角形時連邊線，會蓋到貼圖上緊鄰的別的區塊（裙子出現棕色細線）
            shirt = (CHEST_TEX > 0.5) & ~unused & (Y > RESHAPE["chest_y"][0] - 0.01) & (Y < RESHAPE["chest_y"][1] + 0.01) & (np.abs(X) < 0.15) & (Z < RESHAPE["chest_zmax"] + 0.01 + 0.08 * (Y > 1.47))
        else:
            shirt = (SHIRT_TEX > 0.5) & (np.abs(X) < 0.14) & (h > 8) & (h < 55) & (s > 0.08) & (v > 0.2) & ~unused
        # 2026-11-19 使用者：胸口太裸 → 兩側的布往中間多蓋一些（皮膚只留 skin_hw 半寬），交界做滾邊
        piping = None
        if RESHAPE.get("skin_hw"):
            hy_, hw_ = zip(*RESHAPE["skin_hw"])
            lim = np.interp(Y, hy_, hw_)
            pw = RESHAPE.get("piping", 0.007)
            # 每個高度量出「底」的左右範圍 → 中心與可用半寬（滾邊要整條落在底上，邊界才是平順的曲線，不會參差）
            yb_ = np.arange(RESHAPE["chest_y"][0], RESHAPE["chest_y"][1] + 0.02, 0.005)
            lo_a, hi_a = np.full(len(yb_), np.nan), np.full(len(yb_), np.nan)
            Ysh, Xsh = Y[shirt], X[shirt]
            for i_, t_ in enumerate(yb_):
                m_ = np.abs(Ysh - t_) < 0.004
                if m_.sum() > 30:
                    lo_a[i_], hi_a[i_] = np.percentile(Xsh[m_], [10, 90])   # 兩側各往內 10%：避開參差的三角形邊緣
            ok_ = ~np.isnan(lo_a)
            if ok_.sum() > 4:
                k7 = np.ones(7) / 7
                smo_ = lambda A: np.convolve(np.pad(np.interp(yb_, yb_[ok_], A[ok_]), 3, mode="edge"), k7, mode="valid")
                lo_a, hi_a = smo_(lo_a), smo_(hi_a)
                global CHEST_EDGE
                _c = (lo_a + hi_a) / 2
                _l0 = np.interp(yb_, hy_, hw_)
                CHEST_EDGE = (yb_.copy(), np.maximum(lo_a + pw, _c - _l0), np.minimum(hi_a - pw, _c + _l0))
                xl_, xr_ = np.interp(Y, yb_, lo_a), np.interp(Y, yb_, hi_a)
                cmid = (xl_ + xr_) / 2
                left_, right_ = np.maximum(xl_ + pw, cmid - lim), np.minimum(xr_ - pw, cmid + lim)
                Xc = X - (left_ + right_) / 2
                lim = np.maximum((right_ - left_) / 2, 0)
                print("chest skin left", np.round(lo_a[::8], 3), "right", np.round(hi_a[::8], 3))
            else:
                Xc = X
            if CHEST_POLY is not None:
                # 貼著胸口曲面（前 1.2 cm、後 3 cm 內）的像素一律歸在這一區：壓在皮膚邊緣上的內襯碎片不再是一根根白色的毛邊
                yr_ = (Y > RESHAPE["chest_y"][0] - 0.01) & (Y < RESHAPE["chest_y"][1] + 0.01) & ~unused & (np.abs(Xc) < lim + pw + 0.002)
                iy, ix = np.where(yr_)
                xs2, ys2 = X[iy, ix] / 0.1, (Y[iy, ix] - 1.36) / 0.1
                zfit = np.zeros(len(iy))
                k_ = 0
                for i in range(5):
                    for j in range(5 - i):
                        zfit += CHEST_POLY[k_] * xs2 ** i * ys2 ** j
                        k_ += 1
                dz2 = Z[iy, ix] - zfit
                okz = (dz2 > -0.03) & (dz2 < 0.012)
                near_ = np.zeros_like(shirt)
                near_[iy[okz], ix[okz]] = True
                print("chest surface texels added", int((near_ & ~shirt).sum()))
                shirt = shirt | near_
            shirt = shirt & ~((Y > 1.47) & (kf > 0.9))   # 臉保留區（下巴）不蓋
            piping = shirt & (np.abs(Xc) >= lim) & (np.abs(Xc) < lim + pw) & (lim > 0.004)
            pipe_t = np.clip((np.abs(Xc) - lim) / pw, 0, 1)
            shirt = shirt & (np.abs(Xc) < lim)
        # 脖子（下巴到胸口之間、脖子寬度內）：一律皮膚 → 臉、脖子、胸口連續，不會中間橫一條白布
        if zf is not None:
            nhw = RESHAPE.get("neck_hw", 0.045)
            neck = (np.abs(X) < 0.028) & (Y > 1.47) & (Y < 1.535) & (Z < zf + 0.012) & (Z > zf - 0.10) & ~unused & ~(kf > 0.9)  # 連臉部保留區的柔邊一起蓋（柔邊會混到白布色 → 下巴下方一道白弧）
            print("neck skin texels", int(neck.sum()))
            shirt = shirt | neck
        # 腰部：只有真的貼在腰身橢圓上的像素是皮膚（殼的下緣、裙子平台就算在同高度也是白布）→ 交界分明
        if WAIST_ELL is not None:
            wcx, wcz, rx, rz, wa0, wb0 = WAIST_ELL
            q = np.sqrt(((X - wcx) / rx) ** 2 + ((Z - wcz) / rz) ** 2)
            curve = 1 - 0.07 * np.sin(np.pi * np.clip((Y - wa0) / (wb0 - wa0), 0, 1))
            # 橢圓上與橢圓內（被壓進腰裡的三角形，貼著腰內側）都算皮膚 → 不會出現白斑；下緣留 4 mm 給裙子平台的白邊
            # 上緣 5 mm 內只認貼在橢圓上的（殼下緣被拉進來的布片維持白色）
            on = q < curve + np.where(Y > wb0 - 0.012, 0.025, 0.05)
            ylo = np.full(Y.shape, wa0 + 0.004)
            if RESHAPE.get("back_slope"):
                thx = np.abs(np.arctan2(X - wcx, Z - wcz))
                back = thx > np.radians(sum(RESHAPE["back_sector"]) / 2)
                ylo = np.where(back, RESHAPE["back_slope"][1] - 0.004, ylo)
            whole = np.nan_to_num(YMIN_TEX, nan=9.0) > ylo - 0.002 if YMIN_TEX is not None else True  # 三角形最低點也在腰上
            waist_w = ((Y > ylo) & (Y < wb0 - 0.005) & on & whole & ~unused).astype(np.float32)
        else:
            wa, wb = RESHAPE["waist"][0] + 0.017, RESHAPE["waist"][1]
            waist_w = ((Y > wa) & (Y < wb) & ~unused).astype(np.float32)
        if not RESHAPE.get("bare_waist", True) or not RESHAPE.get("paint_old_waist", False):
            waist_w = waist_w * 0.0
        # 腰部範圍往外多補 3 像素（蓋住貼圖接縫邊緣混到的白布），腰上不留原本布料的明暗紋路
        # 只往貼圖區塊之間的空隙補，不蓋到相鄰的布料（避免殼與裙子邊緣沾到膚色）
        # （不再往空隙擴張：空隙由 pad_islands 依最近的區塊補色；擴張會把膚色補到白布區塊旁 → 縮小取樣時滲出棕線）
        shade = np.where((waist_w > 0)[..., None], 0.98, shade)  # 腰：均勻膚色（重疊的網格層顏色一致，不會閃爍成斑）
        # 胸口不能是一片死平的膚色：加上身體的明暗（越靠兩側、越往下越在殼的陰影裡；鎖骨下方最亮）
        if RESHAPE.get("skin_hw"):
            edge = np.clip(np.abs(Xc) / np.maximum(lim, 0.02), 0, 1.0)
            ao = (1 - 0.26 * edge ** 2.2) * (0.86 + 0.14 * np.clip((Y - 1.24) / 0.2, 0, 1))
            ao *= 1 - 0.05 * np.exp(-(Xc / 0.006) ** 2) * np.clip((1.40 - Y) / 0.06, 0, 1)  # 胸骨中線的淡淡凹影
            shade = np.where((shirt & (waist_w <= 0))[..., None], ao[..., None], shade)  # 原本襯衫的皺褶明暗完全不留
        global SKIN_RGB
        SKIN_RGB = np.array(skin_rgb, dtype=np.float64)
        skin_tex = np.clip(skin_rgb * shade, 0, 1)
        sk = np.maximum(shirt.astype(np.float32), waist_w.astype(np.float32))[..., None]  # 邊界清楚：皮膚和布料不柔化
        waist = waist_w > 0.5
        out = out * (1 - sk) + skin_tex * sk
        if piping is not None and piping.any():
            # 滾邊：一條圓鼓的包邊（中間亮、兩側暗），貼皮膚那側有一道細陰影
            # 一般衣服的收邊：布邊往內折一道，平的；貼皮膚那側有折邊厚度的細陰影，另一側一道淡淡的車縫線
            prof = 0.95 - 0.07 * (pipe_t < 0.16) - 0.035 * ((pipe_t > 0.72) & (pipe_t < 0.86))
            out = np.where(piping[..., None], prof[..., None] * np.array([0.985, 0.985, 0.97]), out)
            print("piping texels", int(piping.sum()))
        skin_w = sk[..., 0]
        skin_paint = (skin_tex, sk[..., 0] > 0.5)
        print("skin recolor: shirt texels", int(shirt.sum()), "waist texels", int(waist.sum()))
    mask = np.where(~skirt_look & (kf < 0.5), 1.0, 0.0).astype(np.float32)
    mask = mask * (1 - skin_w)  # 皮膚（襯衫改的胸口、露腰）不是蛹的布：不發光、不套布料壓暗
    # 補邊：貼圖區塊之間的空隙用相鄰區塊的顏色往外擴，避免接縫滲出白線（臉上尤其明顯）
    used = (~unused).astype(np.float32)
    out = pad_islands(out, used, iters=16 * si)
    mask = pad_islands(mask[..., None], used, iters=16 * si)[..., 0]
    # 臉的區塊緊貼著帽兜白布的區塊：取樣時會混到白色 → 眼皮、鼻孔出現白線。
    # 把臉區塊外圍 3 像素（屬於鄰近白布區塊的邊）改成臉的顏色往外延伸
    facem = kf > 0.5
    band = cv2_dilate(facem, 6 * si) & ~facem & unused  # 只補貼圖空隙；不蓋到相鄰的布（布會被染成臉的顏色）
    import cv2 as _cv
    d_face = _cv.distanceTransform((~facem).astype(np.uint8), _cv.DIST_L2, 3)
    d_oth = _cv.distanceTransform((~(~unused & ~facem & ~in_face)).astype(np.uint8), _cv.DIST_L2, 3)
    band &= d_face < d_oth - 1.5  # 空隙裡離白布區塊比較近（或差不多近）的像素不塗臉色  # 只補空隙與臉框內（不蓋到相鄰白布區塊 → 殼上棕線）
    grown = pad_islands(out, facem.astype(np.float32), iters=7 * si)
    out = np.where(band[..., None], grown, out)
    mask = np.where(cv2_dilate(facem, 3 * si), 0.0, mask)
    # 皮膚、臉的周圍一圈，以及整個下巴／脖子一帶不發光：
    # 臉部保留區的柔邊、皮膚邊界上的布原本會發光 → 下巴下方一圈白、皮膚邊緣像一根根白毛
    import cv2 as _cg
    noglow = _cg.dilate(((skin_w > 0.5) | facem).astype(np.uint8), np.ones((10 * si + 1, 10 * si + 1), np.uint8)) > 0
    if zf is not None:
        noglow |= (np.abs(X) < 0.07) & (Y > 1.43) & (Y < 1.60) & (Z > zf - 0.12) & (Z < zf + 0.05)
    mask = np.where(noglow, 0.0, mask)  # 臉與周圍不發光、不套布料效果
    out = blend_face_seams(out, xyz, cv2_dilate(facem, 6 * si))
    fm_ = facem & ~unused & (Y > 1.46) & (Y < 1.69)
    Lo_ = out.mean(2)
    ybm = np.arange(1.47, 1.685, 0.004)
    pk = np.full(len(ybm), np.nan)
    for i_, t_ in enumerate(ybm):
        m_ = fm_ & (np.abs(Y - t_) < 0.004) & (X > -0.006) & (X < 0.014)
        if m_.sum() < 150:
            continue
        bi_ = np.clip(((X[m_] + 0.006) / 0.0005).astype(int), 0, 39)
        n_ = np.bincount(bi_, minlength=40)
        prof_ = np.bincount(bi_, Lo_[m_], 40) / np.maximum(n_, 1)
        okp = n_ >= 3
        if okp.sum() < 30:
            continue
        xi_ = np.arange(40)
        prof_ = np.interp(xi_, xi_[okp], prof_[okp])
        d_ = prof_ - np.interp(xi_, [0, 39], [prof_[:5].mean(), prof_[-5:].mean()])
        if d_[6:34].max() > 0.035:
            pk[i_] = -0.006 + (6 + np.argmax(d_[6:34]) + 0.5) * 0.0005
    if RESHAPE.get("midline_fill", False) and (~np.isnan(pk)).sum() >= 6:
        xm0 = float(np.nanmedian(pk))
        pk = np.where(np.isnan(pk) | (np.abs(pk - xm0) > 0.003), xm0, pk)
        pk = np.convolve(np.pad(pk, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        Xm = X - np.interp(Y, ybm, pk)
        hw_m = 0.0036
        yb3 = np.arange(1.462, 1.69, 0.002)
        Lc, Rc = np.full((len(yb3), 3), np.nan), np.full((len(yb3), 3), np.nan)
        for i_, t_ in enumerate(yb3):
            hrow = fm_ & (np.abs(Y - t_) < 0.0015)
            l_, r_ = hrow & (Xm < -hw_m) & (Xm > -hw_m - 0.003), hrow & (Xm > hw_m) & (Xm < hw_m + 0.003)
            if l_.sum() > 8 and r_.sum() > 8:
                Lc[i_], Rc[i_] = out[l_].mean(0), out[r_].mean(0)
        okm = ~np.isnan(Lc[:, 0])
        if okm.sum() > 10:
            Lm = np.stack([np.interp(Y, yb3[okm], Lc[okm, c]) for c in range(3)], -1)
            Rm = np.stack([np.interp(Y, yb3[okm], Rc[okm, c]) for c in range(3)], -1)
            t_m = np.clip((Xm + hw_m) / (2 * hw_m), 0, 1)[..., None]
            wm = (np.clip((hw_m + 0.0012 - np.abs(Xm)) / 0.0012, 0, 1) * (cv2_dilate(fm_, 2 * si) & (Y > 1.462) & (Y < 1.69)))[..., None]
            out = out * (1 - wm) + (Lm * (1 - t_m) + Rm * t_m) * wm
            print(f"face midline filled at x = {xm0 * 1000:.1f} mm, rows {int(okm.sum())}")
    # 皮膚區（胸口、腰）在補邊之後再上一次色，並往外多 4 像素：貼圖接縫空隙不會滲進白布色（白線、白斑）
    if skin_paint is not None:
        stex, smask = skin_paint
        reg = smask  # 只重塗皮膚本身；空隙已由 pad_islands 依最近區塊補色（往空隙擴張會讓相鄰白布滲出棕線）
        out = np.where(reg[..., None], stex, out)
        mask = np.where(reg, 0.0, mask)
    global FACE_TEX, COLORED_TEX, FACEM_TEX
    FACEM_TEX = facem.copy()
    FACE_TEX = cv2_dilate(facem, 6 * si)
    COLORED_TEX = (skin_w > 0.5) | facem
    color = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    emissive = Image.fromarray((np.repeat(mask[..., None], 3, 2) * 255).astype(np.uint8))
    print("face texels kept", int(keep.sum()))
    return color, emissive, mask


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pink = "--pink" in sys.argv
    tripo = "--tripo" in sys.argv
    scene = trimesh.load(src)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene

    # 1. 尺寸校正
    lo, hi = mesh.bounds
    s = HEIGHT / (hi[1] - lo[1])
    mesh.apply_scale(s)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])

    # 1b. 比例雕塑：依照片比對調整（數值在 SCULPT）
    if not tripo:
        sculpt(mesh)
    else:
        global SHIRT_V, SHIRT_TEX
        SHIRT_V = shirt_vertices(mesh)
        reshape_v2(mesh)
        smooth_face(mesh)

    # 2~4. 貼圖與材質
    mat = mesh.visual.material
    base_img = mat.baseColorTexture
    if tripo and base_img.size[0] < 2048:
        # 原始貼圖只有 1024：皮膚與布的分界、收邊線只有 2 mm 的解析度，邊界會像鋸齒／毛邊 → 先放大再處理
        xyz_lo = texel_xyz(mesh, base_img.size[0])
        used_lo = (~np.isnan(xyz_lo[..., 1])).astype(np.float32)
        arr_lo = np.asarray(base_img.convert("RGB")).astype(np.float32) / 255
        arr_lo = pad_islands(arr_lo, used_lo, iters=10)
        del xyz_lo
        base_img = Image.fromarray((np.clip(arr_lo, 0, 1) * 255).astype(np.uint8)).resize((2048, 2048), Image.BICUBIC)
    size = base_img.size[0]
    normal_tex = getattr(mat, "normalTexture", None) if tripo else None
    if tripo:
        global CHEST_TEX
        SHIRT_TEX = texel_flag(mesh, size, SHIRT_V) if SHIRT_V is not None else None
        CHEST_TEX = texel_flag(mesh, size, CHEST_V) if CHEST_V is not None and CHEST_V.any() else None
        global YMIN_TEX
        YMIN_TEX = texel_ymin(mesh, size)
        xyz_tex = texel_xyz(mesh, size)
        color, emissive, mask = process_texture_tripo(base_img, xyz_tex)
        if normal_tex is not None and FACE_TEX is not None:
            # 法線貼圖：布料區改成平的（原圖有一條條刮痕狀的雜線，會在白布上變成暗紋）；臉保留
            nt = np.asarray(normal_tex.convert("RGB")).astype(np.float32)
            fm = np.asarray(Image.fromarray(FACE_TEX.astype(np.uint8) * 255).resize(normal_tex.size)).astype(np.float32)[..., None] / 255
            nt = nt * fm + np.array([128.0, 128.0, 255.0]) * (1 - fm)
            normal_tex = Image.fromarray(nt.astype(np.uint8))
    else:
        height = texel_height(mesh, size)
        unused = np.isnan(height)
        height = np.where(unused, 0.75, height)  # 未使用的像素（貼圖島之間的縫）當成蛹 → 填成同樣的白，避免取樣滲色
        color, emissive, mask = process_texture(mat.baseColorTexture, pink, height)
    color.format = "JPEG"
    emissive = emissive.resize((1024, 1024), Image.BILINEAR)
    emissive.format = "JPEG"
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        name="emerge",
        baseColorTexture=color,
        emissiveTexture=emissive,
        emissiveFactor=[1.0, 1.0, 1.0],
        normalTexture=None if tripo else normal_tex,  # Tripo 的法線貼圖在左右半臉的區塊方向不同 → 臉中央出現一條明暗分界；改用幾何本身的法線
        metallicFactor=0.0,
        roughnessFactor=0.66,
        doubleSided=True,
    )

    if tripo:
        if COLORED_TEX is not None:
            protect_white_uv(mesh, COLORED_TEX, ~np.isnan(xyz_tex[..., 1]))
            front_face_skin(mesh, COLORED_TEX)
            if FACEM_TEX is not None:
                remove_chin_collar(mesh, FACEM_TEX)
                clear_chest_front(mesh, COLORED_TEX)
                mesh = bridge_jaw(mesh, COLORED_TEX)
        move_upper(mesh)
        shape_tail(mesh)
        smooth_skirt_top(mesh)
        mesh = cap_shelf(mesh, color, xyz_tex, mask, emissive)
        mesh = fix_stray_uv(mesh, color)

    # 5. LED 光點：在發光布面（遮罩亮、在高度範圍內、朝外）取樣
    rng = np.random.default_rng(9)
    pts, idx = trimesh.sample.sample_surface(mesh, 30000, seed=9)
    uv = mesh.visual.uv
    tri = mesh.faces[idx]
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[idx], pts)
    puv = np.einsum("ij,ijk->ik", bary, uv[tri])
    H, W = mask.shape
    px = np.clip((puv[:, 0] % 1) * (W - 1), 0, W - 1).astype(int)
    py = np.clip((1 - (puv[:, 1] % 1)) * (H - 1), 0, H - 1).astype(int)
    m = mask[py, px]
    y = pts[:, 1] / HEIGHT
    ok = (m > 0.8) & (y > GLOW_BAND[0] + 0.03) & (y < GLOW_BAND[1] - 0.03)
    cand = np.where(ok)[0]
    pick = rng.choice(cand, size=min(LED_COUNT, len(cand)), replace=False)
    normals = mesh.face_normals[idx[pick]]
    leds = pts[pick] + normals * 0.004
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 頂點法線：自己用 numpy 算（trimesh 在沒有 scipy 時會非常慢）
    fn = np.cross(mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
                  mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]])
    # 貼圖接縫上的重複頂點（同位置、不同 UV）依位置合併後一起算 → 接縫兩邊打光連續，不會出現一條線
    key = np.round(mesh.vertices / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    acc = np.zeros((inv.max() + 1, 3))
    for c in range(3):
        np.add.at(acc, inv[mesh.faces[:, c]], fn)
    Nm = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    if tripo and RESHAPE.get("normal_smooth"):
        fm_ = inv[mesh.faces]
        ee = np.vstack([fm_[:, [0, 1]], fm_[:, [1, 2]], fm_[:, [2, 0]]])
        pos_m = np.zeros((len(Nm), 3))
        pos_m[inv] = mesh.vertices
        facebox = (np.abs(pos_m[:, 0]) < 0.09) & (pos_m[:, 1] > 1.45) & (pos_m[:, 1] < 1.72) & (pos_m[:, 2] > 0.0)
        N0 = Nm.copy()
        for _ in range(RESHAPE["normal_smooth"]):
            S_ = Nm.copy()
            for c in range(3):
                S_[:, c] += np.bincount(ee[:, 0], weights=Nm[ee[:, 1], c], minlength=len(Nm)) * 0.5
                S_[:, c] += np.bincount(ee[:, 1], weights=Nm[ee[:, 0], c], minlength=len(Nm)) * 0.5
            Nm = S_ / np.maximum(np.linalg.norm(S_, axis=1, keepdims=True), 1e-12)
        Nm[facebox] = N0[facebox]
    vn = Nm[inv]
    mesh.vertex_normals = vn
    mesh.export(dst, include_normals=True)
    leds_path = dst.with_suffix(".leds.json")
    leds_path.write_text(json.dumps(np.round(leds, 4).tolist()))
    print(f"saved {dst} ({dst.stat().st_size/1e6:.1f} MB), {len(leds)} LEDs, vertices {len(mesh.vertices)}")


if __name__ == "__main__":
    main()
