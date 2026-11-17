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


def process_texture_tripo(img, xyz):
    from PIL import ImageFilter

    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    h, s, v = hsv(a)
    unused = np.isnan(xyz[..., 1])
    X, Y, Z = (np.nan_to_num(xyz[..., i], nan=0.0) for i in range(3))
    Y = np.where(unused, 0.75 * HEIGHT, Y)
    rel = Y / HEIGHT
    L = a.mean(2)
    Limg = Image.fromarray((L * 255).astype(np.uint8))
    low = np.asarray(Limg.filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255
    detail = L - low  # 細皺褶、針織紋
    skirt = rel < SKIRT_Y
    in_face = (np.abs(X) < FACE_BOX["x"]) & (Y > FACE_BOX["y"][0]) & (Y < FACE_BOX["y"][1]) \
        & (Z > FACE_BOX["z"][0]) & (Z < FACE_BOX["z"][1])
    skinish = (h > 3) & (h < 45) & (s > 0.15) & (v > 0.12)
    lips = ((h < 15) | (h > 340)) & (s > 0.3)
    # 臉框內：除了明顯的白布（低飽和且亮），其餘（膚色、嘴唇、頭髮、陰影）全部保留原貌
    white_cloth = (s < 0.14) & (v > 0.42)
    keep = in_face & (skinish | lips | ~white_cloth) & ~unused
    # 臉的中央（眼睛、眼皮、鼻子、嘴）不論顏色全部保留：眼白、眼皮反光不能被當成白布塗掉
    core = (np.abs(X) < 0.06) & (Y > 1.47) & (Y < 1.67) & (Z > 0.03) & ~unused
    keep |= core
    # 帽兜的布（帽沿內側、臉前的翻片）也落在框內、顏色偏暖 → 用深度分開：
    # 以確定是皮膚的像素擬合臉的曲面 z = f(x, y)，離臉面超過 2.5 cm 的不算臉
    sure = core & skinish & (s > 0.22)
    if sure.sum() > 200:
        xs_, ys_, zs_ = X[sure], Y[sure] - 1.56, Z[sure]
        A = np.stack([np.ones_like(xs_), xs_, ys_, xs_ ** 2, ys_ ** 2, xs_ * ys_], 1)
        coef, *_ = np.linalg.lstsq(A, zs_, rcond=None)
        Yc = Y - 1.56
        zf = coef[0] + coef[1] * X + coef[2] * Yc + coef[3] * X ** 2 + coef[4] * Yc ** 2 + coef[5] * X * Yc
        off_face = (np.abs(Z - zf) > 0.025) & (Y > 1.5)  # 只判斷頭部；脖子、胸口照原本規則
        keep &= ~off_face
        print("face surface fit", np.round(coef, 3), "off-face texels removed", int((in_face & off_face).sum()))
    keep = cv2_dilate(keep, 1) & in_face
    # 保留區邊緣柔化
    kf = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32) / 255
    base = np.where(skirt, 0.86, 0.82)
    gain = np.where(skirt, 1.5, 1.1)
    # 細節的暗部限深：小黑點（原始貼圖的污漬）不保留，正常皺褶的陰影不受影響
    dclip = np.where(skirt, np.clip(detail, -0.14, 0.14), np.clip(detail, -0.07, 0.1))
    newL = np.clip(base + dclip * gain, 0, 1)
    tint = np.where(skirt[..., None], np.array([1.0, 0.985, 0.95]), np.array([0.985, 0.99, 1.0]))
    cleaned = newL[..., None] * tint
    # 臉中央被 AI 烤上帽兜白布色的地方（眼皮上的白月牙）→ 換成她的膚色，保留明暗細節
    skin_px = core & skinish & (v > 0.25)
    if skin_px.any():
        skin_col = np.median(a[skin_px], axis=0)
        wc = core & white_cloth
        wc = np.asarray(Image.fromarray((cv2_dilate(wc, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255
        shade = np.clip(0.85 + detail * 2.0, 0.6, 1.1)[..., None]
        a = a * (1 - wc[..., None]) + skin_col * shade * wc[..., None]
    face = face_lift(a)  # 臉提亮（暗部拉起來、亮部不爆），保留原本五官與膚色
    w = kf[..., None]
    out = cleaned * (1 - w) + face * w
    mask = np.where(~skirt & (kf < 0.5), 1.0, 0.0).astype(np.float32)
    # 補邊：貼圖區塊之間的空隙用相鄰區塊的顏色往外擴，避免接縫滲出白線（臉上尤其明顯）
    used = (~unused).astype(np.float32)
    out = pad_islands(out, used)
    mask = pad_islands(mask[..., None], used)[..., 0]
    # 臉的區塊緊貼著帽兜白布的區塊：取樣時會混到白色 → 眼皮、鼻孔出現白線。
    # 把臉區塊外圍 3 像素（屬於鄰近白布區塊的邊）改成臉的顏色往外延伸
    facem = kf > 0.5
    band = cv2_dilate(facem, 6) & ~facem
    grown = pad_islands(out, facem.astype(np.float32), iters=7)
    out = np.where(band[..., None], grown, out)
    mask = np.where(cv2_dilate(facem, 3), 0.0, mask)  # 臉與周圍不發光、不套布料效果
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

    # 2~4. 貼圖與材質
    mat = mesh.visual.material
    size = mat.baseColorTexture.size[0]
    if tripo:
        color, emissive, mask = process_texture_tripo(mat.baseColorTexture, texel_xyz(mesh, size))
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
        normalTexture=getattr(mat, "normalTexture", None) if tripo else None,
        metallicFactor=0.0,
        roughnessFactor=0.66,
        doubleSided=True,
    )

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
    vn = np.zeros_like(mesh.vertices)
    for c in range(3):
        np.add.at(vn, mesh.faces[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    mesh.vertex_normals = vn
    mesh.export(dst, include_normals=True)
    leds_path = dst.with_suffix(".leds.json")
    leds_path.write_text(json.dumps(np.round(leds, 4).tolist()))
    print(f"saved {dst} ({dst.stat().st_size/1e6:.1f} MB), {len(leds)} LEDs, vertices {len(mesh.vertices)}")


if __name__ == "__main__":
    main()
