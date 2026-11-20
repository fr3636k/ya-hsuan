// 最後一章的「破蛹而出」：AI 影片拆成的畫格序列（media/grip/），依進度 0~1 畫到畫布上，
// 再疊上「從光芒中出來」的光效（放射光芒、輪廓光暈、光粒子；破殼瞬間最亮）。
// 影片由 3D 畫面（鏡頭同 story.js 的 finale）當第一格生成：
// 起始畫面寬高比 srcAspect，影片是把它縮放到等寬後裁掉上下 → 依視窗高度對齊即可與 3D 重疊。
// FX：光芒、光暈、光粒子特效。使用者要的是影片維持純黑底 → 預設關閉（保留程式，需要時改成 true）
const FX = false;

export async function loadGripSeq(canvas, base = "./media/grip/") {
  const meta = await fetch(base + "meta.json").then((r) => r.json());
  const frames = new Array(meta.count);
  let loaded = 0;
  const ready = new Promise((resolve) => {
    for (let i = 0; i < meta.count; i++) {
      const im = new Image();
      im.decoding = "async";
      im.onload = im.onerror = () => {
        frames[i] = im.naturalWidth ? im : null;
        if (++loaded === meta.count) resolve();
      };
      im.src = `${base}${String(i).padStart(3, "0")}.jpg`;
    }
  });
  const ctx = canvas.getContext("2d");
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(innerWidth * dpr);
    canvas.height = Math.round(innerHeight * dpr);
  }
  addEventListener("resize", resize);
  resize();

  /* ---------- 光效用的小畫布 ---------- */
  const mk = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });
  const SW = 72,
    SH = Math.round((SW * meta.h) / meta.w);
  const small = mk(SW, SH), // 目前畫格縮小：放大疊回去就是柔和的光暈（不需 filter，手機也快）
    tiny = mk(24, Math.round((24 * meta.h) / meta.w)), // 更大範圍的光暈
    fx = mk(220, 220); // 放射光芒（正方形、比影片寬，桌機上光芒才不會被影片左右邊界切掉；只畫在人物以外的背景）
  const sctx = small.getContext("2d"),
    tctx = tiny.getContext("2d"),
    fctx = fx.getContext("2d");
  // 光粒子：固定亂數，從胸口往外飄散
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const sparks = Array.from({ length: 70 }, () => ({ a: rnd() * Math.PI * 2, sp: 0.05 + rnd() * 0.12, ph: rnd(), s: 0.5 + rnd() }));
  // 破殼瞬間（剪接點的白光）：進度位置
  const flashP = meta.flashAt != null ? meta.flashAt / (meta.count - 1) : 0.62;
  const smooth = (a, b, v) => {
    const k = Math.max(0, Math.min(1, (v - a) / (b - a)));
    return k * k * (3 - 2 * k);
  };

  /** p：0~1 進度；opacity：疊加透明度；time：秒（光芒旋轉、粒子飄散） */
  function draw(p, opacity, time = 0) {
    canvas.style.opacity = String(opacity);
    if (opacity <= 0.001 || loaded < meta.count) return;
    const f = Math.max(0, Math.min(1, p)) * (meta.count - 1);
    const i = Math.floor(f),
      t = f - i;
    const W = canvas.width,
      H = canvas.height;
    // 起始畫面高 = 視窗高；影片寬 = 起始畫面寬
    const dw = H * meta.srcAspect,
      dh = (dw * meta.h) / meta.w;
    const x = (W - dw) / 2,
      y = (H - dh) / 2;
    ctx.clearRect(0, 0, W, H); // 透明底：影片以外（上下邊、左右）透出底下的 3D
    const a = frames[i],
      b = frames[Math.min(meta.count - 1, i + 1)];
    ctx.globalAlpha = 1;
    if (a) ctx.drawImage(a, x, y, dw, dh);
    if (b && t > 0.01) {
      ctx.globalAlpha = t; // 相鄰畫格交叉淡化，慢速播放也順
      ctx.drawImage(b, x, y, dw, dh);
      ctx.globalAlpha = 1;
    }

    // 影片邊緣淡出：上下（影片比起始畫面矮約 2.5%）與左右（桌機上影片黑底與旁邊 3D 背景亮度不同）
    // 先處理影片本身，光效之後再疊，光芒才不會被切掉
    ctx.globalCompositeOperation = "destination-in";
    const fe = dh * 0.05;
    const gv = ctx.createLinearGradient(0, y, 0, y + dh);
    gv.addColorStop(0, "rgba(0,0,0,0)");
    gv.addColorStop(fe / dh, "rgba(0,0,0,1)");
    gv.addColorStop(1 - fe / dh, "rgba(0,0,0,1)");
    gv.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = gv;
    ctx.fillRect(0, 0, W, H);
    if (W > dw * 1.02) {
      const gh = ctx.createLinearGradient(x, 0, x + dw, 0);
      gh.addColorStop(0, "rgba(0,0,0,0)");
      gh.addColorStop(0.22, "rgba(0,0,0,1)");
      gh.addColorStop(0.78, "rgba(0,0,0,1)");
      gh.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = gh;
      ctx.fillRect(0, 0, W, H);
    }
    ctx.globalCompositeOperation = "source-over";

    /* ---------- 從光芒中出來 ---------- */
    // 光的強度：抓殼時慢慢累積，破殼瞬間爆亮，之後維持在她身後
    const build = smooth(0.08, flashP, p);
    const burst = Math.exp(-Math.pow((p - flashP) / 0.06, 2));
    const amt = FX ? Math.min(1.4, build * 0.75 + burst * 0.9) : 0;
    if (amt > 0.01) {
      // 目前畫面縮小（輪廓光暈與人物遮罩都用它）
      sctx.clearRect(0, 0, SW, SH);
      sctx.drawImage(canvas, x, y, dw, dh, 0, 0, SW, SH);
      tctx.clearRect(0, 0, tiny.width, tiny.height);
      tctx.drawImage(small, 0, 0, tiny.width, tiny.height);
      // fx 畫布代表以胸口為中心、邊長 RX 的正方形區域
      const RX = dh * 1.25,
        mcx = x + dw * 0.5,
        mcy = y + dh * 0.36,
        k = fx.width / RX;
      const cx = fx.width / 2,
        cy = fx.height / 2,
        R = fx.width * 0.5;
      // 放射光芒＋中心光：畫在小畫布上，再挖掉人物（亮的地方）→ 光從她身後放射出來
      fctx.globalCompositeOperation = "source-over";
      fctx.clearRect(0, 0, fx.width, fx.height);
      const glow = fctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.8);
      glow.addColorStop(0, "rgba(255,255,255,0.7)");
      glow.addColorStop(0.3, "rgba(235,242,255,0.22)");
      glow.addColorStop(1, "rgba(255,255,255,0)");
      fctx.fillStyle = glow;
      fctx.fillRect(0, 0, fx.width, fx.height);
      const N = 16;
      for (let k = 0; k < N; k++) {
        const ang = (k / N) * Math.PI * 2 + time * 0.06 + Math.sin(k * 3.7) * 0.12;
        const wdt = 0.035 + 0.03 * (0.5 + 0.5 * Math.sin(k * 5.1 + time * 0.8));
        const len = R * (0.68 + 0.26 * Math.sin(k * 2.3 + time * 0.5)); // 不超過正方形範圍，邊緣自然淡出
        const g = fctx.createRadialGradient(cx, cy, 0, cx, cy, len);
        g.addColorStop(0, "rgba(255,255,255,0.75)");
        g.addColorStop(1, "rgba(255,255,255,0)");
        fctx.fillStyle = g;
        fctx.beginPath();
        fctx.moveTo(cx, cy);
        fctx.arc(cx, cy, len, ang - wdt, ang + wdt);
        fctx.closePath();
        fctx.fill();
      }
      fctx.globalCompositeOperation = "destination-out";
      fctx.globalAlpha = 0.85;
      fctx.drawImage(small, cx - (dw / 2) * k, cy - dh * 0.36 * k, dw * k, dh * k);
      fctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = Math.min(1, amt * 0.85);
      ctx.drawImage(fx, mcx - RX / 2, mcy - RX / 2, RX, RX);
      // 輪廓光暈（人物與白繭往外發光）
      // 輪廓光暈：只取大範圍的模糊（tiny），臉才不會被糊掉
      ctx.globalAlpha = Math.min(1, amt * 0.12 + burst * 0.3);
      ctx.drawImage(small, x, y, dw, dh);
      ctx.globalAlpha = Math.min(1, amt * 0.3);
      ctx.drawImage(tiny, x - dw * 0.03, y - dh * 0.03, dw * 1.06, dh * 1.06);
      // 光粒子
      const pcx = x + dw * 0.5,
        pcy = y + dh * 0.36;
      for (const s of sparks) {
        const r = (time * s.sp + s.ph) % 1;
        const d = r * dh * 0.55;
        const px = pcx + Math.cos(s.a) * d,
          py = pcy + Math.sin(s.a) * d * 0.9 - r * dh * 0.05;
        const rad = dw * 0.0032 * s.s * (1 - r * 0.5);
        ctx.globalAlpha = Math.min(1, amt) * Math.sin(Math.PI * r) * 0.9;
        const g = ctx.createRadialGradient(px, py, 0, px, py, rad * 3);
        g.addColorStop(0, "rgba(255,255,255,1)");
        g.addColorStop(0.3, "rgba(255,255,255,0.8)");
        g.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = g;
        ctx.fillRect(px - rad * 3, py - rad * 3, rad * 6, rad * 6);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    }

    ctx.globalCompositeOperation = "source-over";
  }
  return { ready, draw, count: meta.count };
}
