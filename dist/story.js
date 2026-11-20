// 主展示頁：滾動驅動鏡頭、絲線斷裂、模型顯現與細節特寫。
import * as THREE from "three";
import { createStage, fitCamera, loaderUI, rhythmAt } from "./stage.js";
import { loadGarment, HEIGHT } from "./garment-glb.js";
import { MODEL } from "./model-config.js";
import { loadGripSeq } from "./grip-seq.js";

const mobile = matchMedia("(max-width: 720px)").matches;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const canvas = document.getElementById("stage");
const { renderer, scene, camera } = createStage(canvas, { mobile });
const ui = loaderUI();

/* 鏡頭：az 方位角、el 仰角、dist 距離、t 注視點（相對身高的 [x, y, z]） */
// grip（雙手抓住蛹）不跟著滑動：只有滑到最後看全身時，才隨時間慢慢抓住
const KEYS = {
  // 開場不再用絲線包住：光芒綻放、她在光中顯現（顯現由下方的開場時間軸控制）
  hero: { az: 0, el: 4, dist: 4.3, t: [0, 0.52, 0], silk: 1, reveal: 1, burst: 0.4 }, // 開場模特兒放大（原 5.6）
  emerge: { az: 20, el: 3, dist: 5.0, t: [0, 0.5, 0], silk: 1, reveal: 1, burst: 1 },
  face: { az: 12, el: 3, dist: 1.25, t: [0, 0.85, 0.02], silk: 1, reveal: 1, burst: 0.4 }, // Tripo 臉在 y≈1.56 m（0.89）
  fabric: { az: 200, el: 10, dist: 2.8, t: [0, 0.76, 0], silk: 1, reveal: 1, burst: 0.4 },
  tail: { az: 330, el: 12, dist: 2.4, t: [0, 0.13, 0], silk: 1, reveal: 1, burst: 0.4 },
  // 最後定點：與 AI 抓殼影片的第一格同一個構圖（上半身＋整顆蛹），距離依直式／橫式視野換算
  finale: { az: 360, el: 3, dist: 1.7, t: [0, 0.72, 0], silk: 1, reveal: 1, burst: 0.4 },
};
const NUM = ["az", "el", "dist", "silk", "reveal", "burst"];

const sections = [...document.querySelectorAll(".chapter")];
const keys = sections.map((s) => KEYS[s.dataset.key]);
const bar = document.querySelector(".progress i");
const veil = document.querySelector(".veil");
const tagSteps = [...document.querySelectorAll(".lt-step")];
const rhythmSteps = [...document.querySelectorAll(".rhythm li")];
const ease = (t) => t * t * (3 - 2 * t);

function scrollIndex() {
  const mid = innerHeight * 0.5;
  const c = sections.map((s) => {
    const r = s.getBoundingClientRect();
    return r.top + r.height / 2;
  });
  if (mid <= c[0]) return 0;
  for (let i = 0; i < c.length - 1; i++) if (mid >= c[i] && mid < c[i + 1]) return i + (mid - c[i]) / (c[i + 1] - c[i]);
  return c.length - 1;
}
function target(fi) {
  const i = Math.max(0, Math.min(keys.length - 1, Math.floor(fi)));
  const j = Math.min(keys.length - 1, i + 1);
  const t = ease(Math.max(0, Math.min(1, fi - i)));
  const out = { t: [0, 0, 0] };
  for (const f of NUM) out[f] = keys[i][f] + (keys[j][f] - keys[i][f]) * t;
  for (let k = 0; k < 3; k++) out.t[k] = keys[i].t[k] + (keys[j].t[k] - keys[i].t[k]) * t;
  return out;
}

// 同一時間只顯示一章的文字：上一章還沒結束，下一章不出現。
// 目前章節＝頂端已經捲到畫面上方 1/3 以上的最後一章（停留中的那一章一定是目前章節）
let activeSec = null;
function updateActive(holding) {
  let cur = sections[0];
  if (holding) cur = holding;
  else for (const s of sections) if (s.getBoundingClientRect().top <= innerHeight * 0.33) cur = s;
  if (cur === activeSec) return;
  activeSec?.classList.remove("on");
  cur.classList.add("on");
  activeSec = cur;
  dotBtns.forEach((b, i) => {
    b.classList.toggle("on", sections[i] === cur);
    b.setAttribute("aria-current", sections[i] === cur ? "true" : "false");
  });
}
// 章節圓點：顯示目前在第幾段，點一下跳到該段
const dotNames = { hero: "開場", emerge: "破蛹而出", face: "帽沿與臉", fabric: "煙霧與呼吸", tail: "針織魚尾", finale: "破蛹" };
const dotBtns = sections.map((s) => {
  const b = document.createElement("button");
  b.type = "button";
  b.setAttribute("aria-label", dotNames[s.dataset.key] ?? "");
  b.innerHTML = `<span>${dotNames[s.dataset.key] ?? ""}</span>`;
  b.addEventListener("click", () => {
    if (document.documentElement.classList.contains("intro-lock")) return;
    window.scrollTo({ top: Math.round(scrollY + s.getBoundingClientRect().top) + 2, behavior: "instant" });
  });
  document.querySelector(".dots")?.append(b);
  return b;
});
updateActive(null);
// 影片起始畫面：直式視窗（fov 40、距離 ×1.25）下 dist 1.7 → 畫面半高 0.773 m；橫式 fov 30 要同樣半高
const finaleDist = () => (innerWidth / innerHeight < 0.75 ? 1.7 : (1.7 * 1.25 * Math.tan(THREE.MathUtils.degToRad(20))) / Math.tan(THREE.MathUtils.degToRad(15)));
function onResize() {
  fitCamera(renderer, camera);
  KEYS.finale.dist = finaleDist();
}
addEventListener("resize", onResize);
onResize();

// ?at=N：鏡頭固定在第 N 章（截圖檢查用）
const atParam = new URLSearchParams(location.search).get("at");
const forced = atParam !== null && keys[+atParam] ? +atParam : null;
const cur = structuredClone(forced !== null ? keys[forced] : KEYS.hero);

let garment;
try {
  garment = await loadGarment(MODEL.url, MODEL.cfg, (p) => ui.progress(p));
} catch (e) {
  document.documentElement.classList.remove("intro-lock"); // 載入失敗也不要鎖住捲動
  ui.fail("模型載入失敗，請重新整理");
  throw e;
}
scene.add(garment.group);
ui.done();

// 開場「破蛹」時間軸（秒）：全場暗，只有蛹裡的光像呼吸一樣透出（看得到燈籠骨架）
// → 內光越來越亮 → 閃一下 → 外面的燈慢慢亮起，她的臉與全身出現 → 進入 亮/暗/閃 循環
const INTRO = reduceMotion || forced !== null ? 0 : 3; // 開場 3 秒
const introFixed = new URLSearchParams(location.search).get("intro"); // 截圖檢查用：固定開場進度 0~1
// 開場鎖定：滾輪、觸控滑動、方向鍵／空白鍵都先擋住，開場跑完才放開
// 鎖定狀態只看 <html> 的 intro-lock（index.html 的保險移除它時，這裡也跟著放開）
const isLocked = () => document.documentElement.classList.contains("intro-lock");
const blockScroll = (e) => isLocked() && e.cancelable && e.preventDefault();
const blockKeys = (e) => isLocked() && ["ArrowDown", "ArrowUp", "PageDown", "PageUp", " ", "Space", "Home", "End"].includes(e.key) && e.preventDefault();
addEventListener("wheel", blockScroll, { passive: false });
addEventListener("touchmove", blockScroll, { passive: false });
addEventListener("keydown", blockKeys);
/* ---------- 01～04 章節：滑到頂端停留 1 秒 ----------
   停留時仍有回饋，不像當機：滑動會讓內容像有彈性一樣微微拉動（放開彈回）、章節文字下方的細線 1 秒畫滿、
   3D 鏡頭與燈光持續；1 秒後把停留期間累積的滑動量順順接上，繼續往下。手機觸控、滾輪、鍵盤都適用。 */
const holdSections = sections.filter((s) => ["emerge", "face", "fabric", "tail"].includes(s.dataset.key));
const mainEl = document.querySelector("main");
const visited = new Set();
let hold = null; // { section, until, pending }
let lastScrollY = scrollY;
let touchY = null;
// 經過第一個章節後：畫面停止捲動滿 1 秒 → 再次提示「向下滑動」；一滑就收起（最後一章不提示）
let lastActivity = 0,
  holdsDone = 0,
  idleHint = false;
const markActivity = () => {
  lastActivity = performance.now();
  idleHint = false;
};
addEventListener("wheel", markActivity, { passive: true });
addEventListener("touchmove", markActivity, { passive: true });
addEventListener("keydown", markActivity);
const HOLD_MS = 1600; // 停留時間
const DRIFT_PX = 10; // 停留期間內容緩慢上移的距離（微微往上，看起來還在滑）
function pull(px) {
  mainEl.style.transform = px ? `translateY(${-px}px)` : "";
}
function startHold(sec) {
  visited.add(sec);
  const y = Math.round(scrollY + sec.getBoundingClientRect().top);
  window.scrollTo({ top: y, behavior: "instant" });
  hold = { section: sec, start: performance.now(), until: performance.now() + HOLD_MS, pending: 0, y, offset: 0 };
  sec.classList.add("holding");
  mainEl.style.transition = "none"; // 停留期間每一格由 holdFrame() 控制位置
}
// 停留期間每一格：內容緩慢上移（緩入緩出）＋使用者滑動的彈性拉動
function holdFrame(now) {
  const k = Math.min(1, (now - hold.start) / HOLD_MS);
  const drift = DRIFT_PX * (k * k * (3 - 2 * k));
  const elastic = 8 * Math.tanh(hold.pending / 260);
  hold.offset = drift + elastic;
  pull(hold.offset);
}
function endHold() {
  const { section, pending, y, offset } = hold;
  hold = null;
  section.classList.remove("holding");
  // 無縫接回：把畫面上的位移換成真正的捲動位置（看起來不會跳）
  mainEl.style.transition = "none";
  pull(0);
  window.scrollTo({ top: y + Math.round(offset), behavior: "instant" });
  lastScrollY = scrollY;
  holdsDone++;
  lastActivity = performance.now(); // 從停留結束起算 1 秒
  // 停留期間的滑動意圖：順順接上（最多半個畫面），不會被吃掉
  const rest = pending - offset;
  if (rest > 8) window.scrollBy({ top: Math.min(rest, innerHeight * 0.5), behavior: "smooth" });
}
function absorb(dy) {
  hold.pending = Math.max(0, hold.pending + dy);
}
addEventListener(
  "scroll",
  () => {
    const y = scrollY,
      down = y > lastScrollY;
    if (!hold) markActivity(); // 任何捲動（手滑或自動接續）都算還在動
    lastScrollY = y;
    if (hold) {
      // 手機放開後的慣性捲動不會觸發 touchmove：被推走就拉回停留位置，推動量算進累積的滑動
      if (Math.abs(y - hold.y) > 1) {
        absorb(y - hold.y);
        window.scrollTo({ top: hold.y, behavior: "instant" });
        lastScrollY = hold.y;
      }
      return;
    }
    if (isLocked()) return;
    for (const s of holdSections) {
      const top = s.getBoundingClientRect().top;
      if (top > innerHeight * 0.5) visited.delete(s); // 往回滑到它上方 → 下次再停
      else if (down && !visited.has(s) && top <= 2 && top > -innerHeight * 0.6) {
        startHold(s);
        break;
      }
    }
  },
  { passive: true },
);
addEventListener(
  "wheel",
  (e) => {
    if (!hold) return;
    if (e.cancelable) e.preventDefault();
    absorb(e.deltaY * (e.deltaMode === 1 ? 16 : 1));
  },
  { passive: false },
);
addEventListener("touchstart", (e) => (touchY = e.touches[0]?.clientY ?? null), { passive: true });
addEventListener(
  "touchmove",
  (e) => {
    const y = e.touches[0]?.clientY;
    if (hold) {
      if (e.cancelable) e.preventDefault();
      if (touchY != null && y != null) absorb(touchY - y);
    }
    touchY = y ?? touchY;
  },
  { passive: false },
);
addEventListener("keydown", (e) => {
  if (!hold || !["ArrowDown", "PageDown", " ", "Space"].includes(e.key)) return;
  e.preventDefault();
  absorb(innerHeight * 0.4);
});

function unlockIntro() {
  window.scrollTo(0, 0);
  document.documentElement.classList.remove("intro-lock");
  removeEventListener("wheel", blockScroll);
  removeEventListener("touchmove", blockScroll);
  removeEventListener("keydown", blockKeys);
}
let introT = 0,
  fullReady = false;
garment.upgraded.then(() => ((fullReady = true), (document.body.dataset.full = "1")));
const sm = (a, b, v) => {
  const k = Math.max(0, Math.min(1, (v - a) / (b - a)));
  return k * k * (3 - 2 * k);
};
// 場景外部燈光（開場時調暗）；蛹內部的光（PointLight）不動
const extLights = [];
scene.traverse((o) => o.isDirectionalLight && extLights.push([o, o.intensity]));
garment.group.traverse((o) => o.isSpotLight && extLights.push([o, null])); // 臉部補光：每格由 garment.update 設定，之後再乘
const envBase = scene.environmentIntensity;
const sprites = [];
garment.group.traverse((o) => o.isSprite && sprites.push(o));

// AI 生成的抓殼畫格：快滑到最後一章時才開始下載（不跟開場的模型搶頻寬）
let gripSeq = null,
  gripRequested = false;
function requestGrip() {
  gripRequested = true;
  loadGripSeq(document.querySelector(".grip-seq"))
    .then(async (g) => {
      await g.ready;
      gripSeq = g;
    })
    .catch((e) => console.warn("grip frames", e));
}

const look = new THREE.Vector3();
const clock = new THREE.Clock();
let running = true;
document.addEventListener("visibilitychange", () => {
  running = !document.hidden;
  if (running) {
    clock.getDelta();
    requestAnimationFrame(frame);
  }
});

let lastPhase = -1;
let grip = forced === keys.length - 1 ? 1 : 0;
const gripFixed = new URLSearchParams(location.search).get("grip"); // 截圖檢查用：固定抓殼進度
function frame() {
  if (!running) return;
  const rawDt = clock.getDelta();
  const dt = Math.min(rawDt, 0.1);
  const time = clock.elapsedTime;
  const fi = forced !== null ? forced : scrollIndex();
  if (!gripRequested && fi > keys.length - 3) requestGrip();
  const goal = target(fi);
  const k = reduceMotion || forced !== null ? 1 : 1 - Math.pow(0.003, dt);
  for (const f of NUM) cur[f] += (goal[f] - cur[f]) * k;
  for (let i = 0; i < 3; i++) cur.t[i] += (goal.t[i] - cur.t[i]) * k;

  // 到了最後定點、鏡頭停穩後，播放 AI 抓殼影片：雙手從蛹裡伸出、手指一根根抓住殼、再往外撐開（約 8 秒）；往回滑則倒帶收回
  const endW = Math.max(0, Math.min(1, (fi - (keys.length - 2)) / 1)) ** 2;
  const atEnd = (forced !== null ? forced : fi) > keys.length - 1.15;
  const settled = Math.abs(cur.dist - goal.dist) < 0.06 && Math.abs(cur.az - goal.az) < 1.5 && Math.abs(cur.t[1] - goal.t[1]) < 0.01;
  grip = Math.min(1, Math.max(0, grip + (atEnd && (settled || grip > 0) ? dt / 8 : -dt / 3))); // 約 8 秒播完
  if (gripFixed !== null) grip = +gripFixed;
  const gripEased = grip * grip * (3 - 2 * grip);
  // 開場時間軸
  // 完整模型還沒到：停在「蛹裡的光呼吸」（ip ≤ 0.42），到了才閃一下、亮燈（開場最慢約 3 秒）
  if (INTRO && introFixed === null) {
    const hold = !fullReady && time < 1.3; // 最多等到 1.3 秒（開場最慢約 3 秒結束；完整模型之後到了仍會自動換上）
    const idt = Math.min(rawDt, 0.5); // 開場用真實經過時間（畫面卡頓時也準時）
    introT = Math.min(hold ? Math.max(introT, Math.min(introT + idt, 0.42 * INTRO)) : introT + idt, INTRO);
  }
  const ip = introFixed !== null ? +introFixed : INTRO ? introT / INTRO : 1;
  // 開場跑完（燈全亮）才能往下滑，並淡入「向下滑動」
  if (isLocked() && (time >= 3 || forced !== null)) unlockIntro(); // 開場 3 秒（從 3D 模型出現起算）後可以往下滑
  if (hold) performance.now() >= hold.until ? endHold() : holdFrame(performance.now());
  garment.set({ silk: cur.silk, reveal: cur.reveal, burst: cur.burst, grip: gripSeq ? 0 : gripEased }); // 有影片時 3D 不自己撥開
  const rh = rhythmAt(time);
  const phase = rh.phase;
  let level = rh.level + (1 - rh.level) * endW; // 最後定點：燈穩定全亮，與影片一致
  // 開場：蛹裡的光呼吸、漸強 → 閃一下 → 交回正常節奏
  const breath = (0.18 + 0.85 * sm(0.03, 0.42, ip)) * (0.78 + 0.22 * Math.sin(time * 2.6));
  const burstFlash = Math.exp(-Math.pow((ip - 0.47) / 0.06, 2)) * 1.3; // 閃光柔一點、峰值低一點
  const handBack = sm(0.52, 0.9, ip);
  if (ip < 1) level = Math.max(breath, burstFlash) * (1 - handBack) + level * handBack;
  garment.update(time, level);
  // 外面的燈：開場幾乎全暗，閃過之後慢慢亮起
  const ext = 0.03 + 0.97 * sm(0.45, 0.88, ip);
  for (const [l, base] of extLights) l.intensity = (base ?? l.intensity) * ext;
  scene.environmentIntensity = envBase * ext;
  if (gripSeq) {
    const vo = Math.min(1, grip / 0.08);
    gripSeq.draw(grip, vo, time);
    // 影片開始動之後，3D 畫面淡掉：影片裡她往前傾、輪廓和 3D 不同，兩側不能再透出 3D 的蛹（灰色殘影）
    renderer.domElement.style.opacity = String(1 - Math.min(1, Math.max(0, (grip - 0.06) / 0.1)));
    // 影片播放時，3D 的背景光暈淡掉（影片是純黑底，兩邊才不會出現亮帶）
    for (const sp of sprites) sp.material.opacity *= 1 - vo; // garment.update 每格都會重設，這裡乘上去即可
  }

  const drift = reduceMotion ? 0 : Math.sin(time * 0.15) * (cur.dist > 3 ? 6 : 3) * (1 - endW);
  const az = THREE.MathUtils.degToRad(cur.az + drift);
  const el = THREE.MathUtils.degToRad(cur.el);
  const dist = cur.dist * (innerWidth / innerHeight < 0.75 ? 1.25 : 1);
  look.set(cur.t[0] * HEIGHT, cur.t[1] * HEIGHT, cur.t[2] * HEIGHT);
  camera.position.set(
    look.x + Math.sin(az) * Math.cos(el) * dist,
    look.y + Math.sin(el) * dist,
    look.z + Math.cos(az) * Math.cos(el) * dist,
  );
  camera.lookAt(look);
  // 模型偏右（文字在左）；最後一章回到正中央，與抓殼影片對齊
  {
    const cw = renderer.domElement.clientWidth || innerWidth,
      ch = renderer.domElement.clientHeight || innerHeight;
    const shift = (cw / ch < 0.75 ? 0.14 : 0.12) * (1 - endW);
    if (shift > 0.001) camera.setViewOffset(cw, ch, -cw * shift, 0, cw, ch);
    else camera.clearViewOffset();
  }
  renderer.render(scene, camera);

  const max = document.documentElement.scrollHeight - innerHeight;
  bar.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
  updateActive(hold?.section ?? null);
  // 停止捲動滿 1 秒 → 再提示一次（經過第一個章節後才開始；停留中與最後一章不提示）
  idleHint = holdsDone > 0 && !hold && performance.now() - lastActivity > 1000 && fi < keys.length - 1.5;
  document.body.classList.toggle("scrolled", scrollY > 40 && !idleHint);
  if (introFixed !== null || location.search.includes("debughint")) document.body.dataset.hint = `${idleHint ? 1 : 0}|${holdsDone}|${Math.round(performance.now() - lastActivity)}`; // 往下滑之後「向下滑動」提示淡出
  veil.style.opacity = String(0.45 * (1 - Math.min(scrollY / innerHeight, 1)));
  if (phase !== lastPhase) {
    tagSteps.forEach((el) => el.classList.toggle("active", +el.dataset.phase === phase));
    rhythmSteps.forEach((el) => el.classList.toggle("active", +el.dataset.phase === phase));
    lastPhase = phase;
  }
  const rhythmFlash = phase === 2 ? Math.min(1, Math.max(0, level - 0.6)) * (1 - endW) : 0;
  document.body.style.setProperty("--flash", Math.max(rhythmFlash * handBack, burstFlash * 0.22));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
