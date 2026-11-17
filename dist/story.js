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
  hero: { az: 0, el: 4, dist: 5.6, t: [0, 0.5, 0], silk: 1, reveal: 1, burst: 0.4 },
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

const io = new IntersectionObserver((es) => es.forEach((e) => e.target.classList.toggle("on", e.isIntersecting)), {
  threshold: 0.35,
});
sections.forEach((s) => io.observe(s));
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
  ui.fail("模型載入失敗，請重新整理");
  throw e;
}
scene.add(garment.group);
ui.done();

// 開場「破蛹」時間軸（秒）：全場暗，只有蛹裡的光像呼吸一樣透出（看得到燈籠骨架）
// → 內光越來越亮 → 閃一下 → 外面的燈慢慢亮起，她的臉與全身出現 → 進入 亮/暗/閃 循環
const INTRO = reduceMotion || forced !== null ? 0 : 6;
const introFixed = new URLSearchParams(location.search).get("intro"); // 截圖檢查用：固定開場進度 0~1
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

// AI 生成的抓殼畫格：背景載入，不擋住 3D
let gripSeq = null;
loadGripSeq(document.querySelector(".grip-seq"))
  .then(async (g) => {
    await g.ready;
    gripSeq = g;
  })
  .catch((e) => console.warn("grip frames", e));

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
  const dt = Math.min(clock.getDelta(), 0.1);
  const time = clock.elapsedTime;
  const fi = forced !== null ? forced : scrollIndex();
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
  const ip = introFixed !== null ? +introFixed : INTRO ? Math.min(1, time / INTRO) : 1;
  garment.set({ silk: cur.silk, reveal: cur.reveal, burst: cur.burst, grip: gripSeq ? 0 : gripEased }); // 有影片時 3D 不自己撥開
  const rh = rhythmAt(time);
  const phase = rh.phase;
  let level = rh.level + (1 - rh.level) * endW; // 最後定點：燈穩定全亮，與影片一致
  // 開場：蛹裡的光呼吸、漸強 → 閃一下 → 交回正常節奏
  const breath = (0.18 + 0.85 * sm(0.03, 0.42, ip)) * (0.78 + 0.22 * Math.sin(time * 2.6));
  const burstFlash = Math.exp(-Math.pow((ip - 0.47) / 0.045, 2)) * 1.6;
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
  renderer.render(scene, camera);

  const max = document.documentElement.scrollHeight - innerHeight;
  bar.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
  veil.style.opacity = String(0.45 * (1 - Math.min(scrollY / innerHeight, 1)));
  if (phase !== lastPhase) {
    tagSteps.forEach((el) => el.classList.toggle("active", +el.dataset.phase === phase));
    rhythmSteps.forEach((el) => el.classList.toggle("active", +el.dataset.phase === phase));
    lastPhase = phase;
  }
  const rhythmFlash = phase === 2 ? Math.min(1, Math.max(0, level - 0.6)) * (1 - endW) : 0;
  document.body.style.setProperty("--flash", Math.max(rhythmFlash * handBack, burstFlash * 0.45));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
