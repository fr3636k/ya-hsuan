// 3D 檢視子網頁：自由旋轉、細節視角、破蛹與白光調整。
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { createStage, fitCamera, loaderUI, rhythmAt } from "./stage.js";
import { loadGarment, HEIGHT } from "./garment-glb.js";
import { MODEL } from "./model-config.js";

const mobile = matchMedia("(max-width: 720px)").matches;
const canvas = document.getElementById("stage");
const { renderer, scene, camera } = createStage(canvas, { mobile });
const ui = loaderUI();
const $ = (id) => document.getElementById(id);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.35;
controls.maxDistance = 9;
controls.maxPolarAngle = Math.PI * 0.62;
controls.autoRotateSpeed = 0.7;

// 視角：[方位角, 仰角, 距離, 注視點(相對身高)]
const VIEWS = {
  front: [0, 4, 5.4, [0, 0.5, 0]],
  side: [90, 4, 5.4, [0, 0.5, 0]],
  back: [180, 6, 5.4, [0, 0.5, 0]],
  face: [12, 3, 1.1, [0, 0.87, 0.02]],
  knot: [20, 14, 2.0, [0, 0.6, 0.03]],
  fabric: [200, 10, 2.4, [0, 0.78, 0]],
  tail: [330, 12, 2.4, [0, 0.13, 0]],
};
function viewPose(name) {
  const [az, el, dist, t] = VIEWS[name];
  const target = new THREE.Vector3(t[0] * HEIGHT, t[1] * HEIGHT, t[2] * HEIGHT);
  const a = THREE.MathUtils.degToRad(az),
    e = THREE.MathUtils.degToRad(el);
  const d = dist * (innerWidth / innerHeight < 0.75 ? 1.25 : 1);
  const pos = target.clone().add(new THREE.Vector3(Math.sin(a) * Math.cos(e) * d, Math.sin(e) * d, Math.cos(a) * Math.cos(e) * d));
  return { pos, target };
}
let tween = null;
function goView(name, instant = false) {
  const v = viewPose(name);
  if (instant) {
    camera.position.copy(v.pos);
    controls.target.copy(v.target);
    return;
  }
  tween = { t: 0, fromP: camera.position.clone(), fromT: controls.target.clone(), toP: v.pos, toT: v.target };
}
const startView = VIEWS[location.hash.slice(1)] ? location.hash.slice(1) : "front";
goView(startView, true);
// 比對用：?cam=方位角,仰角,距離,注視高度(相對身高)&clean → 固定鏡頭、隱藏介面
const qs = new URLSearchParams(location.search);
const camQ = qs.get("cam");
if (camQ) {
  const [az, el, dist, ty] = camQ.split(",").map(Number);
  VIEWS.custom = [az, el, dist, [0, ty, 0]];
  goView("custom", true);
}
if (qs.has("clean")) document.body.classList.add("clean");

/* ---------- 控制面板 ---------- */
const values = {};
// 截圖用：?grip=、?silk=、?reveal= 可直接指定滑桿值
for (const id of ["grip", "silk", "reveal"]) {
  const q = new URLSearchParams(location.search).get(id);
  if (q !== null) $(id).value = q;
}
["grip", "reveal", "silk", "speed"].forEach((id) => {
  const el = $(id);
  const out = el.parentElement.querySelector("output");
  const sync = () => {
    values[id] = parseFloat(el.value);
    out.textContent = id === "speed" ? `${values[id].toFixed(2)}×` : `${Math.round(values[id] * 100)}`;
  };
  el.addEventListener("input", sync);
  sync();
});
function segment(groupId, attr, onPick, initial) {
  const btns = [...$(groupId).querySelectorAll("button")];
  const pick = (b) => btns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  btns.forEach((b) =>
    b.addEventListener("click", () => {
      pick(b);
      onPick(b.dataset[attr]);
    }),
  );
  if (initial) pick(btns.find((b) => b.dataset[attr] === initial));
}
segment(
  "views",
  "view",
  (v) => {
    goView(v);
    $("spin").checked = v === "front";
  },
  startView,
);
$("spin").checked = startView === "front" && !camQ;
let lightMode = qs.get("light") || "rhythm"; // ?light=on|off 比對截圖用
segment("lightMode", "mode", (m) => (lightMode = m));
controls.addEventListener("start", () => {
  tween = null;
  $("spin").checked = false;
});
const panel = $("panel");
$("panelToggle").addEventListener("click", () => {
  const open = panel.classList.toggle("open");
  $("panelToggle").setAttribute("aria-expanded", String(open));
});

/* ---------- 本機參考照片（serve.mjs 提供；發布版不存在則隱藏） ---------- */
fetch("./_ref/list.json")
  .then((r) => (r.ok ? r.json() : Promise.reject()))
  .then((list) => {
    if (!list.length) return;
    const big = $("refBig");
    list.forEach((name) => {
      const img = new Image();
      img.loading = "lazy";
      img.src = `./_ref/${encodeURIComponent(name)}`;
      img.alt = name;
      img.addEventListener("click", () => {
        big.src = img.src;
        big.classList.add("show");
      });
      $("refs").append(img);
    });
    big.addEventListener("click", () => big.classList.remove("show"));
    $("refGroup").hidden = false;
  })
  .catch(() => {});

addEventListener("resize", () => fitCamera(renderer, camera));
fitCamera(renderer, camera);

let garment;
try {
  garment = await loadGarment(MODEL.url, MODEL.cfg, (p) => ui.progress(p));
} catch (e) {
  ui.fail("模型載入失敗，請重新整理");
  throw e;
}
scene.add(garment.group);
// ?nofx：隱藏光點、粒子、絲線（給 AI 影片當起始畫面用）
// ?unlit：只顯示貼圖顏色（除錯用）
if (qs.has("plain")) garment.group.traverse((o) => o.isMesh && o.material.map && (o.material = new THREE.MeshStandardMaterial({ map: o.material.map, roughness: 0.7, side: THREE.DoubleSide })));
if (qs.has("unlit")) garment.group.traverse((o) => o.isMesh && o.material.map && (o.material = new THREE.MeshBasicMaterial({ map: o.material.map, side: THREE.DoubleSide })));
if (qs.has("nofx")) garment.group.traverse((o) => (o.isPoints || o.isLineSegments || o.isSprite) && (o.visible = false, o.layers.set(31)));
ui.done();

const tagSteps = [...document.querySelectorAll(".lt-step")];
const clock = new THREE.Clock();
let lightTime = 0,
  lastPhase = -1;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  lightTime += dt * values.speed;
  if (tween) {
    tween.t = Math.min(1, tween.t + dt / 1.1);
    const e = tween.t * tween.t * (3 - 2 * tween.t);
    camera.position.lerpVectors(tween.fromP, tween.toP, e);
    controls.target.lerpVectors(tween.fromT, tween.toT, e);
    if (tween.t >= 1) tween = null;
  }
  controls.autoRotate = $("spin").checked;
  controls.update();

  garment.set({ grip: values.grip, reveal: values.reveal, silk: 1 - values.silk, burst: 0.4 });
  const { level, phase } = rhythmAt(lightTime, lightMode);
  garment.update(clock.elapsedTime, level);
  renderer.render(scene, camera);

  if (phase !== lastPhase) {
    tagSteps.forEach((el) => el.classList.toggle("active", lightMode === "rhythm" && +el.dataset.phase === phase));
    lastPhase = phase;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
