// 共用：場景、燈光、白光節奏。
import * as THREE from "three";
import { RoomEnvironment } from "./vendor/RoomEnvironment.js";

/* 白光節奏：亮 → 暗 → 閃一下，循環 */
export const RHYTHM = [
  { key: "bright", label: "亮", dur: 2.4 },
  { key: "dim", label: "暗", dur: 2.0 },
  { key: "flash", label: "閃", dur: 0.5 },
];
const CYCLE = RHYTHM.reduce((s, r) => s + r.dur, 0);
const LOW = 0.25; // 暗的時候保留的亮度（越高越柔和，亮暗落差越小）
const ease = (k) => (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);

/** 回傳 { level: 0~1.6, phase: 0 亮 | 1 暗 | 2 閃 } */
export function rhythmAt(t, mode = "rhythm") {
  if (mode === "on") return { level: 1, phase: 0 };
  if (mode === "off") return { level: 0, phase: 1 };
  if (mode === "flash") return { level: 1.6, phase: 2 }; // 截圖檢查用：固定在閃的最亮瞬間
  let x = t % CYCLE;
  if (x < RHYTHM[0].dur) return { level: LOW + (1 - LOW) * ease(Math.min(x / 1.2, 1)), phase: 0 }; // 1.2 秒慢慢亮起
  x -= RHYTHM[0].dur;
  if (x < RHYTHM[1].dur) return { level: 1 - (1 - LOW) * ease(Math.min(x / 1.2, 1)), phase: 1 }; // 1.2 秒慢慢變暗
  x -= RHYTHM[1].dur;
  const k = x / RHYTHM[2].dur;
  return { level: LOW + 1.15 * Math.exp(-Math.pow((k - 0.3) / 0.16, 2)), phase: 2 }; // 閃一下：柔和一點
}

export function createStage(canvas, { mobile = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.75 : 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(0x000000, 1);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x000000, 6, 12);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.35;

  const camera = new THREE.PerspectiveCamera(30, 1, 0.02, 40);
  const key = new THREE.DirectionalLight(0xffffff, 1.2); // 太強會讓蛹頂過曝、看不到骨架
  key.position.set(1.4, 3, 2.6);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xe8eeff, 1.5);
  rim.position.set(-2.4, 2.4, -2.4);
  scene.add(rim);
  const rim2 = new THREE.DirectionalLight(0xffffff, 0.8);
  rim2.position.set(2.4, 1.2, -2.0);
  scene.add(rim2);
  return { renderer, scene, camera };
}

/** 依視窗調整：直式手機視野放寬 */
// 用畫布實際顯示的大小（CSS 是 100lvh）：手機網址列收起時 innerHeight 會變，但畫布不變，用 innerHeight 會把模型拉寬
export function fitCamera(renderer, camera) {
  const c = renderer.domElement;
  const w = c.clientWidth || innerWidth,
    h = c.clientHeight || innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w / h < 0.75 ? 40 : 30;
  camera.updateProjectionMatrix();
}

/** 載入進度條 */
export function loaderUI() {
  const el = document.querySelector(".loading");
  const bar = el?.querySelector("i");
  return {
    progress(p) {
      if (bar) bar.style.transform = `scaleX(${p})`;
    },
    done() {
      el?.classList.add("hide");
      document.body.classList.add("ready");
    },
    fail(msg) {
      if (el) el.querySelector("span").textContent = msg;
    },
  };
}
