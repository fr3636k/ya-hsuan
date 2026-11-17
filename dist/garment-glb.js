// 載入處理好的服裝 GLB（由 scripts/process-model.py 產生），並加上：
// - 只讓蛹的白布發光（emissiveTexture 為發光遮罩）、白光節奏
// - 照片臉部投影（取代 AI 生成的模糊臉）
// - 破蛹顯現（由下往上溶解）、LED 光點、內部光暈、臉部補光、絲線外繭、光塵
import * as THREE from "three";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { DRACOLoader } from "./vendor/DRACOLoader.js";
import { createFigure } from "./figure.js";

export const HEIGHT = 1.75;
const clamp01 = (x) => Math.min(1, Math.max(0, x));
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}
function radialSprite(stops) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  for (const [o, col] of stops) grad.addColorStop(o, col);
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const additive = (map, extra = {}) => ({ map, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, ...extra });

// 模型用 Draco 壓縮（完整網格、不減面），解碼器放在 vendor/draco
function gltfLoader() {
  const draco = new DRACOLoader().setDecoderPath(new URL("./vendor/draco/", import.meta.url).href);
  return new GLTFLoader().setDRACOLoader(draco);
}

export async function loadGarment(url, cfg = {}, onProgress) {
  cfg = { glowMin: 0.53, glowMax: 1.0, glowCenter: [0, 0.76, -0.02], glowStrength: 0.42, ...cfg };
  const [gltf, leds, faceTex] = await Promise.all([
    gltfLoader().loadAsync(url, (e) => onProgress && e.total && onProgress(e.loaded / e.total)),
    fetch(url.replace(/\.glb$/, ".leds.json")).then((r) => (r.ok ? r.json() : [])),
    cfg.face ? new THREE.TextureLoader().loadAsync(cfg.face.url) : null,
  ]);
  const model = gltf.scene;
  const group = new THREE.Group();
  group.add(model);

  const uniforms = {
    uLevel: { value: 1 },
    uReveal: { value: 1 },
    uGlow: { value: cfg.glowStrength },
    uGlowMin: { value: cfg.glowMin * HEIGHT },
    uGlowMax: { value: cfg.glowMax * HEIGHT },
    uFace: { value: faceTex },
    uFaceOn: { value: faceTex ? 1 : 0 },
    uFaceRect: { value: new THREE.Vector4() },
    uFaceMinZ: { value: 0 },
    uRibs: { value: cfg.ribs ?? 1 },
    uGauze: { value: cfg.gauze ?? 1 },
    uBackZ: { value: cfg.backZ ?? -0.15 },
    uOpen: { value: 0 },
    uOpenX: { value: cfg.openCenterX ?? 0.03 },
    uOpenPush: { value: cfg.openPush ?? 0.05 },
    uCrinkle: { value: cfg.crinkle ?? 1 },
    uBody: { value: cfg.bodyMask ?? 1 },
    uUseMap: { value: cfg.useEmissiveMap ? 1 : 0 },
    // 臉的範圍（公尺）：[半寬 x, 下緣 y, 上緣 y, 最小 z]；範圍內不發光、不套布料效果
    uFacing: { value: cfg.facingGlow ?? 0 },
    uCocoonShade: { value: cfg.cocoonShade ?? 0 },
    uFaceBox: { value: new THREE.Vector4(...(cfg.faceBox ?? [0, 0, 0, 9])) },
  };
  if (faceTex) {
    faceTex.colorSpace = THREE.SRGBColorSpace;
    const [x0, y0, x1, y1] = cfg.face.rect;
    uniforms.uFaceRect.value.set(x0 * HEIGHT, y0 * HEIGHT, x1 * HEIGHT, y1 * HEIGHT);
    uniforms.uFaceMinZ.value = cfg.face.minZ * HEIGHT;
  }

  model.traverse((o) => {
    if (!o.isMesh) return;
    const mat = o.material;
    mat.side = THREE.DoubleSide;
    mat.metalness = 0;
    mat.roughness = 0.66;
    mat.emissive = new THREE.Color(0xffffff);
    mat.emissiveIntensity = 1;
    if (mat.map) mat.map.anisotropy = 8;
    if (mat.normalMap && cfg.normalScale !== undefined) mat.normalScale.setScalar(cfg.normalScale);
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, uniforms);
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNrm;\nuniform float uOpen, uOpenX, uOpenPush;")
        .replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
// 撥開：開口兩側（上半身正面）往外推，離開口越近推越多
{
  float dx = transformed.x - uOpenX;
  float w = exp(-pow((abs(dx) - 0.1) / 0.1, 2.0))
          * smoothstep(1.38, 1.5, transformed.y) * (1.0 - smoothstep(1.7, 1.8, transformed.y)) // 只撥開帽沿一帶，胸口不拉開（避免露出縫）
          * smoothstep(-0.02, 0.1, transformed.z);
  transformed.x += sign(dx) * uOpen * uOpenPush * w;
  transformed.z -= uOpen * uOpenPush * 0.35 * w;
}`,
        )
        .replace(
          "#include <worldpos_vertex>",
          "#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWNrm = normalize(mat3(modelMatrix) * objectNormal);",
        );
      sh.fragmentShader = sh.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
varying vec3 vWPos;
varying vec3 vWNrm;
uniform vec4 uFaceBox;
uniform float uFacing, uCocoonShade;
float faceZone = 0.0;
uniform float uLevel, uReveal, uGlow, uGlowMin, uGlowMax, uFaceOn, uFaceMinZ, uRibs, uGauze, uBackZ, uCrinkle, uBody, uUseMap;
uniform sampler2D uFace;
uniform vec4 uFaceRect;
float hash(vec3 p){ return fract(sin(dot(p, vec3(12.9898,78.233,37.719))) * 43758.5453); }
float faceW = 0.0;
float rib = 0.0;
float body = 0.0;
float crinkle = 0.5;
vec3 faceColor = vec3(0.0);
float vnoise(vec3 p){ vec3 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  return mix(mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x), mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x), mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z); }
// 壓皺紋：細密、略帶起伏的橫向皺褶（照片中的珠光壓皺布）
float crinkleAt(vec3 p){
  float warp = vnoise(p * 18.0) * 5.0 + vnoise(p * 45.0) * 1.5;
  float lines = 0.5 + 0.5 * sin(p.y * 520.0 + warp * 3.0);
  float breakup = smoothstep(0.25, 0.75, vnoise(p * vec3(30.0, 6.0, 30.0)));
  return mix(0.5, lines, 0.35 + 0.65 * breakup);
}
float cocoon = 0.0;
// 骨架條帶：8 條由腰往上放射的縱向條帶 + 2 條起伏的橫向弧帶（公尺）
float ribMask(vec3 p) {
  // 蛹的實際中軸（實測）：1.33–1.54 m 在 z≈−0.023，往頂端前傾到 +0.05；截面左右較寬（約 1.3:1）
  float cz = mix(-0.023, 0.05, smoothstep(1.54, 1.75, p.y));
  vec2 q = vec2((p.x - 0.0) / 1.3, p.z - cz);
  float r = length(q) * 1.14;
  float ang = atan(q.x, q.y);
  float k = 12.0; // 照片背面約 6 條 → 一圈 12 條
  float a = abs(fract(ang / 6.2831853 * k + 0.5) - 0.5) * 6.2831853 / k * r;
  float w = 0.012;
  float m = 1.0 - smoothstep(w, w + 0.004, a);
  // 一條橫向帶：腰上方（前面略高）
  float h1 = abs(p.y - (1.12 + 0.03 * cos(ang)));
  m = max(m, 1.0 - smoothstep(0.012, 0.016, h1));
  return m;
}
// 與最近縱向骨架的距離（公尺），給燈籠面板的明暗用
float ribDist(vec3 p) {
  float cz = mix(-0.023, 0.05, smoothstep(1.54, 1.75, p.y));
  vec2 q = vec2(p.x / 1.3, p.z - cz);
  float r = length(q) * 1.14;
  float ang = atan(q.x, q.y);
  return abs(fract(ang / 6.2831853 * 12.0 + 0.5) - 0.5) * 6.2831853 / 12.0 * r;
}`,
        )
        .replace(
          "#include <clipping_planes_fragment>",
          `#include <clipping_planes_fragment>
// 破蛹顯現：由下往上帶雜訊的溶解
float n = hash(floor(vWPos * 60.0));
float edge = uReveal * 2.1 - 0.15 - vWPos.y + n * 0.12;
if (edge < 0.0) discard;`,
        )
        .replace(
          "#include <map_fragment>",
          `#include <map_fragment>
if (uFaceOn > 0.5) {
  vec2 fuv = (vWPos.xy - uFaceRect.xy) / (uFaceRect.zw - uFaceRect.xy);
  if (fuv.x > 0.0 && fuv.x < 1.0 && fuv.y > 0.0 && fuv.y < 1.0 && vWPos.z > uFaceMinZ) {
    vec4 f = texture2D(uFace, fuv);
    // AI 模型臉部的法線可能朝內，用絕對值判斷是否面向正前方
    faceW = f.a; // 不看法線，避免下巴接縫露出白線
    faceColor = f.rgb;
  }
}
// 蛹的範圍（腰以上）
// 正面中央是身體（臉、襯衫），不是蛹的布：不畫骨架、不發光
// 以蛹的實際中軸判斷「前方」（蛹前傾後頂端的背面 z 也 > 0，不能用固定 z）
float czB = mix(-0.023, 0.05, smoothstep(1.54, 1.75, vWPos.y));
body = (1.0 - smoothstep(0.08, 0.13, abs(vWPos.x))) * smoothstep(czB + 0.02, czB + 0.08, vWPos.z) * (1.0 - smoothstep(1.64, 1.68, vWPos.y)) * uBody;
cocoon = smoothstep(uGlowMin, uGlowMin + 0.06, vWPos.y) * (1.0 - faceW) * (1.0 - body);
faceZone = (1.0 - smoothstep(uFaceBox.x - 0.01, uFaceBox.x + 0.01, abs(vWPos.x)))
  * smoothstep(uFaceBox.y - 0.01, uFaceBox.y + 0.01, vWPos.y) * (1.0 - smoothstep(uFaceBox.z - 0.01, uFaceBox.z + 0.01, vWPos.y))
  * smoothstep(uFaceBox.w - 0.01, uFaceBox.w + 0.01, vWPos.z);
cocoon *= 1.0 - faceZone;
#ifdef USE_EMISSIVEMAP
// Tripo 模型：發光遮罩為 0 的地方（臉、頸、頭髮）不是蛹的布，不套布料光澤與發光
if (uUseMap > 0.5) cocoon *= smoothstep(0.2, 0.8, texture2D(emissiveMap, vEmissiveMapUv).r);
#endif
// 骨架：白色條帶較亮、較平滑，邊緣一道細陰影讓它浮起來
rib = ribMask(vWPos) * cocoon * uRibs;
// 條帶是不透光的厚布：顏色較灰白、不發光，發光時形成骨架剪影
diffuseColor.rgb = mix(diffuseColor.rgb, mix(vec3(0.68, 0.69, 0.7), vec3(0.6, 0.6, 0.62), smoothstep(0.05, uBackZ, vWPos.z) * uLevel), rib);
crinkle = mix(0.5, crinkleAt(vWPos), uCrinkle);
// 燈籠：背面的布繃在骨架上，只留輕微皺褶
float lanternD = smoothstep(0.05, uBackZ, vWPos.z) * cocoon;
diffuseColor.rgb *= 1.0 - cocoon * (1.0 - rib) * (0.5 - crinkle) * 0.28 * (1.0 - lanternD * 0.8);
// 臉：直接呈現精修照片（不受 3D 凹凸光影影響）
diffuseColor.rgb *= 1.0 - faceW;
// 紗：條帶之間的布較薄，底色壓暗一點，讓透出的光成為主角
float facingCam = abs(dot(normalize(vWNrm), normalize(cameraPosition - vWPos)));
float gauze = cocoon * (1.0 - rib) * uGauze;
diffuseColor.rgb *= 1.0 - gauze * 0.12;
diffuseColor.rgb *= 1.0 - uCocoonShade * cocoon; // 蛹的布外表略暗，讓內部透出的光成為主角
// 臉、身體的皺褶（眼皮、鼻孔）露出的網格背面：是陰影，不是布的內側 → 壓暗，避免出現白線
if (!gl_FrontFacing) diffuseColor.rgb *= mix(0.25, 1.0, cocoon);`,
        )
        .replace(
          "#include <roughnessmap_fragment>",
          `#include <roughnessmap_fragment>
roughnessFactor = mix(roughnessFactor, 0.38 + crinkle * 0.2, cocoon * (1.0 - rib));
roughnessFactor = mix(roughnessFactor, 0.85, faceZone);`,
        )
        .replace(
          "#include <aomap_fragment>",
          `#include <aomap_fragment>
// 臉：皮膚不反射環境（眼皮、鼻孔的皺褶在斜角會反出白光）
reflectedLight.indirectSpecular *= 1.0 - faceZone * 0.92;
reflectedLight.directSpecular *= 1.0 - faceZone * 0.8;`,
        )
        .replace(
          "#include <emissivemap_fragment>",
          `#include <emissivemap_fragment>
// 蛹的布面均勻發光（不用 AI 貼圖的斑駁遮罩，否則光會一塊一塊）
totalEmissiveRadiance = mix(totalEmissiveRadiance, emissive, cocoon * (1.0 - uUseMap));
float band = smoothstep(uGlowMin, uGlowMin + 0.08, vWPos.y) * (1.0 - smoothstep(uGlowMax - 0.06, uGlowMax, vWPos.y));
// 光源在背後的圓弧：越靠背面越亮；條帶擋光
float back = mix(0.35, 1.0, smoothstep(0.1, uBackZ, vWPos.z));
// 燈籠：越靠背面，光越均勻柔和、面板中央越亮
float lantern = smoothstep(0.05, uBackZ, vWPos.z) * cocoon;
float panel = smoothstep(0.012, 0.07, ribDist(vWPos));
back *= 1.0 + lantern * (panel * 0.55 - 0.1);
// 紗的透光：正面看光比較能透出來（薄），斜看較密
float sheer = mix(0.55, 1.25, facingCam * gauze + (1.0 - gauze) * 0.5);
totalEmissiveRadiance *= band * uLevel * uGlow * (1.0 - faceW) * (1.0 - body * 0.85) * back * sheer * (1.0 - rib * 0.97) * mix(0.75 + crinkle * 0.5, 1.0, lantern * 0.85);

// 布的內側（背面）補一點柔光，避免看起來像破洞的黑塊
// 燈籠：布面正對鏡頭的地方（面板中央、布最薄）透光最強，折邊與骨架縫較暗
totalEmissiveRadiance *= mix(1.0, mix(0.2, 1.35, pow(facingCam, 1.6)), uFacing * cocoon);
totalEmissiveRadiance *= 1.0 - faceZone;
if (!gl_FrontFacing) totalEmissiveRadiance += diffuseColor.rgb * 0.32 * cocoon;
// 照片臉部：直接呈現照片亮度，不被帽沿陰影壓暗
totalEmissiveRadiance += faceColor * faceW * 0.92;
// 溶解邊緣的白光
float rim = 1.0 - smoothstep(0.0, 0.05, edge);
totalEmissiveRadiance += vec3(rim * 2.5) * step(uReveal, 0.999);`,
        );
    };
    mat.needsUpdate = true;
  });

  // 外輪廓半徑（給絲線用）
  const radii = new Array(48).fill(0.05);
  model.traverse((o) => {
    if (!o.isMesh) return;
    const pos = o.geometry.attributes.position;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i += 3) {
      v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
      const k = Math.round(clamp01(v.y / HEIGHT) * (radii.length - 1));
      radii[k] = Math.max(radii[k], Math.hypot(v.x, v.z));
    }
  });
  for (let pass = 0; pass < 2; pass++)
    for (let k = 1; k < radii.length - 1; k++) radii[k] = Math.max(radii[k], (radii[k - 1] + radii[k + 1]) / 2);
  const radiusAt = (y) => radii[Math.max(0, Math.min(radii.length - 1, Math.round((y / HEIGHT) * (radii.length - 1))))];

  /* ---------- LED 光點 ---------- */
  const dotTex = radialSprite([
    [0, "rgba(255,255,255,1)"],
    [0.3, "rgba(255,255,255,0.85)"],
    [1, "rgba(255,255,255,0)"],
  ]);
  const dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute("position", new THREE.Float32BufferAttribute(leds.flat(), 3));
  const dotMat = new THREE.PointsMaterial(additive(dotTex, { size: 0.016 }));
  group.add(new THREE.Points(dotGeo, dotMat));

  /* ---------- 內部光、光暈、臉部補光、地面 ---------- */
  const gc = new THREE.Vector3(cfg.glowCenter[0], cfg.glowCenter[1] * HEIGHT, cfg.glowCenter[2]);
  const inner = new THREE.PointLight(0xffffff, 1, 1.6, 2);
  inner.position.copy(gc).add(new THREE.Vector3(0, 0, -0.12));
  group.add(inner);
  // 從正面柔和打亮臉（照片中臉是被打光的）
  const faceLight = new THREE.SpotLight(0xfff6ee, 4, 3, 0.32, 0.8, 1.5);
  faceLight.position.set(0.15, 0.9 * HEIGHT, 1.4);
  faceLight.target.position.set(0, 0.87 * HEIGHT, 0);
  group.add(faceLight, faceLight.target);

  const halo = new THREE.Sprite(
    new THREE.SpriteMaterial(
      additive(
        radialSprite([
          [0, "rgba(255,255,255,0.8)"],
          [0.25, "rgba(255,255,255,0.25)"],
          [0.6, "rgba(255,255,255,0.05)"],
          [1, "rgba(255,255,255,0)"],
        ]),
      ),
    ),
  );
  // 光暈放在蛹的內部中心：從背面看不會蓋在殼上（之前放在背後，背面視角會整片洗白）
  halo.position.set(0, gc.y, -0.02);
  group.add(halo);
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(1.6, 64),
    new THREE.MeshBasicMaterial(
      additive(
        radialSprite([
          [0, "rgba(255,255,255,0.5)"],
          [0.3, "rgba(255,255,255,0.1)"],
          [1, "rgba(255,255,255,0)"],
        ]),
      ),
    ),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0.002;
  group.add(floor);

  /* ---------- 絲線外繭 ---------- */
  const THREADS = 90,
    STEPS = 90;
  const base = [],
    dir = [],
    delay = [];
  const rs = rng(5);
  for (let t = 0; t < THREADS; t++) {
    const a0 = rs() * Math.PI * 2;
    const turns = (rs() < 0.5 ? -1 : 1) * (2 + rs() * 5);
    const y0 = 0.02 + rs() * 0.25,
      y1 = HEIGHT - 0.05 + rs() * 0.12;
    const d0 = rs() * 0.45;
    const pts = [];
    for (let s2 = 0; s2 < STEPS; s2++) {
      const k = s2 / (STEPS - 1);
      const y = y0 + (y1 - y0) * k;
      const a = a0 + turns * Math.PI * 2 * k;
      const top = clamp01((y - (HEIGHT - 0.15)) / 0.2);
      const rr = (radiusAt(Math.min(y, HEIGHT)) + 0.04 + rs() * 0.015) * (1 - top * 0.7);
      pts.push([rr * Math.sin(a), y, rr * Math.cos(a)]);
    }
    for (let s2 = 0; s2 < STEPS - 1; s2++)
      for (const p of [pts[s2], pts[s2 + 1]]) {
        base.push(...p);
        const len = Math.hypot(p[0], p[2]) || 1,
          f = 0.5 + rs() * 0.9;
        dir.push((p[0] / len) * f, 0.25 + rs() * 0.5, (p[2] / len) * f);
        delay.push(d0 + (s2 / STEPS) * 0.15);
      }
  }
  const silkBase = new Float32Array(base),
    silkDir = new Float32Array(dir),
    silkDelay = new Float32Array(delay);
  const silkGeo = new THREE.BufferGeometry();
  silkGeo.setAttribute("position", new THREE.BufferAttribute(silkBase.slice(), 3));
  silkGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(silkBase.length), 3));
  const silk = new THREE.LineSegments(silkGeo, new THREE.LineBasicMaterial(additive(null, { vertexColors: true })));
  group.add(silk);

  /* ---------- 光塵 ---------- */
  const DUST = 380;
  const dustPos = new Float32Array(DUST * 3);
  const rp = rng(23);
  const seeds = Array.from({ length: DUST }, () => ({
    a: rp() * Math.PI * 2,
    r: 0.2 + Math.pow(rp(), 0.6) * 1.1,
    y: rp() * 2.3,
    sp: 0.03 + rp() * 0.1,
    w: rp() * 6,
  }));
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  const dustMat = new THREE.PointsMaterial(additive(dotTex, { size: 0.012 }));
  group.add(new THREE.Points(dustGeo, dustMat));

  // 程序化的頭與雙臂（取代 AI 模型模糊的臉，並能做出抓住蛹的動作）
  const figure = cfg.figure ? createFigure(cfg.figure) : null;
  if (figure) group.add(figure.group);

  // 腰間束結：兩片前襟的末端從左上、右上下來，在腰前互相扭轉成結（照片 358/361/339）
  if (cfg.knot) {
    const knotMat = new THREE.MeshStandardMaterial({ color: 0xf1f1ef, roughness: 0.55, metalness: 0 });
    knotMat.emissive = new THREE.Color(0x1c1c1c);
    const knot = new THREE.Group();
    const [kx, ky, kz] = cfg.knot.pos;
    const jr = rng(31);
    for (const side of [-1, 1]) {
      // 布條：從側上方下來 → 在中央扭一圈（寬、矮、不規則）→ 收進裙口
      const pts = [
        new THREE.Vector3(side * 0.12, 0.08, -0.02),
        new THREE.Vector3(side * 0.08, 0.045, 0.005),
      ];
      for (let i = 0; i <= 8; i++) {
        const t = i / 8;
        const a = t * Math.PI * 1.15 + (side < 0 ? 0.2 : Math.PI + 0.2);
        const r = 0.055 * (1 - 0.3 * t);
        pts.push(new THREE.Vector3(
          Math.cos(a) * r + (jr() - 0.5) * 0.012,
          0.02 - 0.055 * t + (jr() - 0.5) * 0.01,
          Math.sin(a) * r * 0.45 + 0.01,
        ));
      }
      pts.push(new THREE.Vector3(-side * 0.03, -0.06, -0.02));
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 120, 0.026, 16);
      const tube = new THREE.Mesh(geo, knotMat);
      tube.scale.set(1, 0.7, 0.55); // 扁平的布條
      knot.add(tube);
    }
    knot.position.set(kx, ky, kz);
    knot.scale.setScalar(cfg.knot.scale ?? 1);
    group.add(knot);
  }

  const state = { silk: 1, reveal: 1, burst: 0.3, grip: 0 };
  let lastGrip = -1;
  let lastSilk = -1,
    lastLvl = -1;
  function updateSilk(level) {
    if (Math.abs(lastSilk - state.silk) < 1e-4 && Math.abs(lastLvl - level) < 0.02) return;
    lastSilk = state.silk;
    lastLvl = level;
    const pos = silkGeo.attributes.position.array,
      col = silkGeo.attributes.color.array;
    for (let i = 0; i < silkDelay.length; i++) {
      const l = clamp01((state.silk - silkDelay[i]) / 0.5);
      const e = l * l * (3 - 2 * l);
      const k = i * 3;
      pos[k] = silkBase[k] + silkDir[k] * e * 1.4;
      pos[k + 1] = silkBase[k + 1] + silkDir[k + 1] * e * 1.4;
      pos[k + 2] = silkBase[k + 2] + silkDir[k + 2] * e * 1.4;
      col[k] = col[k + 1] = col[k + 2] = (1 - e) * (0.16 + 0.26 * Math.min(level, 1.3));
    }
    silkGeo.attributes.position.needsUpdate = true;
    silkGeo.attributes.color.needsUpdate = true;
    silk.visible = state.silk < 0.999;
  }

  return {
    group,
    model,
    state,
    set(next) {
      Object.assign(state, next);
    },
    update(time, level) {
      uniforms.uLevel.value = level;
      uniforms.uReveal.value = state.reveal;
      if (figure) {
        figure.group.visible = state.reveal > 0.55;
        // 雙手抓住之後才開始撥開
        uniforms.uOpen.value = THREE.MathUtils.smoothstep(state.grip, 0.75, 1.0);
        if (Math.abs(lastGrip - state.grip) > 1e-4) {
          figure.update(state.grip);
          lastGrip = state.grip;
        }
      }
      inner.intensity = 0.1 + level * 0.8;
      faceLight.intensity = (cfg.faceLight ?? 4) * state.reveal;
      dotMat.opacity = Math.min(1, 0.05 + level * 0.95);
      dotMat.size = 0.012 + level * 0.01;
      halo.material.opacity = 0.1 + level * 0.45;
      halo.scale.setScalar(1.6 + level * 0.7);
      floor.material.opacity = 0.15 + level * 0.3;
      updateSilk(level);
      const spread = 0.6 + 0.6 * state.burst;
      for (let i = 0; i < DUST; i++) {
        const d = seeds[i];
        const a = d.a + time * 0.05 + Math.sin(time * 0.3 + d.w) * 0.2;
        dustPos[i * 3] = Math.sin(a) * d.r * spread;
        dustPos[i * 3 + 1] = (d.y + time * d.sp) % 2.3;
        dustPos[i * 3 + 2] = Math.cos(a) * d.r * spread;
      }
      dustGeo.attributes.position.needsUpdate = true;
      dustMat.opacity = Math.max(state.burst, 0.15) * (0.35 + 0.65 * Math.min(level, 1.2));
    },
  };
}
