// 程序化的人體部分：時裝人台風格的女性頭部（沒有眼睛嘴巴，只有鼻子；任何角度都像一張臉）、黑髮、頸部、
// 以及會隨滑動「慢慢抬起、抓住蛹的開口、準備破蛹」的雙臂與雙手。
// 單位：公尺；正面朝 +Z。位置依 AI 模型實測（見 model-config.js 的 figure 設定）。
import * as THREE from "three";

const g = (x, s) => Math.exp(-(x * x) / (s * s));

/* ---------- 頭部雕塑 ---------- */
function sculptHead() {
  const geo = new THREE.SphereGeometry(1, 120, 90);
  const pos = geo.attributes.position;
  const col = new Float32Array(pos.count * 3);
  const skin = new THREE.Color(0xeeeae7); // 人台白
  const c = new THREE.Color();
  const A = 0.072,
    B = 0.116,
    C = 0.094;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i),
      y = pos.getY(i),
      z = pos.getZ(i);
    let px = x * A,
      py = y * B,
      pz = z * C;
    const front = Math.max(z, 0);
    // 下顎往下收窄成 V 臉，後腦下方收進頸部
    if (y < -0.05) {
      const t = (-y - 0.05) / 0.95;
      px *= 1 - 0.42 * Math.pow(t, 1.2);
      pz *= z > 0 ? 1 - 0.12 * t : 1 - 0.35 * t;
    }
    // 額頭略飽滿、頭頂後方圓
    if (y > 0.2 && z > 0) pz += 0.004 * front * y;
    // 眉骨到鼻樑的柔和隆起
    pz += 0.002 * g(Math.abs(x) - 0.28, 0.22) * g(y - 0.16, 0.08) * front;
    // 鼻樑與鼻尖
    const ridge = g(x, 0.07) * (y < 0.12 && y > -0.32 ? 1 : 0) * Math.min(1, (0.12 - y) / 0.4);
    pz += (0.011 * ridge + 0.009 * g(x, 0.1) * g(y + 0.3, 0.07)) * front;
    for (const sx of [-1, 1]) pz += 0.003 * g(x - sx * 0.12, 0.06) * g(y + 0.33, 0.05) * front;
    // 下巴
    pz += 0.005 * g(x, 0.22) * g(y + 0.8, 0.1) * front;
    // 顴骨
    for (const sx of [-1, 1]) {
      const k = g(x - sx * 0.55, 0.2) * g(y + 0.12, 0.15) * front;
      px += sx * 0.003 * k;
      pz += 0.002 * k;
    }
    pos.setXYZ(i, px, py, pz);

    // 單一白色（人台質感）
    c.copy(skin);
    col.set([c.r, c.g, c.b], i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();
  return geo;
}

/* ---------- 手（局部座標：+Y 沿著手指方向，+Z 為掌心方向） ---------- */
function makeHand(mat) {
  const hand = new THREE.Group();
  const palm = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mat);
  palm.scale.set(0.038, 0.048, 0.016);
  palm.position.y = 0.045;
  hand.add(palm);
  const fingers = [];
  const seg = (len, r) => {
    const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 4, 10), mat);
    m.position.y = len / 2;
    const pivot = new THREE.Group();
    pivot.add(m);
    return pivot;
  };
  const spec = [
    [-0.024, 0.04, 0.0072],
    [-0.008, 0.046, 0.0078],
    [0.008, 0.044, 0.0076],
    [0.023, 0.036, 0.0066],
  ];
  for (const [fx, len, r] of spec) {
    const p1 = seg(len * 0.45, r);
    p1.position.set(fx, 0.088, 0);
    const p2 = seg(len * 0.33, r * 0.92);
    p2.position.y = len * 0.45 + r * 0.6;
    const p3 = seg(len * 0.25, r * 0.85);
    p3.position.y = len * 0.33 + r * 0.6;
    p2.add(p3);
    p1.add(p2);
    hand.add(p1);
    fingers.push([p1, p2, p3]);
  }
  const thumb1 = seg(0.03, 0.0085);
  thumb1.position.set(0.03, 0.03, 0.006);
  const thumb2 = seg(0.026, 0.0078);
  thumb2.position.y = 0.034;
  thumb1.add(thumb2);
  hand.add(thumb1);
  hand.userData = { fingers, thumb: [thumb1, thumb2] };
  return hand;
}
function setGrip(hand, k) {
  // k：0 放鬆、1 握住。手指一根一根依序彎曲（食指 → 中指 → 無名指 → 小指），拇指最後扣住
  const { fingers, thumb } = hand.userData;
  const ss = THREE.MathUtils.smoothstep;
  const order = [3, 2, 1, 0]; // spec 由小指(0)排到食指(3)：食指先
  fingers.forEach(([a, b, c], i) => {
    const start = order.indexOf(i) * 0.14;
    const f = ss(k, start, start + 0.42);
    a.rotation.x = 0.15 + f * 1.05;
    b.rotation.x = 0.1 + f * 1.25;
    c.rotation.x = 0.08 + f * 0.85;
  });
  const t = ss(k, 0.6, 1.0);
  thumb[0].rotation.set(0.2 + t * 0.7, 0, -0.75 + t * 0.45);
  thumb[1].rotation.x = 0.1 + t * 0.8;
}

/* ---------- 兩點之間的肢段 ---------- */
const UP = new THREE.Vector3(0, 1, 0);
function limb(mat, r0, r1) {
  const geo = new THREE.CylinderGeometry(r1, r0, 1, 20, 1, true);
  geo.translate(0, 0.5, 0);
  const m = new THREE.Mesh(geo, mat);
  const j = new THREE.Mesh(new THREE.SphereGeometry(r0, 16, 12), mat);
  const grp = new THREE.Group();
  grp.add(m, j);
  grp.userData.place = (a, b) => {
    const d = new THREE.Vector3().subVectors(b, a);
    const len = d.length();
    m.position.copy(a);
    m.scale.set(1, len, 1);
    m.quaternion.setFromUnitVectors(UP, d.normalize());
    j.position.copy(a);
  };
  return grp;
}

/** 兩段式 IK：肩 S、腕 T，回傳肘位置 */
function solveElbow(S, T, L1, L2, pole) {
  const d = new THREE.Vector3().subVectors(T, S);
  let dist = d.length();
  dist = Math.min(dist, L1 + L2 - 1e-3);
  const dir = d.normalize();
  const a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(0, L1 * L1 - a * a));
  const p = pole.clone().sub(dir.clone().multiplyScalar(pole.dot(dir))).normalize();
  return S.clone().addScaledVector(dir, a).addScaledVector(p, h);
}

const lerp3 = (a, b, t) => new THREE.Vector3().lerpVectors(new THREE.Vector3(...a), new THREE.Vector3(...b), t);
const ease = (t) => t * t * (3 - 2 * t);
function path(keys, t) {
  // keys: [[t, [x,y,z]], ...]
  for (let i = 0; i < keys.length - 1; i++) {
    const [t0, a] = keys[i],
      [t1, b] = keys[i + 1];
    if (t <= t1) return lerp3(a, b, ease(Math.max(0, (t - t0) / (t1 - t0))));
  }
  return new THREE.Vector3(...keys[keys.length - 1][1]);
}

export function createFigure(cfg) {
  const group = new THREE.Group();
  const skinMat = new THREE.MeshStandardMaterial({ color: 0xeeeae7, roughness: 0.48, metalness: 0, side: THREE.DoubleSide });
  skinMat.emissive = new THREE.Color(0x262524); // 帽沿下也不會黑掉
  const skinColor = new THREE.Color(cfg.skin ?? "#d4b8aa");
  skinMat.color.copy(skinColor);
  skinMat.emissive = skinColor.clone().multiplyScalar(0.18);
  const headMat = new THREE.MeshStandardMaterial({ color: skinColor, roughness: 0.5 });
  headMat.emissive = skinColor.clone().multiplyScalar(0.18);
  if (cfg.face) {
    // 照片五官從正面投影到雕塑頭上（頭的局部座標），邊緣柔和融進膚色
    const faceTex = new THREE.TextureLoader().load(cfg.face.url);
    faceTex.colorSpace = THREE.SRGBColorSpace;
    const [x0, y0, x1, y1] = cfg.face.rect;
    headMat.onBeforeCompile = (sh) => {
      sh.uniforms.uFace = { value: faceTex };
      sh.uniforms.uRect = { value: new THREE.Vector4(x0, y0, x1, y1) };
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", `#include <common>
varying vec3 vLocal;
varying vec3 vLNrm;`)
        .replace("#include <begin_vertex>", `#include <begin_vertex>
vLocal = position;
vLNrm = normal;`);
      sh.fragmentShader = sh.fragmentShader
        .replace("#include <common>", `#include <common>
varying vec3 vLocal;
varying vec3 vLNrm;
uniform sampler2D uFace;
uniform vec4 uRect;`)
        .replace(
          "#include <map_fragment>",
          `#include <map_fragment>
vec2 fuv = (vLocal.xy - uRect.xy) / (uRect.zw - uRect.xy);
if (fuv.x > 0.0 && fuv.x < 1.0 && fuv.y > 0.0 && fuv.y < 1.0 && vLocal.z > 0.0) {
  vec4 f = texture2D(uFace, fuv);
  float w = f.a * smoothstep(0.15, 0.55, normalize(vLNrm).z);
  diffuseColor.rgb = mix(diffuseColor.rgb, f.rgb * 1.08, w);
}`,
        );
    };
  }

  /* 頭、髮、頸 */
  const head = new THREE.Group();
  const face = new THREE.Mesh(sculptHead(), headMat);
  face.geometry.deleteAttribute("color");
  head.add(face);
  // 人台頭：無五官（只有鼻子）、無頭髮，與身體同膚色
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.043, 0.16, 24), skinMat);
  neck.position.set(0, -0.14, -0.012);
  head.add(neck);
  head.position.set(...cfg.head.pos);
  head.rotation.set(cfg.head.tilt ?? 0, cfg.head.turn ?? 0, cfg.head.roll ?? 0);
  head.scale.setScalar(cfg.head.scale ?? 1);
  group.add(head);

  /* 雙臂 */
  const arms = [-1, 1].map((side) => {
    const a = cfg.arms;
    const upper = limb(skinMat, 0.034, 0.028);
    const fore = limb(skinMat, 0.028, 0.021);
    const hand = makeHand(skinMat);
    group.add(upper, fore, hand);
    return { side, upper, fore, hand, S: new THREE.Vector3(side * a.shoulderX, a.shoulderY, a.shoulderZ) };
  });

  const tmpM = new THREE.Matrix4();
  function update(p) {
    // p：0 雙手在腰間 → 1 舉起抓住蛹的開口、往外撐
    const a = cfg.arms;
    for (const arm of arms) {
      const key = arm.side < 0 ? a.left : a.right;
      const T = path(key, p);
      // 肘往下、往前收在蛹裡（不往外張開）
      const E = solveElbow(arm.S, T, a.upperLen, a.foreLen, new THREE.Vector3(arm.side * 0.05, -1, 0.65));
      arm.upper.userData.place(arm.S, E);
      arm.upper.visible = false; // 上臂在蛹裡面，不露出
      arm.fore.userData.place(new THREE.Vector3().lerpVectors(E, T, 0.5), T); // 只露出前臂靠手的一半（斷口藏在前襟後面）
      const show = p > 0.02;
      arm.fore.visible = show;
      arm.hand.visible = show;
      // 手：反手抓殼——從開口內側伸出，掌心朝外（朝向殼），手背對著臉，手指勾住殼的邊緣往外撥
      const yAxis = new THREE.Vector3().subVectors(T, E).normalize();
      const k = THREE.MathUtils.smoothstep(p, 0.4, 0.85);
      const want = new THREE.Vector3(arm.side * (0.4 + 0.6 * k), 0.0, 0.9 - 0.75 * k);
      const zAxis = want.sub(yAxis.clone().multiplyScalar(want.dot(yAxis))).normalize();
      const xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis);
      tmpM.makeBasis(xAxis, yAxis, zAxis);
      arm.hand.quaternion.setFromRotationMatrix(tmpM);
      arm.hand.position.copy(T);
      arm.hand.scale.set(arm.side * 0.85, 0.85, 0.85); // 左手鏡像、手掌大小
      setGrip(arm.hand, THREE.MathUtils.smoothstep(p, 0.5, 0.85));
    }
  }
  update(0);
  return { group, head, update };
}
