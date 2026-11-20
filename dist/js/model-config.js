// 目前使用的模型與調整參數（換模型時只改這裡）
// 模型：V2 Tripo（models/v2-tripo/raw-tripo.glb），由 scripts/process-model.py --tripo 處理：
// 形狀、臉、皺褶完全保留，只把偏灰的白布提亮去污、做發光遮罩與 LED 位置。
// 高度相關數值皆為「相對身高」的比例 0~1。
// V1（TRELLIS）模型保存在本機 models/v1-trellis/，不再發布；目前為 V2（Tripo）。
// 載入：先顯示預覽模型（emerge-c-preview.glb，約 170 KB：網格 1.5%、貼圖 256），
// 完整模型（emerge-c.glb，約 6 MB）在背景下載後原地換上 → 畫面約 1～2 秒出現，最終品質不變
export const MODEL = {
  url: "./models/emerge-c.glb",
  cfg: {
    preview: "./models/emerge-c-preview.glb",
    leds: "./models/emerge-c.leds.json", // LED 光點位置
    glowMin: 0.58, // 蛹的發光範圍：從腰
    glowMax: 1.0, //            到頂
    glowStrength: 0.85, // 白光最亮時的布面發光強度
    wholeGlow: 0.06, // 平常整個人微微透光（閃的瞬間另外整個人一起亮）
    glowCenter: [0, 0.8, -0.12], // 內部光源位置：背後的圓弧
    backZ: -0.03, // 發光越靠背面越亮（公尺；Tripo 蛹的中軸約 z 0.02，整個背半部都要亮）
    useEmissiveMap: true, // 用處理時產生的發光遮罩（臉不發光）
    ribs: 0, // Tripo 模型本身就有骨架分片，不再疊畫條帶
    crinkle: 0, // 不疊加程序化壓皺紋，保留模型原本的皺褶
    bodyMask: 0, // 不用舊模型的身體範圍
    gauze: 1, // 紗的透光 0~1
    facingGlow: 1, // 燈籠：面板中央透光強、折邊暗 0~1
    cocoonShade: 0.3, // 蛹布外表壓暗比例（讓透光明顯）
    // 最後剝殼：開口兩側被撥開（公尺）
    openCenterX: 0.0,
    openPush: 0.05,
    faceBox: [0.075, 1.3, 1.72, 0.0], // 臉的範圍（公尺）：半寬、下緣、上緣、最小 z
    skinBand: [1.1, 1.18], // 露腰範圍（公尺）：網格背面顯示膚色
    faceLight: 6, // 打在臉上的柔光（帽兜陰影裡的臉才不會灰暗）
    face: null, // 臉用 Tripo 原本的臉（只加強平滑）；影片臉投影的工具保留在 scripts/face-decal.py
    knot: null,
  },
};
