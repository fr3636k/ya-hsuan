# 破蛹 EMERGE：光繭服裝 3D 展示

凱達格蘭族服裝設計作業。凱達格蘭族為母系社會，作品以「女性破蛹而出」為意象。
原作內部為粉紅／虹光 LED，網頁中全部改為**白光**，節奏為 **亮 → 暗 → 閃一下** 循環。

## 頁面

| 頁面 | 檔案 | 用途 |
|---|---|---|
| 主展示頁 | `dist/index.html` + `dist/story.js` | 仿 iPhone 產品頁，往下滑動時鏡頭轉動、絲線散開、光繭打開 |
| 3D 檢視子網頁 | `dist/viewer.html` + `dist/viewer.js` | 自由旋轉，調整開啟程度／前傾／絲線／燈光，用來確認模型 |

## 要改哪個檔案？

| 需求 | 檔案 |
|---|---|
| 光繭形狀、裙子輪廓、束結、人形、LED 光點、絲線 | `dist/garment.js` |
| 燈光節奏（亮／暗／閃的秒數） | `dist/garment.js` 的 `RHYTHM` 與 `rhythmAt()` |
| 每一章的鏡頭角度與開啟程度 | `dist/story.js` 的 `KEYS` |
| 章節文字 | `dist/index.html` |
| 顏色、字體、手機版面 | `dist/style.css` |

模型完全由程式（Three.js）產生，不需要 GLB 檔。Three.js 隨站放在 `dist/vendor/`，不依賴外部 CDN。

## 本機預覽

```powershell
.\start-preview.ps1      # 或 node serve.mjs
```

- 主展示頁：http://127.0.0.1:8788
- 3D 檢視：http://127.0.0.1:8788/viewer.html
- 手機（同一個 Wi-Fi）：用電腦 IP，例如 `http://192.168.x.x:8788`（啟動時會顯示）
- 截圖檢查用：`?at=4` 讓主展示頁固定在第 4 章（0 起算）。

`data/` 的參考照片只在本機預覽時，於 3D 檢視頁的「對照照片」區出現。照片不在 `dist/` 內，所以**不會被發布**。

## 部署

與超音波木盒專案相同：純靜態網站，發布目錄為 `dist`（`netlify.toml` 已設定）。

- 修改後需重新發布，公開網站不會自動讀取本機檔案。
- 發布前執行 `npm run check`。
- 可用的方式：Netlify（拖曳 `dist` 資料夾到 app.netlify.com/drop，或連結 GitHub 儲存庫）、GitHub Pages，或其他任何靜態網站託管。

## Git 提交規範

沿用 Conventional Commits：`<type>(<scope>): <中文簡短說明>`，例如 `feat(model): 調整光繭前傾角度`。


## Tripo 模型與最後一章的抓殼影片

- 模型：`models-raw/C-tripo.glb`（Tripo FBX 用 Blender 轉檔）→ `python scripts/process-model.py models-raw/C-tripo.glb <暫存>.glb --tripo`（白布提亮去污、臉保留原貌、發光遮罩、LED）→ Blender 執行 `scripts/draco-compress.py <暫存>.glb dist/models/emerge-c.glb`（Draco 壓縮完整網格，約 6 MB），再把 `<暫存>.leds.json` 複製成 `dist/models/emerge-c.leds.json`。
  Draco 解碼器在 `dist/vendor/draco/`（Three.js r169）。
- 抓殼影片（圖生影片，需外部 AI 服務）：
  1. 起始畫面：`models-raw/grip/keyframe-720x1280.png`（與網頁最後一章同構圖）。
  2. A 段提示詞 `models-raw/grip/prompt-segA.json`（左手抓左下、右手抓右上、頭往前探）；B 段 `prompt-segB.json`（用 A 段最後一格當起始畫面：一隻手往前伸直、像要爬出來）。
  3. `python scripts/grip-frames.py A.mp4 B.mp4` → 寫入 `dist/media/grip/`。沒有這個資料夾時，最後一章自動改用 3D 撥開效果。

## 公開網址與發布（GitHub Pages）

- 公開網址：https://fr3636k.github.io/ya-hsuan/
- 原始碼：https://github.com/fr3636k/ya-hsuan
- 發布方式：`dist/` 的內容推到 `gh-pages` 分支，GitHub Pages 直接提供該分支（約 1～2 分鐘生效）：
  `git subtree split --prefix dist -b gh-pages && git push -f origin gh-pages`（之後每次更新都重跑這兩個指令，再 `git branch -D gh-pages`）。
- 修改流程：改檔 → `npm run check` → 本機預覽（`node serve.mjs`）確認手機與桌機 → `git commit` → `git push origin main` → 上面的 gh-pages 指令。
- 不會上傳：`data/`（參考照片）、`ai-input/`、`models-raw/`（AI 原始檔、影片素材、舊模型）、`qa/`、Tripo FBX 原始檔。
