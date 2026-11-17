# 開發與提交規範

- 提交使用 Conventional Commits：`<type>(<scope>): <中文簡短說明>`。
- 訊息必須描述實際變更；禁止使用 `Update`、`pages build and deployment` 等泛用訊息。
- 提交前執行 `npm run check`，並在本機預覽確認手機與桌機畫面。
- 保留使用者未提交的修改，不擅自納入其他工作。
- 公開網站由 GitHub Actions（`.github/workflows/pages.yml`）在推上 `main` 後發布 `dist/`；每次發布的名稱沿用該次提交訊息，不另做泛用提交。
- `data/`、`models/`、`video-src/`、`qa/` 只留本機，不進版本庫。

## 提交類型

| 類型 | 用途 | 範例 |
| --- | --- | --- |
| `feat` | 新功能或新內容 | `feat(finale): 新增左手抓殼影片` |
| `fix` | 修正錯誤 | `fix(model): 修正眼皮白線` |
| `docs` | 文件更新 | `docs(readme): 改寫設計核心說明` |
| `build` | 建置、發布設定 | `build(deploy): 改用 GitHub Actions 發布` |
| `refactor` | 不改變功能的程式整理 | `refactor(viewer): 整理視角切換` |

主旨用現在式、精簡、結尾不加句號；一個 commit 只做一件事。
