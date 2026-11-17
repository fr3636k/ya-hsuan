# 開發與提交規範

先閱讀 README.md 的「Git 提交規範」。

- 提交使用 Conventional Commits：`<type>(<scope>): <中文簡短說明>`。
- 訊息必須描述實際變更；禁止使用 `Update`、`pages build and deployment` 等泛用訊息。
- 提交前執行 `npm run check`，並在本機預覽確認手機與桌機畫面。
- 保留使用者未提交的修改，不擅自納入其他工作。
- 公開網站由 GitHub Actions（`.github/workflows/pages.yml`）在推上 `main` 後發布 `dist/`；每次發布的名稱沿用該次提交訊息，不另做泛用提交。
- `data/`、`ai-input/`、`models-raw/`、`qa/` 只留本機，不進版本庫。
