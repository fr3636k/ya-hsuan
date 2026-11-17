import http from "node:http";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("./dist/", import.meta.url));
// 參考照片只在本機預覽時提供（/_ref/），不會放進 dist，也就不會被發布
const refDir = fileURLToPath(new URL("./data/", import.meta.url));
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".glb": "model/gltf-binary",
  ".wasm": "application/wasm",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
};
const port = Number(process.env.PORT) || 8788;

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    let relative = decodeURIComponent(url.pathname).replace(/^\/+/, "") || "index.html";
    let base = root;
    if (relative === "_ref/list.json") {
      const files = (await readdir(refDir)).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort();
      res.writeHead(200, { "Content-Type": types[".json"], "Cache-Control": "no-cache" });
      res.end(JSON.stringify(files));
      return;
    }
    if (relative.startsWith("_ref/")) {
      base = refDir;
      relative = relative.slice(5);
    }
    const file = path.resolve(base, relative);
    if (!file.startsWith(base)) {
      res.writeHead(403);
      res.end();
      return;
    }
    const bytes = await readFile(file);
    res.writeHead(200, {
      "Content-Type": types[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(bytes);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});
// 0.0.0.0：同一個 Wi-Fi 下的手機也能用電腦 IP 開啟預覽
server.listen(port, "0.0.0.0", () => {
  console.log(`3D preview: http://127.0.0.1:${port}`);
  console.log(`3D viewer : http://127.0.0.1:${port}/viewer.html`);
});
