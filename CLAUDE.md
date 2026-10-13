# 影片口譯側欄 — 給 Claude Code 的專案說明

回覆請用台灣繁體中文。

## 這是什麼
Chrome MV3 擴充功能：擷取目前分頁的影片聲音 → 本機 Whisper（transformers.js）辨識 → Chrome 內建 Translator API 翻成繁體中文（zh-Hant）→ 一段段顯示在側邊欄。
需求是「免費、不用太即時、不用疊字幕」，所以全部在本機跑，沒有任何雲端 API。主要用途是看日文影片。

## 架構
- `background.js`：點工具列圖示 → `chrome.sidePanel.open`。這次點擊同時給了 tabCapture 對該分頁的權限（activeTab）。
- `sidepanel.html/js/css`：主程式全部在側欄裡（刻意不用 offscreen document）。
  - `chrome.tabCapture.getMediaStreamId({targetTabId})` → `getUserMedia({chromeMediaSource:'tab'})`
  - 擷取後分頁會被靜音，所以 `source.connect(ctx.destination)` 把聲音接回喇叭。
  - `capture-worklet.js`（AudioWorklet）把音訊混成單聲道、每 4096 樣本送回主執行緒。
  - `makeSegmenter`：降取樣到 16kHz，用 100ms 音框的 RMS 判斷停頓切段（3～15 秒）。參數在檔案開頭。
  - 段落排進 queue，依序送 worker 辨識 → 過濾 Whisper 幻覺句 → 翻譯。
  - Translator / LanguageDetector 下載模型需要使用者手勢，所以在「開始」按鈕的 click 裡先觸發；自動偵測語言時若需要下載，會顯示一個按鈕讓使用者點。
- `asr-worker.js`：module worker，跑 transformers.js 的 `automatic-speech-recognition` pipeline。有 WebGPU 用 webgpu（encoder fp32/fp16、decoder q4），沒有就退回 wasm q8。模型從 Hugging Face 下載後由瀏覽器快取。
- `lib/`：transformers.js 3.8.1 的 `transformers.min.js` 和 ONNX Runtime 的 `ort-wasm-simd-threaded.jsep.{mjs,wasm}`。MV3 不能載入遠端程式碼，所以 `env.backends.onnx.wasm.wasmPaths` 指向本地 `lib/`。CSP 有 `'wasm-unsafe-eval'`。

## 狀態
- 切段邏輯用合成音訊在 Node 測過。
- **整體還沒在真實 Chrome 上跑過**，第一步應該是載入測試、看 side panel 的 DevTools console。

## 已知風險 / 可能要先處理的
- 側欄按鈕呼叫 `getMediaStreamId` 是否被視為已 invoke（依賴前一次點圖示的 activeTab）——若失敗，改成在 `action.onClicked` 裡取 streamId，或改用 offscreen document 架構。
- transformers.js 在 extension worker 裡載入 jsep wasm / WebGPU 是否正常。
- Whisper 對日文的斷句與幻覺；切段參數可能要依實際影片調整。

## 可能的改進方向
- 把前後段原文一起送翻譯以保留上下文（日文主詞省略常翻錯）。
- 翻譯品質不足時，可考慮用 Chrome 的 Prompt API（Gemini Nano）做帶上下文的翻譯。
- 顯示影片實際播放時間（content script 讀 `video.currentTime`）取代從開始計時。
- 逐字稿存到 chrome.storage，關掉側欄不會消失。

## 開發方式
`chrome://extensions` → 開發人員模式 → 載入未封裝項目 → 選這個資料夾。改完按擴充功能卡片上的重新整理。
