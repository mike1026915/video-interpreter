# 影片口譯側欄 — 給 Claude Code 的專案說明

回覆請用台灣繁體中文。

## 這是什麼
Chrome MV3 擴充功能：擷取目前分頁的影片聲音 → 本機 Whisper（transformers.js）辨識 → Chrome 內建 Translator API 翻成繁體中文（zh-Hant）→ 一段段顯示在側邊欄。
需求是「免費、不用太即時、不用疊字幕」，所以全部在本機跑，沒有任何雲端 API。主要用途是看日文影片。

## 架構
- `background.js`：點工具列圖示 → `chrome.sidePanel.open`。這次點擊同時給了 tabCapture 對該分頁的權限（activeTab）。
- `sidepanel.html/js/css`：主程式全部在側欄裡（刻意不用 offscreen document）。
  - `chrome.tabCapture.getMediaStreamId({targetTabId})` → `getUserMedia({chromeMediaSource:'tab'})`。
  - 擷取後分頁會被靜音，所以 `source.connect(ctx.destination)` 把聲音接回喇叭。
  - `capture-worklet.js`（AudioWorklet）把音訊混成單聲道、每 4096 樣本送回主執行緒。
  - `makeSegmenter`：降取樣到 16kHz（簡單平均，沒有抗混疊濾波），用 100ms 音框的 RMS 切段，3～8 秒一段。
    - 自適應靜音門檻：最近 10 秒最安靜的 10% 音框當底噪，門檻 = 底噪 × 2（0.003～0.03）。
    - 停頓 0.6 秒切段；超過 6 秒後停頓 0.35 秒就切；到 8 秒仍沒停頓，回頭在音量最低的音框切（`cutAtDip`），剩下的留給下一段。
    - 段落開頭多帶 0.3 秒；說話部分太小聲時放大（最多 8 倍）。
    - `status()` 回報「是否正在錄段落」與「安靜多久」，給整句翻譯判斷說話者是否停下。
  - 段落排進 queue，依序送 worker 辨識 → 過濾幻覺句 → 排入翻譯。被略過的段落（空結果、幻覺、太短、錯誤）在畫面留淡色提示。
  - 翻譯有兩種模式（「翻譯方式」下拉選單）：
    - 精準：等句尾標點，或說話者停下來（`checkGroup` 每 250ms 檢查），才把多段原文合併成一句翻譯。
    - 即時：每段立刻翻。
  - Translator / LanguageDetector 下載模型需要使用者手勢，所以在「開始」按鈕的 click 裡先觸發；自動偵測語言時若需要下載，會顯示一個按鈕讓使用者點。
  - 辨識 worker 可以重建（`spawnWorker`）：換模型、上次載入失敗、辨識時出過 WebGPU 錯誤、或 worker 回報 `restart` 時，都換一個全新的 worker。
- `asr-worker.js`：module worker，跑 transformers.js 的 `automatic-speech-recognition` pipeline。
  - 內建模型（`onnx-community/whisper-*`）：WebGPU 用 encoder fp32（large 用 fp16）＋ decoder q4；沒有 WebGPU 用 wasm q8。只用這一組，不嘗試其他量化。
  - 自訂模型：找不到檔案時才換下一組量化；其他錯誤直接停止，避免一套接一套地下載。
  - 每組設定載入後用 1 秒靜音實際跑一次（兼預熱）。只有 WebGPU 執行階段錯誤算失敗，此時回報 `restart`，由側欄在新 worker 裡以 `forceWasm` 用 CPU 載入。
  - 解碼：beam search 4、`max_new_tokens` 400、`no_repeat_ngram_size` 6。beam search 失敗會退回貪婪解碼；模型不接受 `language`/`task` 時改成不傳。
- `lib/`：transformers.js 3.8.1 的 `transformers.min.js` 和 ONNX Runtime 的 `ort-wasm-simd-threaded.jsep.{mjs,wasm}`。MV3 不能載入遠端程式碼，所以 `env.backends.onnx.wasm.wasmPaths` 指向本地 `lib/`。CSP 有 `'wasm-unsafe-eval'`。
- `manifest.json` 設了 COOP/COEP（跨來源隔離），讓 CPU 模式能開多執行緒。

## 狀態
**已在真實 Chrome 上確認可用：**
- 點圖示開側欄後，在側欄按「開始」呼叫 `getMediaStreamId` 可以正常擷取分頁聲音。
- transformers.js 在 extension worker 裡用 WebGPU 跑 whisper-small 與 whisper-large-v3-turbo。
- 加了 COOP/COEP 後，模型下載與載入仍正常。
- Chrome 內建翻譯 日文 → 繁中。
- 計時顯示、精準 / 即時切換、整句合併。
- 實測速度（large-v3-turbo、GPU）：3～5 秒的音訊辨識約 1.5～2 秒，翻譯 0.0～0.3 秒，沒有積壓。

**只用合成音訊或模擬在 Node 測過、還沒在實機確認效果：**
- 切段器：自適應門檻、低點切段、開頭留白、音量正規化。
- worker 的載入驗證、退回 CPU、重建 worker 流程（用模擬的 transformers.js 與 Worker）。
- 這些測試是臨時寫的腳本，**沒有放進 repo**。

**尚未確認：**
- beam search 在 WebGPU 上是否真的生效，還是已退回貪婪解碼（看 console 有沒有 `[asr] beam search 失敗`）。
- `willopcbeta/whisper-ja-760M-ONNX` 改用 CPU 後能不能跑、速度如何。
- 模型是否每次都從快取讀，還是被清掉後重新下載。
- CPU 模式的多執行緒是否真的開啟、有沒有變快。

## 已知問題
- **`willopcbeta/whisper-ja-760M-ONNX` 無法在 WebGPU 上執行**：`[WebGPU] Kernel "[Attention] Attention_0" failed. Error: Mask not supported`。
  - 之後同一個 worker 裡的任何 session（含 CPU、含其他模型）都回報同一個錯誤，所以才有重建 worker 的機制。這是從實機現象推出來的，沒有在 ONNX Runtime 原始碼層面確認。
  - 新版 transformers.js / ONNX Runtime 是否已支援，尚未確認。
- **含糊、收音差、有方言的音訊**：large-v3-turbo 常常少輸出字或漏整句，其餘部分是猜出來的。這是模型的極限，調參數幫助有限。
- 辨識進行中才出現 WebGPU 錯誤時，該次口譯會持續出錯，要按停止再開始才會換新 worker。
- 影片分頁重新整理或換頁後，擷取會中斷（activeTab 失效），要再點一次圖示、再按開始；重新開始時時間軸從 0 算起。
- 側欄關掉，逐字稿就消失。
- Chrome 內建翻譯只能單句翻，人名、地名、多義詞常翻錯（例：山梨県 → 山橋縣、撮れた → 服用）。

## 可能的改進方向
- 翻譯品質：自訂詞彙表、把前一句一起送翻譯再切出本句（要實測 Translator 會不會保留換行）、或改用 Prompt API（Gemini Nano）做帶上下文的翻譯。
- 辨識品質：找能在 WebGPU 上跑的日文微調模型（例如 kotoba-whisper 的 ONNX 版，尚未找到）、升級 transformers.js / ONNX Runtime、降噪或正確的重取樣（抗混疊濾波）。
- 辨識進行中出現 WebGPU 錯誤時自動換新 worker，不必手動停止再開始。
- 分頁重新整理後的明確提示，以及重新開始時保留逐字稿與時間軸。
- 顯示影片實際播放時間（content script 讀 `video.currentTime`）取代從開始計時。
- 逐字稿存到 chrome.storage，關掉側欄不會消失。
- `unlimitedStorage` 權限，降低大模型被瀏覽器從快取清掉的機率；狀態列區分「從快取載入」與「下載」。
- 把 Node 上的切段器與 worker 測試整理進 repo。

## 開發方式
`chrome://extensions` → 開發人員模式 → 載入未封裝項目 → 選這個資料夾。改完按擴充功能卡片上的重新整理。

除錯時看**側欄**的 DevTools（在側欄上按右鍵 →「檢查」），console 會有 `[asr]`、`[timing]`、`[skip]`、`[vad]`。影片分頁的 DevTools 裡看到的（例如 `DroppedFramesRule`）是網站播放器自己的 log，不是這個擴充功能印的。
