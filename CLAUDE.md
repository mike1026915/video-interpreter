# 影片口譯側欄 — 給 Claude Code 的專案說明

回覆請用台灣繁體中文。

## 這是什麼
Chrome MV3 擴充功能：擷取目前分頁的影片聲音 → 本機語音辨識（Whisper / Cohere Transcribe 走 transformers.js；Qwen3-ASR 走 onnxruntime-web）→ 翻成繁體中文（Chrome 內建 Translator API，或本機 HY-MT1.5）→ 一段段顯示在側邊欄。
需求是「免費、不用太即時、不用疊字幕」，所以全部在本機跑，沒有任何雲端 API。主要用途是看日文影片。

## 架構
- `background.js`：點工具列圖示 → `chrome.sidePanel.open`。這次點擊同時給了 tabCapture 對該分頁的權限（activeTab）。
- `sidepanel.html/js/css`：主程式全部在側欄裡（刻意不用 offscreen document）。
  - `chrome.tabCapture.getMediaStreamId({targetTabId})` → `getUserMedia({chromeMediaSource:'tab'})`
  - 擷取後分頁會被靜音，所以 `source.connect(ctx.destination)` 把聲音接回喇叭。
  - `capture-worklet.js`（AudioWorklet）把音訊混成單聲道、每 4096 樣本送回主執行緒。
  - `makeSegmenter`：降取樣到 16kHz，用 100ms 音框的 RMS 判斷停頓切段（3～15 秒）。參數在檔案開頭。判斷之前先經過自動增益（`shared.js` 的 `makeAgc`：最近 10 秒音框音量的 95 百分位拉到 0.1，最多 16 倍，降得快升得慢、峰值限制 0.99），所以很小聲的影片不會整段被當成靜音。音量沒到門檻（底噪 ×2）、但超過底噪 ×1.25 而且有人聲的音高（`shared.js` 的 `periodicity`：70～400Hz 的正規化自相關 ≥ 0.6），也算有人講話，抓得到跟背景音差不多大的小聲說話；背景本身有音高（音樂，平均週期性 ≥ 0.45）時不補判，否則會把 BGM 當成講話。有聲音的部分只有 0.2～0.3 秒時，至少一個音框有音高才送去辨識（「え？」、被笑聲隔開的句尾），沒有就略過（碰撞聲）；只有 0.1 秒時要音高 ≥ 0.8 才送。太短、連提示都不留的聲音不會重設空白期間的最大音量（`gapPeak`）。門檻上限 0.03，但底噪本身超過上限時改成底噪 ×1.4（`HIGH_FLOOR_MULT`），否則背景很大聲時所有聲音都算講話、只能 8 秒硬切。
  - 段落排進 queue，依序送 worker 辨識 → 過濾 Whisper 幻覺句 → 翻譯。
  - 自動偵測語言時，Qwen3-ASR 會把自己判斷的語言（ISO 代碼）跟著結果回傳，翻譯就直接用，不再跑 LanguageDetector。
  - 依模型選 worker：模型 ID 含 `qwen3-asr` → `qwen3-asr-worker.js`，其他 → `asr-worker.js`。送給 worker 的語言是 ISO 代碼（`ja`），自動偵測時為 null。
  - 沒選過模型時：有 WebGPU 預設 large-v3-turbo，沒有就 small。
  - 時間顯示的是影片時間：切段當下用 `chrome.scripting.executeScript`（靠點圖示時的 activeTab 授權，manifest 有 `scripting`）讀分頁裡正在播放、畫面最大的 `<video>` 的 currentTime，扣掉這段的長度換算成段落開頭（`shared.js` 的 `videoStartFrom`，有算倍速）。讀不到（沒有 video、影片在跨網域 iframe 裡）就退回從開始口譯起算，狀態列會註明。
  - 前後段之間比較影片時間和擷取時間（`classifyGap`）：影片時間倒退或走得比實際播放多 → 「↪ 影片跳到」（前轉／後轉，同時送出等句尾的原文、清掉 HY-MT 上下文）；連續播放中空了 2 秒以上 → 「⋯ 沒有辨識到聲音」，並附上空白期間最大音量是門檻的幾 %（≥50% 提示可能有小聲說話被當成靜音）。暫停不會讓影片時間前進，所以不會被誤判。
  - 翻譯引擎選 HY-MT1.5 時另開 `mt-worker.js`，帶前 2 句原文當上下文；載入失敗就改用 Chrome 內建翻譯。
  - HY-MT1.5（1.8B）偶爾會把上下文也翻進輸出：`shared.js` 的 `dropEchoedContext` 把跟前 2 句譯文幾乎相同的句子刪掉（bigram Dice ≥ 0.7 或互相包含，4 字以下不刪）。它也會把提示詞本身輸出（「參考上述的信息，將下面的文本翻譯成中文」），或把上下文換句話說再翻一次（字面不同，比對不出來）：`suspiciousContextOutput` 判斷譯文含提示詞字眼、或句數比原文多 2 句以上時，這句不帶上下文重翻，最後再用 `dropPromptEcho` 刪掉殘留的提示詞句子。偶爾也會翻成韓文（「そうか。」→「그렇군요。」）：原文沒有韓文時，譯文出現韓文也算可疑、會重翻，殘留的韓文句子用 `dropHangul` 刪掉。
  - Translator / LanguageDetector 下載模型需要使用者手勢，所以在「開始」按鈕的 click 裡先觸發；自動偵測語言時若需要下載，會顯示一個按鈕讓使用者點。
- `asr-worker.js`：module worker，跑 transformers.js 的 `automatic-speech-recognition` pipeline（Whisper、Cohere Transcribe）。有 WebGPU 用 webgpu，依顯示卡有沒有 `shader-f16` 選檔案（turbo：fp16 或 q4 encoder + q4 decoder；Cohere：q4f16 或 q4），沒有 WebGPU 時 Whisper 退回 wasm q8，Cohere 直接報錯。模型從 Hugging Face 下載後由瀏覽器快取。
- `qwen3-asr-worker.js` + `qwen3-asr-core.js`：Qwen3-ASR（`jiangzhuo9357/Qwen3-ASR-1.7B-ONNX`，layout v2，介面寫在模型卡和 `prompt_config.json`）。transformers.js 不支援，所以自己做 log-mel、prompt、嵌入（int8 表 × scale）、encoder → decoder_init → decoder_step 貪婪解碼、byte-level BPE 解碼。KV cache 用 `preferredOutputLocation: 'gpu-buffer'` 留在 GPU。只支援 WebGPU。模型檔自己下載、存在 Cache Storage `qwen3-asr-models`。
- `mt-worker.js`：HY-MT1.5-1.8B（`onnx-community/HY-MT1.5-1.8B-ONNX`），transformers.js text-generation，只支援 WebGPU。提示詞用模型卡的中文模板（目標「繁体中文」）。
- `lib/`：transformers.js 4.3.1 的 `transformers.min.js`（內含 onnxruntime-web）、onnxruntime-web 1.31.0-dev 的 `ort.webgpu.min.mjs`（給 Qwen3-ASR），兩者共用 `ort-wasm-simd-threaded.asyncify.{mjs,wasm}`（同一版 ORT）。transformers.js 4.x 預設會去 CDN 抓 wasm，MV3 不能載入遠端程式碼，所以每個 worker 都把 `wasmPaths` 設成指向本地 `lib/` 的 `{ mjs, wasm }`。CSP 有 `'wasm-unsafe-eval'`。transformers.js 的 worker 一定要設 `env.useWasmCache = false`：開著的話它會把 ORT 的 .mjs 轉成 `blob:` 網址再 import，被 CSP 擋掉，錯誤是「no available backend found」（在 Chromium 實測重現過）。升級時兩個套件的 ORT 版本要一致；wasm 檔名只寫在 `shared.js` 的 `ORT_WASM_PATHS`。`transformers.min.js` 有一處手動修改：`"Mistral3ForConditionalGeneration"` 拆成 `"Mistral3For"+"ConditionalGeneration"`（旁邊有註解），因為 GitHub 密鑰掃描會把原字串誤判成 Mistral API Key 而擋下 push；升級時要再改一次。
- `shared.js`：側欄和 worker 共用的 `ORT_WASM_PATHS`、`gpuInfo()`（WebGPU 與 shader-f16 偵測）、`isQwen3Asr` / `isCohere`。
- `tests/`：`node --test tests/*.test.mjs`。Qwen3-ASR 核心邏輯的單元測試，加上用假 ONNX 模型（`tests/fixtures/mock-qwen3/`，介面同真模型）在 Node 的 ORT wasm 上跑完整解碼迴圈。

## 狀態
- 切段邏輯用合成音訊在 Node 測過。
- Qwen3-ASR worker 用假模型在 headless Chromium（SwiftShader WebGPU、q4 路徑）跑通過；**真模型沒在這個開發環境跑過**（連不到 huggingface.co），q4f16 路徑只在 Node 用假模型測過。
- Cohere Transcribe、HY-MT1.5、transformers.js 4.3.1 下的 Whisper：確認能在擴充功能 CSP 下載入、走到下載模型那一步，真模型未實測。

## 已知風險 / 可能要先處理的
- 側欄按鈕呼叫 `getMediaStreamId` 是否被視為已 invoke（依賴前一次點圖示的 activeTab）——若失敗，改成在 `action.onClicked` 裡取 streamId，或改用 offscreen document 架構。
- transformers.js 在 extension worker 裡載入 jsep wasm / WebGPU 是否正常。
- Whisper 對日文的斷句與幻覺；切段參數可能要依實際影片調整。
- 量化模型遇到模糊音訊會卡在重複迴圈（例如「もう、もう、…」）：Qwen3-ASR 解碼時偵測到結尾同一段重複 4 次就停（`loopCut`），Whisper 不論 beam 或貪婪都帶 `no_repeat_ngram_size`，側欄再把連續重複 4 次以上的片段收斂成 2 次（`shared.js` 的 `collapseRepeats`）。

## 可能的改進方向
- Qwen3-ASR 的日文微調版（neosophie/Qwen3-ASR-1.7B-JA、jaykwok 的 Galgame 版）沒有現成的 layout v2 ONNX，要自己用 Sokuji 公開的匯出流程轉。
- Chrome 的 Prompt API（Gemini Nano）文件列的輸出語言沒有中文，不適合拿來翻繁中。
- 只有 CPU 的電腦：可試 Moonshine 日文版（`wmoto-ai/moonshine-tiny-ja-ONNX`）或 Chrome 內建離線語音辨識（Web Speech API 的 `processLocally` + `start(audioTrack)`）。
- 逐字稿存到 chrome.storage，關掉側欄不會消失。

## 開發方式
`chrome://extensions` → 開發人員模式 → 載入未封裝項目 → 選這個資料夾。改完按擴充功能卡片上的重新整理。
