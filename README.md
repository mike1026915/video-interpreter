# 影片口譯側欄（Chrome 擴充功能）

擷取目前分頁的影片聲音 → 本機語音辨識（Whisper / Qwen3-ASR / Cohere Transcribe）→ 翻成繁體中文（Chrome 內建翻譯或本機 HY-MT1.5）→ 顯示在側邊欄。
全部在你的電腦上跑，免費、不需要任何 API 金鑰，聲音不會上傳。

## 安裝
1. 解壓縮到一個固定的資料夾。
2. 打開 `chrome://extensions`，右上角開啟「開發人員模式」。
3. 按「載入未封裝項目」，選這個資料夾。
4. 建議把擴充功能釘選到工具列。
5. 之後改了程式，在擴充功能卡片上按重新整理即可。

需求：桌面版 Chrome 138 以上。有獨立或內建 GPU（支援 WebGPU）會快很多。

## 使用
1. 打開有影片的分頁，點工具列上的擴充功能圖示 → 右側出現側邊欄。
2. 選原文語言、辨識模型、翻譯引擎（原文語言預設為日文），按「開始口譯」。
3. 播放影片。大約每 3～8 秒會出現一段原文，隨後補上中文。
4. 「匯出」可以下載逐字稿 .txt。

第一次使用會下載模型（見下表），之後都有快取。

### 辨識模型怎麼選
| 模型 | 下載量 | 適合 |
|---|---|---|
| Whisper base | 約 150MB | 沒有 GPU，或速度跟不上時 |
| Whisper small | 約 300MB | 沒有 GPU 時的預設 |
| Whisper large-v3-turbo | 0.8～1.6GB | **有 WebGPU 時的預設**。日文比 small 準很多 |
| Qwen3-ASR 1.7B | 2.0～2.7GB | 有 WebGPU、想要日文最準時。需要約 3.5～5GB 顯示記憶體 |
| Cohere Transcribe | 1.5～2.1GB | 實驗性質。需要 WebGPU，不能自動偵測語言 |
| 自訂 | — | Hugging Face 上的 ONNX 模型 ID，見下 |

- 沒有選過模型時，有 WebGPU 會預設 large-v3-turbo，沒有就預設 small。選過之後會記住你的選擇。
- 下載量有範圍，是因為依顯示卡選檔案：支援 `shader-f16` 的顯卡（Windows、macOS 大多支援）用 fp16 版，比較小也比較快；不支援的（例如 Linux 上的 NVIDIA）用 q4 版。
- 日文準確度參考（CER，越低越好）：FLEURS 朗讀語音 turbo 6.6%、Qwen3-ASR 1.7B 5.4%；對話語音 turbo 0.184、Qwen3-ASR 1.7B 0.140。small 在 CommonVoice 日文約 15%，large-v3 約 8.5%。動畫、有背景音樂的影片請自己比較看看。

### Qwen3-ASR 1.7B
- 模型：[`jiangzhuo9357/Qwen3-ASR-1.7B-ONNX`](https://huggingface.co/jiangzhuo9357/Qwen3-ASR-1.7B-ONNX)（Apache-2.0，Sokuji 專案轉的瀏覽器版）。
- transformers.js 不支援 Qwen3-ASR，所以由 `qwen3-asr-worker.js` 直接用 onnxruntime-web 執行：encoder + decoder_init + decoder_step，KV cache 留在 GPU。
- 選了原文語言時，會把「language Japanese<asr_text>」直接放進 prompt（強制語言），避免量化模型偶爾跳過語言標記而輸出錯誤內容；「自動偵測」時由模型自己判斷，console 會印出 `[qwen3-asr] 偵測到語言`，翻譯也直接用這個結果，不再另外偵測。
- 只能用 WebGPU（CPU 太慢跟不上影片），沒有 WebGPU 會顯示錯誤，請改用 Whisper。
- 模型檔存在瀏覽器的 Cache Storage（`qwen3-asr-models`），第二次開始不會重新下載。
- 自訂模型欄也可以填其他同格式（layout v2，有 `prompt_config.json`）的 Qwen3-ASR ONNX，模型 ID 含 `qwen3-asr` 就會用這個 worker。

### Cohere Transcribe
- 模型：[`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)，2B，透過 transformers.js 4.x 的語音辨識 pipeline 執行。
- 沒有自動偵測語言，「原文語言」選「自動偵測」時無法開始。
- 公開數據裡日文不見得比 turbo 好（FLEURS 約 8.3%），放著給你用自己的影片比較。

### 自訂模型
選「自訂」後會出現輸入欄，填 `作者/模型名`（貼整個 huggingface.co 網址也可以），設定會被記住。
例如日文微調版：`willopcbeta/whisper-ja-760M-ONNX`（社群轉換，**未驗證過品質與相容性**）。
- 模型必須有 ONNX 權重，才能被 Transformers.js 載入；沒有的話會顯示載入失敗的錯誤。
- 自訂模型的檔案命名不一定和內建模型相同，所以**找不到某組量化的檔案時**，會依序換下一組（例如 encoder fp16 / decoder q4、fp16、q8）。其他錯誤不會再換，因為換了也一樣會壞，還會一套接一套地下載模型檔。為避免悄悄下載好幾 GB，不會嘗試純 fp32。
- 內建模型的檔案已知，只用固定的一組量化設定，絕不嘗試其他版本。
- 如果模型不接受 `language` / `task` 參數（例如單語模型），會自動改成不傳。
- 每組設定載入後會先用 1 秒靜音**實際跑一次**，能跑才算成功。WebGPU 在執行時才發現某個運算不支援的話（例如 `[WebGPU] Kernel "[Attention] Attention_0" failed. Error: Mask not supported`），會直接改用 CPU，不再嘗試其他量化（換量化也一樣會壞，而且每換一組就要重新下載一套模型檔）。狀態列會標示「WebGPU 無法執行這個模型，改用 CPU」。
- 改用 CPU 時會**關掉原本的辨識 worker、開一個全新的**再載入：WebGPU 推論失敗後，同一個 worker 裡的 ONNX Runtime 可能已經壞掉，連 CPU 也會回報同樣的 WebGPU 錯誤。同一次側欄開著期間，這個模型之後都直接用 CPU，不再試 WebGPU；關掉側欄重開才會重新嘗試。
- 換模型、上次載入失敗，或辨識時出過 WebGPU 錯誤之後，下次按「開始」都會換一個全新的 worker，不會沿用可能已經壞掉的那個（模型檔有快取，不會重新下載）。所以試過跑不動的自訂模型之後，換回內建模型可以直接用，不必關掉側欄。
- 已知：`willopcbeta/whisper-ja-760M-ONNX` 在 WebGPU 上會出現上面那個 Attention mask 錯誤，只能用 CPU 跑；760M 的模型在 CPU 上很慢，多半跟不上影片，請看「排隊」秒數。
- 第一次載入要下載模型，之後走瀏覽器快取。

### 翻譯引擎
- **Chrome 內建**（預設）：Chrome 的 Translator API，快，但一次只翻一句，看不到前文。
- **HY-MT1.5 本機模型**：騰訊的翻譯專用模型 [`onnx-community/HY-MT1.5-1.8B-ONNX`](https://huggingface.co/onnx-community/HY-MT1.5-1.8B-ONNX)，官方支援繁體中文。翻譯時會把前 2 句原文當上下文一起送進去（`MT_CONTEXT_SENTENCES`），日文省略主詞時比較不容易翻錯。
  - 需要 WebGPU，下載 1.2～1.4GB；跟辨識模型同時在 GPU 上，顯示記憶體要夠（搭配 Qwen3-ASR 1.7B 大約要 5GB 以上）。
  - 載入失敗時自動改用 Chrome 內建翻譯，狀態列會註明。
  - 提示詞用模型卡建議的中文模板；在 `mt-worker.js` 的 `buildPrompt`。

### 即時 / 精準
側欄的「翻譯方式」可以隨時切換（設定會記住）：
- **精準**（預設）：等到句尾再整句翻譯，譯文品質較好，但比原文晚出現。
- **即時**：每辨識出一段就立刻翻，譯文最快出現，但一句話被拆開時會失去上下文，翻譯較容易出錯。切到即時時，已經在等句尾的原文會馬上翻掉。

### 整句翻譯（精準模式）
日文常把一句話拆在好幾段音訊裡，單段翻譯會失去上下文。所以：
- 辨識出的原文會**立刻顯示**，譯文則等到句尾標點（。！？）才把累積的幾段原文合併成一句再翻譯。
- 被合併的段落左邊會有色條，上面幾段標示「↓ 與下方合併成一句翻譯」，譯文顯示在最後一段。
- 沒有句尾標點時，是否繼續等取決於說話者**是不是還在講**：切段器目前有正在錄的段落就繼續等；沒有正在錄的段落、已經安靜約 1 秒、辨識佇列也空了，就視為講完，直接送出。（不用固定逾時，因為下一段要錄 3～5 秒再辨識 1～2 秒才會到，固定逾時一定先過期。）
- 以下情況也不會再等：累積超過 80 字、已合併 4 段、從第一段原文出現起超過 12 秒。按「停止」時會把還沒湊成句的原文翻掉。
- 代價是譯文會比原文晚出現，晚多少取決於整句講多久。

### 看哪裡慢
每段時間旁會顯示各階段耗時，同樣的資料也會印在側欄 DevTools 的 console（`[timing]`）：

`00:18 · 音訊4.3s · 排隊0.0s · 辨識0.8s · 等句尾2.1s · 翻譯0.1s`

- **排隊**持續變大：辨識跟不上，改用較小的模型。
- **辨識**：模型推論時間。
- **等句尾**：為了湊成整句多等的時間（只有合併多段時才有）。
- 模型載入後會先用靜音預熱一次，所以第一段不會特別慢。

### 語速快 / 音訊模糊時
- **在最安靜處切段**：快語速幾乎沒有停頓。累積到 8 秒（`MAX_SEG_S`）仍沒遇到停頓時，會回頭在最近幾秒內音量最低的位置切，後面的音訊留給下一段，不會硬切在字中間，也不會丟音訊。
- **開頭留白**：每段往前多帶 0.3 秒，避免吃掉輕聲的字頭。
- **音量正規化**：說話部分太小聲時會放大（最多 8 倍，只放大不縮小）再送辨識。
- **Beam search**：辨識用 4 條路徑搜尋（`NUM_BEAMS`），比貪婪解碼不容易選錯字，代價是辨識較慢；如果環境不支援，會自動退回貪婪解碼。
- **輸出上限與重複抑制**：`max_new_tokens` 提高到 400，避免快語速的長段落尾巴被截斷；`no_repeat_ngram_size` 擋掉模糊音訊常見的重複迴圈。
- 以上都會增加辨識耗時，請留意每段時間旁的「排隊」秒數有沒有開始變大；變大就改用較小的模型，或把 `NUM_BEAMS` 調回 1。

### 時間與空白標示
- 每段前面的時間是**影片的播放時間**（讀分頁裡影片的 currentTime 換算），不是開始口譯後經過的時間；倍速播放也有算進去。讀不到影片時間時（例如影片嵌在別的網站的 iframe 裡），退回從開始口譯起算，狀態列會註明。
- 前轉／後轉時會出現「↪ 影片跳到 mm:ss」，跳轉前後的句子不會被合併成一句翻譯。
- 連續播放中有 2 秒以上沒有辨識到聲音，會出現「⋯ mm:ss～mm:ss 約 N 秒沒有辨識到聲音」：
  - 「期間都很安靜」：多半是影片本來就沒人講話。
  - 「期間最大音量到門檻的 N%，可能有小聲說話被當成靜音」：那段可能有人小聲講話被漏掉，可以把影片播放器的音量調大。
- 暫停不會被當成空白。

### 懷疑漏掉句子時
被略過的段落不會進逐字稿，但畫面上會留一行淡色小字（console 也有 `[skip]`），說明漏在哪一關：
- **「辨識結果為空」**：Whisper 什麼都沒輸出，通常是太小聲或只有雜音。
- **「被當成幻覺濾掉」**：辨識出的文字符合 `HALLUCINATIONS` 的規則（例如「Thank you」、被括號包住的內容）。如果濾掉的其實是正常內容，就該調整那份規則。
- **「有聲音的部分只有 x 秒，太短」**：低於 `MIN_VOICED_S`（0.3 秒）。
- **「辨識錯誤」**：worker 回報的錯誤。
- **完全沒有任何提示**：代表這段音量一直低於靜音門檻，被當成靜音，連段落都沒開始。門檻現在是自適應的（見下），console 的 `[vad]` 會每 20 秒印一次目前估計的底噪與門檻，可以用來判斷。

### 自適應靜音門檻
收音小聲、底噪大、有背景音樂的音檔，固定門檻會把人聲當成靜音（漏句），或把底噪當成人聲（切不出停頓）。所以門檻會隨底噪調整：
- 往回看最近 10 秒，取最安靜的 10% 音框當底噪，門檻 = 底噪 × 2，限制在 0.003～0.03 之間。
- 開頭約 1 秒還沒估出底噪時，使用預設的 `SILENCE_RMS`（0.008）。
- 限制：人聲和底噪音量差不多時，任何門檻都分不開；連續講話完全沒有空隙時，底噪會被高估。

## 注意事項
- 換到別的分頁要用時，要在那個分頁再點一次圖示（Chrome 的權限限制）。
- 影片分頁重新整理或換頁後，擷取會中斷，需要再點一次圖示、再按「開始口譯」。側欄本身不用重新整理。
- 側邊欄關掉就會停止擷取，已顯示的逐字稿也會消失，需要的話先匯出。
- 「自動偵測」語言時，翻譯模型下載需要點一下側欄裡出現的提示按鈕。
- 背景音樂很大聲的影片，辨識品質會下降；有 DRM 的串流平台可能抓不到聲音。
- 沒有 WebGPU 時 Whisper 會退回 CPU（wasm），明顯比較慢；Qwen3-ASR、Cohere Transcribe、HY-MT1.5 沒有 WebGPU 就不能用。manifest 設了跨來源隔離，讓 CPU 模式可以開多執行緒；如果模型下載或載入出現跨來源（COOP/COEP）錯誤，刪掉 `manifest.json` 裡 `cross_origin_embedder_policy` 與 `cross_origin_opener_policy` 兩行即可，程式會退回單執行緒。
- Chrome 內建翻譯只能單句翻，詞義錯誤（例如人名、地名、多義詞）無法完全避免。

## 可調參數
`sidepanel.js` 最上面：
- 切段：`MIN_SEG_S`、`MAX_SEG_S`、`END_SILENCE_S`，以及超過 `LONG_SEG_S` 後改用的 `LONG_END_SILENCE_S`、開頭的預設門檻 `SILENCE_RMS`，以及自適應門檻的 `NOISE_WINDOW_FRAMES`、`NOISE_PERCENTILE`、`NOISE_MULT`、`MIN_THRESHOLD`、`MAX_THRESHOLD`。
- 快語速 / 模糊音訊：`CUT_SEARCH_FROM_S`（上限時從第幾秒之後找低點）、`PRE_ROLL_FRAMES`（開頭留白）、`TARGET_RMS`、`MAX_GAIN`（音量正規化）。
- 整句合併（在「句子合併」區塊）：`MAX_GROUP_CHARS`、`MAX_GROUP_SEGS`、`GROUP_QUIET_S`、`GROUP_MAX_WAIT_MS`。

`asr-worker.js` 最上面：`NUM_BEAMS`、`MAX_NEW_TOKENS`、`NO_REPEAT_NGRAM`；另外 CPU 模式的執行緒數也在這支檔案。

`qwen3-asr-worker.js`：`TOKENS_PER_SECOND`（每秒音訊最多生成幾個 token，限制模型重複迴圈的耗時）。

`sidepanel.js`：`MT_CONTEXT_SENTENCES`（HY-MT1.5 帶幾句前文）。

## 函式庫與測試
- `lib/`：transformers.js 4.3.1（`transformers.min.js`）、onnxruntime-web 1.31.0-dev（`ort.webgpu.min.mjs`，給 Qwen3-ASR 用），以及兩者共用的 `ort-wasm-simd-threaded.asyncify.{mjs,wasm}`。transformers.js 4.x 預設會從 CDN 載入 wasm，MV3 不允許，所以每個 worker 都把 `wasmPaths` 指向本地 `lib/`（路徑統一寫在 `shared.js`）。`transformers.min.js` 有一處手動修改：`"Mistral3ForConditionalGeneration"` 拆成 `"Mistral3For"+"ConditionalGeneration"`，因為 GitHub 密鑰掃描會把原字串誤判成 API Key 擋下 push，升級時要再改一次。
- `node --test tests/*.test.mjs`：測 Qwen3-ASR 的 log-mel、float16 轉換、prompt、BPE 解碼，以及用假模型（`tests/fixtures/mock-qwen3/`，介面跟真模型相同）跑完整的解碼迴圈。
- 參考答案由 `tests/gen_fixtures.py`（numpy）產生；假模型由 `tests/make_mock_qwen3.py`（numpy + onnx）產生。
