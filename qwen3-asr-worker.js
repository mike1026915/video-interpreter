// Qwen3-ASR 辨識 worker。transformers.js 不支援 Qwen3-ASR，所以直接用 onnxruntime-web：
// encoder + decoder_init + decoder_step 三個 session，KV cache 留在 GPU 上。
// 模型格式見 jiangzhuo9357/Qwen3-ASR-1.7B-ONNX 的模型卡（layout v2，所有常數都在 prompt_config.json）。
// 跟 asr-worker.js 用同一套訊息：load → loaded、transcribe → result / error、progress、status。
import * as ort from './lib/ort.webgpu.min.mjs';
import { makeBpeDecoder, makeBpeEncoder, makeEmbedder, transcribe, fixRoleTokens } from './qwen3-asr-core.js';
import { ORT_WASM_PATHS, gpuInfo, echoesContext } from './shared.js';

ort.env.wasm.wasmPaths = { ...ORT_WASM_PATHS }; // 不從 CDN 載入，見 shared.js

const CACHE_NAME = 'qwen3-asr-models';
const CONTEXT_FIXED_TOKENS = 120;  // 背景資訊：使用者填的關鍵字＋影片標題，最多這麼多 token（從前面算）
const CONTEXT_RECENT_TOKENS = 120; // 背景資訊：前幾句辨識結果，最多這麼多 token（從最近的句子往回算）
const TOKENS_PER_SECOND = 25;   // 生成上限 = 音訊秒數 × 25 + 20（日文快語速約每秒 10 個 token），再受模型的 256 限制

const errMsg = (err) => String((err && err.message) || err);
const status = (text) => self.postMessage({ type: 'status', text });

let state = null;   // { model, sessions, cfg, embedder, decode, encode, filters, f16 }

// 下載一個檔案（有快取就直接讀），回報進度
async function fetchFile(model, file) {
  const url = `https://huggingface.co/${model}/resolve/main/${file}`;
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch {}
  const hit = cache && (await cache.match(url));
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    self.postMessage({ type: 'progress', file, loaded: buf.length, total: buf.length });
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下載 ${model}/${file} 失敗（HTTP ${res.status}）`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  // 知道大小就直接寫進預先配置好的陣列，不然數 GB 的檔案會同時有 chunks 和合併後兩份
  let buf = total ? new Uint8Array(total) : null;
  const chunks = [];
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && loaded + value.length <= total) buf.set(value, loaded);
    else {
      if (buf) { chunks.push(buf.subarray(0, loaded)); buf = null; } // 比宣告的大小還大：改用 chunks
      chunks.push(value);
    }
    loaded += value.length;
    if (loaded - lastPost > 4e6) {
      lastPost = loaded;
      self.postMessage({ type: 'progress', file, loaded, total: Math.max(total, loaded) });
    }
  }
  // 比宣告的少代表下載被截斷：不要寫進快取，否則之後每次都讀到壞檔。
  // （比宣告的多是 gzip 傳輸：content-length 是壓縮後的大小，截斷的話解壓時就會拋錯）
  if (total && loaded < total) {
    throw new Error(`下載 ${file} 不完整（${loaded} / ${total} bytes），請再按一次開始`);
  }
  if (!buf) {
    buf = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    chunks.length = 0;
  }
  self.postMessage({ type: 'progress', file, loaded, total: loaded });
  if (cache) {
    try {
      await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch (err) {
      console.warn('[qwen3-asr] 無法寫入快取，下次會重新下載', file, err);
    }
  }
  return buf;
}

const fetchJson = async (model, file) => JSON.parse(new TextDecoder().decode(await fetchFile(model, file)));

async function load(model) {
  if (state?.model === model) return;
  const gpu = await gpuInfo();
  if (!gpu.ok) throw new Error('Qwen3-ASR 需要 WebGPU（這台電腦或瀏覽器沒有可用的 WebGPU），請改用 Whisper 模型');
  const f16 = gpu.f16;

  status('下載模型設定…');
  const cfg = await fetchJson(model, 'prompt_config.json');
  if (cfg.layout_version !== 2) throw new Error(`不支援的 Qwen3-ASR 模型格式（layout_version ${cfg.layout_version}），需要 layout v2`);
  const variantKey = f16 && cfg.variants.q4f16 ? 'q4f16' : 'q4';
  const v = cfg.variants[variantKey];
  if (!v) throw new Error(`模型沒有 ${variantKey} 版本`);

  // 先下載小檔，再依序下載大檔（一次一個，避免同時佔用好幾 GB 記憶體）
  const [vocab, added, filters] = await Promise.all([
    fetchJson(model, 'vocab.json'),
    fetchJson(model, 'added_tokens.json'),
    fetchJson(model, cfg.mel.filters_file || 'mel_filters.json'),
  ]);
  const emb = cfg.embedding;
  const tableBytes = await fetchFile(model, emb.file);
  const table = emb.dtype === 'int8' ? new Int8Array(tableBytes.buffer)
    : emb.dtype === 'float16' ? new Uint16Array(tableBytes.buffer)
    : new Float32Array(tableBytes.buffer);
  const scales = emb.scales_file ? new Float32Array((await fetchFile(model, emb.scales_file)).buffer) : null;

  const opt = { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' };
  const kvOnGpu = { present_keys: 'gpu-buffer', present_values: 'gpu-buffer' };
  status(`建立辨識模型（WebGPU ${variantKey}）…`);
  const encoder = await ort.InferenceSession.create(await fetchFile(model, v.encoder), opt);
  const weights = v.weights ? await fetchFile(model, v.weights) : null;
  const ext = weights ? { externalData: [{ path: v.weights, data: weights }] } : {};
  const init = await ort.InferenceSession.create(await fetchFile(model, v.decoder_init),
    { ...opt, ...ext, preferredOutputLocation: kvOnGpu });
  const step = await ort.InferenceSession.create(await fetchFile(model, v.decoder_step),
    { ...opt, ...ext, preferredOutputLocation: kvOnGpu });

  // prefix 裡的角色名稱改回「system」「user」（見 fixRoleTokens 的說明）
  const encode = makeBpeEncoder(vocab);
  const imStart = added['<|im_start|>'];
  if (imStart != null) {
    const fixed = fixRoleTokens(cfg.prompt.prefix_ids, imStart, encode);
    if (fixed.join() !== cfg.prompt.prefix_ids.join()) {
      console.log('[qwen3-asr] 修正 prompt 的角色 token', cfg.prompt.prefix_ids, '→', fixed);
      cfg.prompt.prefix_ids = fixed;
    }
  }

  state = {
    model, cfg, filters, f16: variantKey === 'q4f16',
    sessions: { encoder, init, step },
    embedder: makeEmbedder(cfg, table, scales),
    decode: makeBpeDecoder(vocab, added),
    encode,
  };

  // 用 1 秒靜音實際跑一次：確認 WebGPU 能執行，也順便預熱
  status(`驗證模型能否執行（WebGPU ${variantKey}）…`);
  const t0 = performance.now();
  await run(new Float32Array(16000), 'ja', 4);
  console.log(`[qwen3-asr] ${model} WebGPU ${variantKey} 驗證＋預熱 ${Math.round(performance.now() - t0)}ms`);
}

function isoOf(name) {
  const hit = Object.entries(state.cfg.language_names || {}).find(([, n]) => n.toLowerCase() === name.toLowerCase());
  return hit ? hit[0] : null;
}

function run(audio, lang, maxNewTokens, contextIds = []) {
  const s = state;
  const forced = lang && s.cfg.language_prefix_ids?.[lang] ? lang : null;
  return transcribe({
    ort, sessions: s.sessions, cfg: s.cfg, embedder: s.embedder, decode: s.decode, filters: s.filters,
    audio, lang: forced, f16: s.f16, contextIds,
    maxNewTokens: maxNewTokens ?? Math.ceil((audio.length / 16000) * TOKENS_PER_SECOND) + 20,
  });
}

// 背景資訊 → token。context：{ fixed：關鍵字與影片標題, recent：前幾句辨識結果（舊 → 新）}
function contextIds(context) {
  if (!context) return [];
  const fixed = context.fixed?.trim() ? state.encode(context.fixed.trim()).slice(0, CONTEXT_FIXED_TOKENS) : [];
  let recent = [];
  for (const line of [...(context.recent || [])].reverse()) {   // 從最近的句子往回加，整句放不下就停
    const ids = state.encode(line);
    if (recent.length + ids.length > CONTEXT_RECENT_TOKENS) break;
    recent = [...ids, ...recent];
  }
  if (!fixed.length) return recent;
  return recent.length ? [...fixed, ...state.encode('\n'), ...recent] : fixed;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      await load(m.model);
      self.postMessage({ type: 'loaded', device: 'webgpu', fellBack: false, reason: '' });
    } else if (m.type === 'transcribe') {
      if (!state) throw new Error('辨識模型尚未載入成功');
      const ctxIds = contextIds(m.context);
      let out = await run(m.audio, m.lang || null, undefined, ctxIds);
      if (ctxIds.length && m.context.recent?.length && echoesContext(out.text, m.context.recent)) {
        // 照抄了前文：這段不帶背景資訊重新辨識
        const retry = await run(m.audio, m.lang || null);
        console.log('[qwen3-asr] 辨識結果照抄前文，不帶背景資訊重新辨識', { first: out.text, retry: retry.text });
        out = retry;
      }
      // 自動偵測時，模型自己判斷的語言（例如「Japanese」）轉成 ISO 代碼回傳，側欄就不必再偵測一次
      const lang = out.language ? isoOf(out.language) : null;
      if (out.language) console.log('[qwen3-asr] 偵測到語言', out.language, lang);
      self.postMessage({ type: 'result', id: m.id, text: out.text, lang });
    }
  } catch (err) {
    console.warn('[qwen3-asr]', err);
    self.postMessage({ type: 'error', id: m.id, message: errMsg(err) });
  }
};
