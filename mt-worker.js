// 本機 LLM 翻譯 worker：騰訊 HY-MT1.5-1.8B（翻譯專用，官方支援繁體中文）。
// 跟 Chrome 內建翻譯不同，可以把前面幾句原文當上下文一起送進去（日文常省略主詞）。
// 訊息：load → loaded / error、translate {id, text, context} → result / error、progress、status。
import { pipeline, env } from './lib/transformers.min.js';
import { ORT_WASM_PATHS, gpuInfo } from './shared.js';

env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = { ...ORT_WASM_PATHS }; // 不從 CDN 載入，見 shared.js
// 不要用 transformers.js 的 wasm 快取：它會把 ORT 的 .mjs 轉成 blob: 網址再 import，
// 擴充功能的 CSP（script-src 'self'）會擋掉，結果是「no available backend found」
env.useWasmCache = false;

const MODEL = 'onnx-community/HY-MT1.5-1.8B-ONNX';
const TARGET_NAME = '繁体中文';   // HY-MT 的語言表用簡體中文寫語言名稱
const MAX_NEW_TOKENS = 256;

let generator = null;
let chain = Promise.resolve();  // 一次只翻一句：同時跑兩個 generate 會搶同一個 GPU session
const errMsg = (err) => String((err && err.message) || err);

// HY-MT1.5 模型卡建議的提示詞（目標語言是中文時用中文模板；有上下文時用「參考上面的資訊」模板）
function buildPrompt(text, context) {
  if (context) {
    return `${context}\n参考上面的信息，把下面的文本翻译成${TARGET_NAME}，注意不需要翻译上文，也不要额外解释：\n${text}`;
  }
  return `将以下文本翻译为${TARGET_NAME}，注意只需要输出翻译后的结果，不要额外解释：\n\n${text}`;
}

async function load() {
  if (generator) return;
  const gpu = await gpuInfo();
  if (!gpu.ok) throw new Error('HY-MT1.5 翻譯需要 WebGPU，這台電腦沒有可用的 WebGPU');
  const dtype = gpu.f16 ? 'q4f16' : 'q4';
  self.postMessage({ type: 'status', text: `載入翻譯模型 HY-MT1.5（WebGPU ${dtype}）…` });
  generator = await pipeline('text-generation', MODEL, {
    device: 'webgpu',
    dtype,
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
      }
    },
  });
  const t0 = performance.now();
  await translate('はい', '');  // 預熱
  console.log(`[mt] HY-MT1.5 ${dtype} 預熱 ${Math.round(performance.now() - t0)}ms`);
}

async function translate(text, context) {
  const out = await generator([{ role: 'user', content: buildPrompt(text, context) }],
    { max_new_tokens: MAX_NEW_TOKENS, do_sample: false });
  const g = out?.[0]?.generated_text;
  const s = Array.isArray(g) ? g[g.length - 1]?.content : g;
  return String(s || '').trim();
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      await load();
      self.postMessage({ type: 'loaded' });
    } else if (m.type === 'translate') {
      if (!generator) throw new Error('翻譯模型尚未載入');
      const job = chain.then(() => translate(m.text, m.context || ''));
      chain = job.catch(() => {});
      self.postMessage({ type: 'result', id: m.id, text: await job });
    }
  } catch (err) {
    console.warn('[mt]', err);
    self.postMessage({ type: 'error', id: m.id, message: errMsg(err) });
  }
};
