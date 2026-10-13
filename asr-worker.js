// 在 Web Worker 裡跑 Whisper，避免卡住側邊欄畫面
import { pipeline, env } from './lib/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true; // 模型下載一次後會快取
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;

// 解碼參數：語速快或音質差時，beam search 比貪婪解碼不容易選錯字（較慢）
const NUM_BEAMS = 4;
const MAX_NEW_TOKENS = 400;     // 快語速的 10 秒日文可能超過 224 個 token；Whisper 上限 448
const NO_REPEAT_NGRAM = 6;      // 禁止連續重複 6 個 token 的片語，擋掉模糊音訊常見的重複迴圈
let beamsOk = true;             // beam search 失敗一次就退回貪婪解碼，之後不再嘗試
let langOk = true;              // 模型不接受 language / task 參數時（例如單語模型）就不再傳

let asr = null;
let loadedKey = null;

async function hasWebGPU() {
  try {
    return !!(self.navigator.gpu && (await self.navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

// 依序嘗試的量化設定。內建模型第一組就會成功；自訂模型的檔案命名不一定相同，失敗就換下一組。
// 刻意不放純 fp32：萬一前面都失敗，不要悄悄下載好幾 GB。
function dtypeCandidates(model, gpu) {
  if (!gpu) return ['q8', 'q4'];
  const big = /large|turbo|medium|760M|1\.5B/i.test(model);
  const list = [
    { encoder_model: big ? 'fp16' : 'fp32', decoder_model_merged: 'q4' },
    { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    { encoder_model: 'fp16', decoder_model_merged: 'fp16' },
    'q8',
    'fp16',
  ];
  const seen = new Set();
  return list.filter((d) => !seen.has(JSON.stringify(d)) && seen.add(JSON.stringify(d)));
}

async function load(model) {
  const gpu = await hasWebGPU();
  const key = model + (gpu ? '@webgpu' : '@wasm');
  if (asr && loadedKey === key) return { device: gpu ? 'webgpu' : 'wasm' };
  if (asr) { try { await asr.dispose(); } catch {} asr = null; }

  const opts = {
    device: gpu ? 'webgpu' : 'wasm',
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
      }
    },
  };
  if (!gpu) {
    // 只有跨來源隔離（manifest 的 COOP/COEP）生效時才有 SharedArrayBuffer，才能開多執行緒
    env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(4, Math.floor((self.navigator.hardwareConcurrency || 2) / 2)))
      : 1;
  }

  let usedDtype = null, lastErr = null;
  for (const dtype of dtypeCandidates(model, gpu)) {
    try {
      asr = await pipeline('automatic-speech-recognition', model, { ...opts, dtype });
      usedDtype = dtype;
      break;
    } catch (err) {
      lastErr = err;
      const msg = String((err && err.message) || err);
      console.warn('[asr] 載入失敗，換下一組量化設定', dtype, err);
      self.postMessage({ type: 'status', text: `量化設定 ${JSON.stringify(dtype)} 載入失敗，嘗試其他設定…（${msg.slice(0, 120)}）` });
    }
  }
  if (!asr) throw new Error(`模型 ${model} 載入失敗：${String((lastErr && lastErr.message) || lastErr)}`);

  // 預熱：第一次推論要編譯 shader / 初始化，先用靜音跑一次，免得第一段特別慢
  const t0 = performance.now();
  try {
    await asr(new Float32Array(16000), { language: 'english', task: 'transcribe' });
  } catch (err) {
    console.warn('[asr] 預熱失敗（不影響使用）', err);
  }
  console.log(`[asr] 模型 ${model} 裝置 ${opts.device} 量化 ${JSON.stringify(usedDtype)} 執行緒 ${env.backends.onnx.wasm.numThreads} 預熱 ${Math.round(performance.now() - t0)}ms`);

  loadedKey = key;
  return { device: opts.device };
}

const errMsg = (err) => String((err && err.message) || err);

async function decode(audio, language) {
  const opts = { chunk_length_s: 30, max_new_tokens: MAX_NEW_TOKENS };
  if (langOk) { opts.language = language || null; opts.task = 'transcribe'; }

  if (beamsOk) {
    try {
      return await asr(audio, { ...opts, num_beams: NUM_BEAMS, no_repeat_ngram_size: NO_REPEAT_NGRAM });
    } catch (err) {
      if (langOk && /language|task/i.test(errMsg(err))) { langOk = false; return decode(audio, language); }
      // beam search 在某些環境可能不支援，退回貪婪解碼，並且之後不再嘗試
      console.warn('[asr] beam search 失敗，改用貪婪解碼', err);
      beamsOk = false;
    }
  }
  try {
    return await asr(audio, opts);
  } catch (err) {
    if (langOk && /language|task/i.test(errMsg(err))) {
      console.warn('[asr] 這個模型不接受 language / task 參數，之後不再傳', err);
      langOk = false;
      return decode(audio, language);
    }
    throw err;
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      langOk = true; // 換模型時重新嘗試
      const info = await load(m.model);
      self.postMessage({ type: 'loaded', ...info });
    } else if (m.type === 'transcribe') {
      const out = await decode(m.audio, m.language);
      self.postMessage({ type: 'result', id: m.id, text: (out.text || '').trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: errMsg(err) });
  }
};
