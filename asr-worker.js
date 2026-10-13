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
let beamsOk = true;

let asr = null;
let loadedKey = null;

async function hasWebGPU() {
  try {
    return !!(self.navigator.gpu && (await self.navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function load(model) {
  const gpu = await hasWebGPU();
  const key = model + (gpu ? '@webgpu' : '@wasm');
  if (asr && loadedKey === key) return { device: gpu ? 'webgpu' : 'wasm' };
  if (asr) { try { await asr.dispose(); } catch {} asr = null; }

  const opts = {
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
      }
    },
  };
  if (gpu) {
    opts.device = 'webgpu';
    opts.dtype = model.includes('large')
      ? { encoder_model: 'fp16', decoder_model_merged: 'q4' }
      : { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  } else {
    opts.device = 'wasm';
    opts.dtype = 'q8';
    // 只有跨來源隔離（manifest 的 COOP/COEP）生效時才有 SharedArrayBuffer，才能開多執行緒
    env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(4, Math.floor((self.navigator.hardwareConcurrency || 2) / 2)))
      : 1;
  }
  asr = await pipeline('automatic-speech-recognition', model, opts);

  // 預熱：第一次推論要編譯 shader / 初始化，先用靜音跑一次，免得第一段特別慢
  const t0 = performance.now();
  try {
    await asr(new Float32Array(16000), { language: 'english', task: 'transcribe' });
  } catch (err) {
    console.warn('[asr] 預熱失敗（不影響使用）', err);
  }
  console.log(`[asr] 模型 ${model} 裝置 ${opts.device} 執行緒 ${env.backends.onnx.wasm.numThreads} 預熱 ${Math.round(performance.now() - t0)}ms`);

  loadedKey = key;
  return { device: opts.device };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      const info = await load(m.model);
      self.postMessage({ type: 'loaded', ...info });
    } else if (m.type === 'transcribe') {
      const base = {
        language: m.language || null,
        task: 'transcribe',
        chunk_length_s: 30,
        max_new_tokens: MAX_NEW_TOKENS,
      };
      let out;
      if (beamsOk) {
        try {
          out = await asr(m.audio, { ...base, num_beams: NUM_BEAMS, no_repeat_ngram_size: NO_REPEAT_NGRAM });
        } catch (err) {
          // beam search 在某些環境可能不支援，退回貪婪解碼，並且之後不再嘗試
          console.warn('[asr] beam search 失敗，改用貪婪解碼', err);
          beamsOk = false;
        }
      }
      if (!out) out = await asr(m.audio, base);
      self.postMessage({ type: 'result', id: m.id, text: (out.text || '').trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
