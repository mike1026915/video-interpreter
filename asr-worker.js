// 在 Web Worker 裡跑 Whisper，避免卡住側邊欄畫面
import { pipeline, env } from './lib/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache = true; // 模型下載一次後會快取
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;

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
  }
  asr = await pipeline('automatic-speech-recognition', model, opts);
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
      const out = await asr(m.audio, {
        language: m.language || null,
        task: 'transcribe',
        chunk_length_s: 30,
      });
      self.postMessage({ type: 'result', id: m.id, text: (out.text || '').trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
