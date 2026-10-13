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
let loadedModel = null;
let loadedInfo = null;

const errMsg = (err) => String((err && err.message) || err);
const status = (text) => self.postMessage({ type: 'status', text });

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

// 載入成功不代表能跑：例如 WebGPU 可能在推論時才發現某個運算不支援
// （"[WebGPU] Kernel ... Mask not supported"）。所以用 1 秒靜音實際跑一次，順便預熱。
async function validate() {
  try {
    await asr(new Float32Array(16000), { language: 'english', task: 'transcribe' });
  } catch (err) {
    if (!/language|task/i.test(errMsg(err))) throw err;
    await asr(new Float32Array(16000)); // 模型不接受 language / task：不帶參數再試一次
  }
}

// forceWasm：這個模型先前在 WebGPU 上跑不動，側欄已經換了一個全新的 worker，只用 CPU 載入
async function load(model, forceWasm = false, forcedReason = '') {
  if (asr && loadedModel === model) return loadedInfo;
  if (asr) { try { await asr.dispose(); } catch {} asr = null; loadedModel = null; }

  const gpu = !forceWasm && (await hasWebGPU());
  const attempts = [
    ...(gpu ? dtypeCandidates(model, true).map((dtype) => ({ device: 'webgpu', dtype })) : []),
    ...dtypeCandidates(model, false).map((dtype) => ({ device: 'wasm', dtype })),
  ];
  const progress_callback = (p) => {
    if (p.status === 'progress' && p.total) {
      self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
    }
  };

  let lastErr = null, gpuErr = null;
  for (const { device, dtype } of attempts) {
    if (device === 'wasm') {
      if (gpu && gpuErr) status(`WebGPU 無法執行這個模型，改用 CPU（會慢很多）…（${errMsg(gpuErr).slice(0, 120)}）`);
      if (forceWasm) status('改用 CPU 載入模型（會慢很多）…');
      // 只有跨來源隔離（manifest 的 COOP/COEP）生效時才有 SharedArrayBuffer，才能開多執行緒
      env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
        ? Math.max(1, Math.min(4, Math.floor((self.navigator.hardwareConcurrency || 2) / 2)))
        : 1;
    }
    const label = `${device} ${JSON.stringify(dtype)}`;
    try {
      asr = await pipeline('automatic-speech-recognition', model, { device, dtype, progress_callback });
      status(`驗證模型能否執行（${label}）…`);
      const t0 = performance.now();
      await validate();
      console.log(`[asr] 模型 ${model} 使用 ${label} 執行緒 ${env.backends.onnx.wasm.numThreads} 驗證＋預熱 ${Math.round(performance.now() - t0)}ms`);
      loadedModel = model;
      loadedInfo = {
        device,
        fellBack: forceWasm || (gpu && device === 'wasm'),
        reason: (forcedReason || (gpuErr ? errMsg(gpuErr) : '')).slice(0, 160),
      };
      return loadedInfo;
    } catch (err) {
      lastErr = err;
      console.warn(`[asr] ${label} 失敗`, err);
      if (asr) { try { await asr.dispose(); } catch {} asr = null; }
      if (device === 'webgpu') {
        gpuErr = err;
        // WebGPU 執行階段的錯誤（某個運算不支援）跟量化無關，換量化也一樣會壞，而且每換一組就要重新下載一套模型檔。
        // 這種錯誤之後，同一個 worker 裡的 ONNX Runtime 可能已經壞掉，連 CPU session 也會回報同樣的錯誤，
        // 所以不在這裡改用 CPU，而是請側欄關掉這個 worker、開一個全新的只用 CPU。
        if (/\[WebGPU\]|Kernel/i.test(errMsg(err))) {
          throw Object.assign(new Error(errMsg(err)), { restartForCpu: true });
        }
      }
      status(`${label} 無法使用，嘗試其他設定…（${errMsg(err).slice(0, 120)}）`);
    }
  }
  throw new Error(`模型 ${model} 無法載入或執行：${errMsg(lastErr)}`);
}

async function decode(audio, language) {
  if (!asr) throw new Error('辨識模型尚未載入成功');
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
      langOk = true;  // 換模型時重新嘗試
      beamsOk = true;
      let info;
      try {
        info = await load(m.model, !!m.forceWasm, m.reason || '');
      } catch (err) {
        if (!err.restartForCpu) throw err;
        self.postMessage({ type: 'restart', model: m.model, reason: errMsg(err) });
        return;
      }
      self.postMessage({ type: 'loaded', ...info });
    } else if (m.type === 'transcribe') {
      const out = await decode(m.audio, m.language);
      self.postMessage({ type: 'result', id: m.id, text: (out.text || '').trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: errMsg(err) });
  }
};
