// 在 Web Worker 裡跑 Whisper / Cohere Transcribe（transformers.js），避免卡住側邊欄畫面
import { pipeline, env } from './lib/transformers.min.js';
import { ORT_WASM_PATHS, isCohere, gpuInfo } from './shared.js';

env.allowLocalModels = false;
env.useBrowserCache = true; // 模型下載一次後會快取
env.backends.onnx.wasm.wasmPaths = { ...ORT_WASM_PATHS }; // 不從 CDN 載入，見 shared.js
env.backends.onnx.wasm.numThreads = 1;

const WHISPER_LANG = {
  en: 'english', ja: 'japanese', ko: 'korean', fr: 'french',
  de: 'german', es: 'spanish', zh: 'chinese',
};

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

// WebGPU 執行階段的錯誤（某個運算不支援）。這種錯誤之後同一個 worker 可能已經壞掉。
const isGpuRuntimeErr = (err) => /\[WebGPU\]|Kernel/i.test(errMsg(err));
// 找不到某組量化的檔案。只有這種錯誤才值得換下一組量化：其他錯誤換了也一樣會壞，
// 而且每換一組就要重新下載一套模型檔（大模型一套就要 1GB 以上）。
const isMissingFileErr = (err) => /could not locate|not found|404|no such file/i.test(errMsg(err));

// 依序嘗試的量化設定。內建模型的檔案已知，只用一組，絕不嘗試其他量化（避免悄悄下載別的版本）；
// 自訂模型的檔案命名不一定相同，找不到檔案時才換下一組。刻意不放純 fp32，避免下載好幾 GB。
// f16：顯示卡支援 shader-f16 才用 fp16 的檔案，否則改用 q4（大模型）或 fp32（小模型）的 encoder。
function dtypeCandidates(model, gpu, f16) {
  if (isCohere(model)) return gpu ? [f16 ? 'q4f16' : 'q4'] : [];
  const builtin = model.startsWith('onnx-community/whisper-');
  const big = /large|turbo|medium|760M|1\.5B/i.test(model);
  if (!gpu) return builtin ? ['q8'] : ['q8', 'q4'];
  if (builtin) {
    if (!big) return [{ encoder_model: 'fp32', decoder_model_merged: 'q4' }];
    return [{ encoder_model: f16 ? 'fp16' : 'q4', decoder_model_merged: 'q4' }];
  }
  const list = f16 ? [
    { encoder_model: big ? 'fp16' : 'fp32', decoder_model_merged: 'q4' },
    { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    { encoder_model: 'fp16', decoder_model_merged: 'fp16' },
    'q8',
    'fp16',
  ] : [
    { encoder_model: big ? 'q4' : 'fp32', decoder_model_merged: 'q4' },
    'q8',
    'q4',
  ];
  const seen = new Set();
  return list.filter((d) => !seen.has(JSON.stringify(d)) && seen.add(JSON.stringify(d)));
}

// 載入成功不代表能跑：例如 WebGPU 可能在推論時才發現某個運算不支援
// （"[WebGPU] Kernel ... Mask not supported"）。所以用 1 秒靜音實際跑一次，順便預熱。
// 只有 WebGPU 執行階段錯誤算失敗；其他預熱錯誤跟以前一樣只警告（實際辨識通常仍然正常）。
async function validate(model) {
  const silence = new Float32Array(16000);
  if (isCohere(model)) {
    try {
      await asr(silence, { language: 'en', max_new_tokens: 4 });
    } catch (err) {
      if (isGpuRuntimeErr(err)) throw err;
      console.warn('[asr] 預熱失敗（不影響使用）', err);
    }
    return;
  }
  try {
    await asr(silence, { language: 'english', task: 'transcribe' });
    return;
  } catch (err) {
    if (isGpuRuntimeErr(err)) throw err;
    if (!/language|task/i.test(errMsg(err))) { console.warn('[asr] 預熱失敗（不影響使用）', err); return; }
  }
  try {
    await asr(silence); // 模型不接受 language / task：不帶參數再試一次
  } catch (err) {
    if (isGpuRuntimeErr(err)) throw err;
    console.warn('[asr] 預熱失敗（不影響使用）', err);
  }
}

// forceWasm：這個模型先前在 WebGPU 上跑不動，側欄已經換了一個全新的 worker，只用 CPU 載入
async function load(model, forceWasm = false, forcedReason = '') {
  if (asr && loadedModel === model) return loadedInfo;
  if (asr) { try { await asr.dispose(); } catch {} asr = null; loadedModel = null; }

  const info = forceWasm ? { ok: false, f16: false } : await gpuInfo();
  const gpu = info.ok;
  if (isCohere(model) && !gpu) {
    throw new Error('Cohere Transcribe 需要 WebGPU（模型 2B，用 CPU 跟不上影片），請改用 Whisper 模型');
  }
  const attempts = [
    ...(gpu ? dtypeCandidates(model, true, info.f16).map((dtype) => ({ device: 'webgpu', dtype })) : []),
    ...dtypeCandidates(model, false, false).map((dtype) => ({ device: 'wasm', dtype })),
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
      await validate(model);
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
      if (device === 'webgpu') gpuErr = err;
      // WebGPU 執行階段的錯誤（某個運算不支援）跟量化無關，換量化也一樣會壞。
      // 這種錯誤之後，同一個 worker 裡的 ONNX Runtime 可能已經壞掉，連 CPU session 也會回報同樣的錯誤，
      // 所以不在這裡改用 CPU，而是請側欄關掉這個 worker、開一個全新的只用 CPU。
      if (device === 'webgpu' && isGpuRuntimeErr(err) && !isCohere(model)) {
        throw Object.assign(new Error(errMsg(err)), { restartForCpu: true });
      }
      // 其他錯誤不再換下一組：換了也一樣會壞，還會一套接一套地下載模型檔
      if (!isMissingFileErr(err)) break;
      status(`${label} 找不到對應的模型檔，嘗試其他量化設定…（${errMsg(err).slice(0, 120)}）`);
    }
  }
  throw new Error(`模型 ${model} 無法載入或執行：${errMsg(lastErr)}`);
}

// lang：ISO 代碼（ja、en…），自動偵測時為 null
async function decode(audio, lang) {
  if (!asr) throw new Error('辨識模型尚未載入成功');
  if (isCohere(loadedModel)) {
    if (!lang) throw new Error('Cohere Transcribe 不能自動偵測語言，請在「原文語言」選擇影片的語言');
    // 貪婪解碼；448 個 token 足夠 15 秒的段落，也限制模型重複迴圈時的耗時
    return asr(audio, { language: lang, max_new_tokens: 448 });
  }
  return decodeWhisper(audio, WHISPER_LANG[lang] || null);
}

async function decodeWhisper(audio, language) {
  const opts = { chunk_length_s: 30, max_new_tokens: MAX_NEW_TOKENS };
  if (langOk) { opts.language = language || null; opts.task = 'transcribe'; }

  if (beamsOk) {
    try {
      return await asr(audio, { ...opts, num_beams: NUM_BEAMS, no_repeat_ngram_size: NO_REPEAT_NGRAM });
    } catch (err) {
      if (langOk && /language|task/i.test(errMsg(err))) { langOk = false; return decodeWhisper(audio, language); }
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
      return decodeWhisper(audio, language);
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
      const out = await decode(m.audio, m.lang || null);
      self.postMessage({ type: 'result', id: m.id, text: (out.text || '').trim() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: errMsg(err) });
  }
};
