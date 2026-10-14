import {
  makeAgc, periodicity, isQwen3Asr, isCohere, gpuInfo, collapseRepeats, dropEchoedContext, dropPromptEcho, dropHangul, suspiciousContextOutput, videoStartFrom, classifyGap,
} from './shared.js';

// ===== 設定 =====
const TARGET = 'zh-Hant';
const SR = 16000;               // Whisper 需要 16kHz
const FRAME = SR / 10;          // 100ms 一個音框，用來判斷有沒有人說話
const SILENCE_RMS = 0.008;      // 底噪還沒估出來前（開頭約 1 秒）使用的預設門檻；之後改用自適應門檻
// 自適應靜音門檻：用最近一段時間裡「最安靜的那一小部分音框」估計底噪，門檻 = 底噪 × 倍數。
// 收音小聲、底噪大、背景音樂的音檔，固定門檻會把人聲當靜音（漏句）或把底噪當人聲（切不出停頓）。
const NOISE_WINDOW_FRAMES = 100; // 往回看 10 秒
const NOISE_PERCENTILE = 0.1;    // 取最安靜的 10% 當底噪（說話中的換氣、字間空隙）
const NOISE_MULT = 2;
// 音量沒到門檻、但超過底噪 × SOFT_NOISE_MULT，而且聲音有人聲的週期性（有音高），也算有人講話：
// 小聲說話跟背景音差不多大時才抓得到，又不會把起伏的雜音當成講話（見 shared.js 的 periodicity）
const SOFT_NOISE_MULT = 1.25;
const VOICED_PERIODICITY = 0.6;
const MUSIC_PERIODICITY = 0.45;  // 背景（底噪附近的音框）平均週期性超過這個值，代表背景是音樂，有音高不代表有人講話：不補判
const NOISE_WARMUP_FRAMES = 10;  // 累積這麼多音框（1 秒）後才開始用估計值
const MIN_THRESHOLD = 0.003;
const MAX_THRESHOLD = 0.03;      // 上限：收音很小聲時，不要讓門檻高到把人聲也蓋掉
// 底噪本身就超過上限時（BGM、環境音很大聲），上限改成底噪 × HIGH_FLOOR_MULT：
// 門檻至少要比底噪高一點，不然所有聲音都算「有人講話」，段落只能等到 8 秒硬切，常切在句子中間
const HIGH_FLOOR_MULT = 1.4;
const MIN_SEG_S = 3;            // 一段至少 3 秒才切（太短會把一句話拆開，翻譯失去上下文）
const MAX_SEG_S = 8;            // 累積到 8 秒還沒遇到停頓，就在最安靜的地方切（快語速也不會一段塞太多字）
const CUT_SEARCH_FROM_S = 4;    // 到上限時，只在這個時間點之後找最安靜的音框來切
const PRE_ROLL_FRAMES = 3;      // 段落開頭往前多帶 0.3 秒，避免吃掉輕聲的字頭
const TARGET_RMS = 0.1;         // 送辨識前，把說話部分的音量拉到這個程度（只放大，不縮小）
const MAX_GAIN = 8;             // 最多放大幾倍，避免把底噪放大成人聲
const INPUT_MAX_GAIN = 16;      // 判斷有沒有人講話之前的自動增益上限（很小聲的影片才會用到，見 shared.js 的 makeAgc）
const END_SILENCE_S = 0.6;      // 講完停頓 0.6 秒就切段
const LONG_SEG_S = 6;           // 累積超過 6 秒後，改用較短的停頓就切，避免硬切在字中間
const LONG_END_SILENCE_S = 0.35;
const MIN_VOICED_S = 0.3;       // 有聲音的部分不到 0.3 秒就丟掉（避免 Whisper 對靜音亂講；太高會吃掉「はい」這類短回應）
const MIN_PITCHED_VOICED_S = 0.2; // 有聲音的部分 0.2～0.3 秒、但有人聲的音高（「え？」、被笑聲隔開的句尾）也送去辨識；碰撞聲、雜音沒有音高，照樣丟掉
const STRONG_PERIODICITY = 0.8;    // 只有一個音框（0.1 秒）超過門檻時，音高要這麼明顯才送（短促的「え」）
const MIN_NOTE_VOICED_S = 0.15; // 被丟掉的段落若有聲音部分超過這個長度，就在畫面上留一行提示
const GAP_QUIET_RATIO = 0.5;    // 空白期間最大音量超過門檻的這個比例，就提示「可能有小聲說話被當成靜音」

const TURBO = 'onnx-community/whisper-large-v3-turbo';
const SMALL = 'onnx-community/whisper-small';
const MT_CONTEXT_SENTENCES = 2; // HY-MT1.5 翻譯時，帶上前面幾句原文當上下文

// 靜音或音樂時 Whisper 常見的幻覺句
const HALLUCINATIONS = [
  /^thank(s| you)( for watching)?[.!]?$/i, /^you$/i, /^\.+$/, /^ご視聴ありがとうございました/,
  /^字幕/, /^\[.*\]$/, /^\(.*\)$/, /^♪+$/,
];

// ===== DOM =====
const $ = (id) => document.getElementById(id);
const els = {
  srcLang: $('srcLang'), model: $('model'), customModel: $('customModel'), customRow: $('customRow'),
  mode: $('mode'), engine: $('engine'), start: $('startBtn'), stop: $('stopBtn'),
  exp: $('exportBtn'), clear: $('clearBtn'), status: $('status'), progress: $('progress'),
  dl: $('dlTranslatorBtn'), list: $('list'),
};

function setStatus(text, isErr = false) {
  els.status.textContent = text;
  els.status.classList.toggle('err', isErr);
}

// ===== 設定記憶 =====
function showCustomRow() { els.customRow.hidden = els.model.value !== 'custom'; }

let modelPicked = false;          // 使用者在這次開側欄後自己選過模型：不要再套用預設值
chrome.storage.local.get(['srcLang', 'model', 'customModel', 'mode', 'engine']).then(async (s) => {
  if (s.srcLang) els.srcLang.value = s.srcLang;
  if (s.customModel) els.customModel.value = s.customModel;
  if (s.mode) els.mode.value = s.mode;
  if (s.engine) els.engine.value = s.engine;
  if (s.model) els.model.value = s.model;
  else {
    const def = (await gpuInfo()).ok ? TURBO : SMALL; // 沒選過模型：有 WebGPU 用 turbo，日文準很多
    if (!modelPicked) els.model.value = def;
  }
  showCustomRow();
});
els.srcLang.onchange = () => chrome.storage.local.set({ srcLang: els.srcLang.value });
els.engine.onchange = () => chrome.storage.local.set({ engine: els.engine.value });

els.model.onchange = () => {
  modelPicked = true;
  chrome.storage.local.set({ model: els.model.value });
  showCustomRow();
};
els.customModel.onchange = () => chrome.storage.local.set({ customModel: els.customModel.value.trim() });

// 回傳要載入的模型 ID；自訂模型格式不對時丟出錯誤
function getModel() {
  if (els.model.value !== 'custom') return els.model.value;
  const id = els.customModel.value.trim().replace(/^https?:\/\/huggingface\.co\//, '').replace(/\/+$/, '');
  if (!/^[\w.-]+\/[\w.-]+$/.test(id)) {
    throw new Error('自訂模型 ID 格式應為「作者/模型名」，例如 willopcbeta/whisper-ja-760M-ONNX');
  }
  return id;
}
els.mode.onchange = () => {
  chrome.storage.local.set({ mode: els.mode.value });
  // 切到即時：把已經在等句尾的原文馬上翻掉
  if (els.mode.value === 'live') flushGroup();
};

// ===== 辨識 worker（Whisper / Cohere：asr-worker.js；Qwen3-ASR：qwen3-asr-worker.js）=====
let worker = null;
let workerReady = null;
const pendingAsr = new Map();
const fileProgress = new Map();
// WebGPU 跑不動的模型 -> 失敗原因。這些模型直接在新的 worker 裡用 CPU 載入，不再試 WebGPU。
const gpuBroken = new Map();

let workerModel = null;         // 目前這個 worker 載入（或正在載入）的模型
let workerDirty = false;        // 這個 worker 出過錯（WebGPU 失敗後可能整個壞掉），下次載入要換新的

function spawnWorker(model) {
  worker?.terminate();
  // 舊 worker 裡還沒回來的辨識結果不會再回來了，先讓它們失敗，免得佇列卡住
  for (const p of pendingAsr.values()) p.reject(new Error('辨識 worker 已重新啟動'));
  pendingAsr.clear();
  worker = new Worker(isQwen3Asr(model) ? 'qwen3-asr-worker.js' : 'asr-worker.js', { type: 'module' });
  worker.onmessage = onWorkerMessage;
  // worker 崩潰（例如顯示記憶體不足）或程式載入失敗時不會有任何訊息：讓等待中的載入和辨識失敗，不要卡住
  worker.onerror = (e) => {
    const err = new Error(`辨識 worker 異常結束${e.message ? `：${e.message}` : '（可能是記憶體不足）'}`);
    workerDirty = true;
    if (workerReady && !workerReady.done) workerReady.reject(err);
    for (const p of pendingAsr.values()) p.reject(err);
    pendingAsr.clear();
  };
  workerDirty = false;
}

// 下載進度：所有檔案加總顯示
function showProgress(key, loaded, total, what) {
  fileProgress.set(key, [loaded, total]);
  let l = 0, t = 0;
  for (const [a, b] of fileProgress.values()) { l += a; t += b; }
  els.progress.hidden = false;
  els.progress.firstElementChild.style.width = `${(l / t) * 100}%`;
  setStatus(`下載${what}模型中… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB（只有第一次需要）`);
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.type === 'progress') {
    showProgress(m.file, m.loaded, m.total, '辨識');
  } else if (m.type === 'status') {
    setStatus(m.text);
  } else if (m.type === 'restart') {
    // WebGPU 推論失敗後，同一個 worker 裡的 ONNX Runtime 可能已經壞掉，連 CPU 也跑不動。
    // 所以整個 worker 關掉，開一個全新的只用 CPU 載入。
    gpuBroken.set(m.model, m.reason);
    setStatus(`WebGPU 無法執行這個模型，重新啟動並改用 CPU…（${m.reason.slice(0, 120)}）`, true);
    spawnWorker(m.model);
    fileProgress.clear();
    worker.postMessage({ type: 'load', model: m.model, forceWasm: true, reason: m.reason });
  } else if (m.type === 'loaded') {
    els.progress.hidden = true;
    workerReady?.resolve(m);
  } else if (m.type === 'result' || m.type === 'error') {
    if (m.type === 'error' && workerReady && !workerReady.done) {
      workerReady.reject(new Error(m.message));
      return;
    }
    // 辨識時出現 WebGPU 執行錯誤：這個 worker 可能已經壞掉，下次按「開始」時換一個新的
    if (m.type === 'error' && /\[WebGPU\]|Kernel/i.test(m.message)) workerDirty = true;
    const p = pendingAsr.get(m.id);
    pendingAsr.delete(m.id);
    if (!p) return;
    m.type === 'result' ? p.resolve({ text: m.text, lang: m.lang || null }) : p.reject(new Error(m.message));
  }
}

function loadModel(model) {
  // 換模型、上次載入失敗、或辨識時出過 WebGPU 錯誤：都換一個全新的 worker，
  // 不要在可能已經壞掉的 ONNX Runtime 上繼續載入（那樣會一直失敗、一直重新下載）。
  // 模型檔有瀏覽器快取，換 worker 不會重新下載。
  if (!worker || workerDirty || workerModel !== model) spawnWorker(model);
  workerModel = model;
  fileProgress.clear();
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  workerReady = {
    promise,
    resolve: (v) => { workerReady.done = true; resolve(v); },
    reject: (e) => { workerReady.done = true; reject(e); },
  };
  const reason = gpuBroken.get(model);
  worker.postMessage({ type: 'load', model, forceWasm: !!reason, reason });
  promise.catch(() => { workerDirty = true; });
  return promise;
}

let asrSeq = 0;
// lang：ISO 代碼（ja、en…），自動偵測時為 null
function transcribe(audio, lang) {
  const id = ++asrSeq;
  return new Promise((resolve, reject) => {
    pendingAsr.set(id, { resolve, reject });
    worker.postMessage({ type: 'transcribe', id, audio, lang }, [audio.buffer]);
  });
}

// ===== HY-MT1.5 翻譯 worker =====
let mtWorker = null;
let mtReady = null;               // 載入中 / 已載入的 Promise
let mtFailed = false;             // 這次口譯載入失敗，改用 Chrome 內建翻譯
const pendingMt = new Map();
let mtSeq = 0;
const mtContext = [];             // 最近幾句原文，翻譯時當上下文
const mtRecentZh = [];            // 最近幾句譯文，用來刪掉模型重翻上下文的部分

function loadMt() {
  if (mtReady) return mtReady;
  mtWorker = new Worker('mt-worker.js', { type: 'module' });
  let rejectLoad;
  mtReady = new Promise((resolve, reject) => {
    rejectLoad = reject;
    mtWorker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        showProgress(`mt:${m.file}`, m.loaded, m.total, '翻譯');
      } else if (m.type === 'status') {
        setStatus(m.text);
      } else if (m.type === 'loaded') {
        els.progress.hidden = true;
        resolve();
      } else if (m.type === 'result' || m.type === 'error') {
        const p = pendingMt.get(m.id);
        pendingMt.delete(m.id);
        if (p) m.type === 'result' ? p.resolve(m.text) : p.reject(new Error(m.message));
        else if (m.type === 'error') reject(new Error(m.message));
      }
    };
  });
  mtWorker.onerror = (e) => {
    const err = new Error(`翻譯 worker 異常結束${e.message ? `：${e.message}` : '（可能是記憶體不足）'}`);
    rejectLoad(err);
    for (const p of pendingMt.values()) p.reject(err);
    pendingMt.clear();
    unloadMt();
    mtFailed = true; // 之後的句子改用 Chrome 內建翻譯
  };
  mtWorker.postMessage({ type: 'load' });
  mtReady.catch(() => unloadMt());
  return mtReady;
}

// 不用 HY-MT1.5 時把 worker 關掉，釋放約 1.5GB 的顯示記憶體給辨識模型
function unloadMt() {
  mtWorker?.terminate();
  mtWorker = null;
  mtReady = null;
}

function mtTranslate(text, context) {
  const id = ++mtSeq;
  return new Promise((resolve, reject) => {
    pendingMt.set(id, { resolve, reject });
    mtWorker.postMessage({ type: 'translate', id, text, context });
  });
}

const useMt = () => els.engine.value === 'hymt' && !mtFailed && !!mtWorker;

// ===== Chrome 內建翻譯 / 語言偵測 =====
const translators = new Map();    // src -> Promise<Translator>
let detector = null;
const waitingForTranslator = new Map(); // src -> [翻譯單位...]

function langName(code) {
  try { return new Intl.DisplayNames(['zh-Hant'], { type: 'language' }).of(code); } catch { return code; }
}

// withActivation=true 代表目前在使用者點擊的處理流程中，可以觸發模型下載
async function getTranslator(src, withActivation) {
  if (translators.has(src)) return translators.get(src);
  if (!('Translator' in self)) throw new Error('這個 Chrome 版本不支援內建翻譯（需要桌面版 Chrome 138 以上）');
  const avail = await Translator.availability({ sourceLanguage: src, targetLanguage: TARGET });
  if (avail === 'unavailable') throw new Error(`Chrome 內建翻譯不支援 ${langName(src)} → 繁體中文`);
  if (avail !== 'available' && !withActivation) return null; // 需要使用者點一下才能下載
  const p = Translator.create({
    sourceLanguage: src, targetLanguage: TARGET,
    monitor(m) {
      m.addEventListener('downloadprogress', (e) =>
        setStatus(`下載翻譯模型（${langName(src)}→繁中）… ${Math.round(e.loaded * 100)}%`));
    },
  });
  translators.set(src, p);
  p.catch(() => translators.delete(src));
  return p;
}

async function detectLang(text) {
  if (!('LanguageDetector' in self)) return null;
  if (!detector) detector = LanguageDetector.create();
  const d = await detector;
  const [top] = await d.detect(text);
  return top && top.confidence > 0.4 ? top.detectedLanguage.split('-')[0] : null;
}

function showTranslatorButton() {
  const langs = [...waitingForTranslator.keys()];
  if (!langs.length) { els.dl.hidden = true; return; }
  els.dl.hidden = false;
  els.dl.textContent = `點這裡下載「${langs.map(langName).join('、')} → 繁中」翻譯模型（只需一次）`;
}

els.dl.onclick = async () => {
  const entries = [...waitingForTranslator.entries()];
  waitingForTranslator.clear();
  showTranslatorButton();
  for (const [src, list] of entries) {
    try {
      const tr = await getTranslator(src, true);
      for (const unit of list) translateUnit(unit, src, tr);
    } catch (err) {
      for (const unit of list) unit.last.setZh(`（翻譯失敗：${err.message}）`);
    }
  }
};

// 翻譯單位 = 一個「句子」，可能由好幾段音訊的原文合併而成；譯文顯示在最後一段
async function translateUnit(unit, src, tr) {
  const { last } = unit;
  if (src === 'zh') {
    unit.entries.forEach((e) => e.setZh(e.text));
    return;
  }
  const t = performance.now();
  try {
    if (tr) {
      last.setZh(await tr.translate(unit.text));
    } else {
      const context = mtContext.join('\n');
      mtContext.push(unit.text);
      if (mtContext.length > MT_CONTEXT_SENTENCES) mtContext.shift();
      let raw = await mtTranslate(unit.text, context);
      if (context && suspiciousContextOutput(unit.text, raw)) {
        // 把提示詞或上下文翻進來了、或翻成韓文：這句不帶上下文重翻
        const retry = await mtTranslate(unit.text, '');
        console.log('[mt] 帶上下文的譯文可疑，不帶上下文重翻', { raw, retry });
        raw = retry;
      }
      const cleaned = dropHangul(dropPromptEcho(raw), unit.text);
      const zh = dropEchoedContext(cleaned || raw, mtRecentZh);
      if (zh !== raw) console.log('[mt] 刪掉重翻的上下文', { raw, zh });
      mtRecentZh.push(zh);
      if (mtRecentZh.length > MT_CONTEXT_SENTENCES) mtRecentZh.shift();
      last.setZh(zh);
    }
  } catch (err) {
    last.setZh(`（翻譯失敗：${err.message}）`);
  }
  last.timing.translateMs = Math.round(performance.now() - t);
  last.showTiming();
}

// ===== 畫面上的每一段 =====
const transcript = [];

function addEntry(t0, text, timing = {}) {
  els.list.querySelector('.empty')?.remove();
  const div = document.createElement('div');
  div.className = 'item';
  div.innerHTML = '<div class="t"></div><div class="src"></div><div class="zh pending">翻譯中…</div>';
  const tEl = div.querySelector('.t');
  tEl.textContent = fmtTime(t0);
  div.querySelector('.src').textContent = text;
  const zhEl = div.querySelector('.zh');
  const nearBottom = els.list.scrollHeight - els.list.scrollTop - els.list.clientHeight < 80;
  els.list.appendChild(div);
  if (nearBottom) els.list.scrollTop = els.list.scrollHeight;
  const entry = {
    t0, text, zh: '', timing,
    // 在時間旁顯示各階段耗時，方便判斷慢在哪：音訊長度 / 排隊 / 辨識 / 翻譯
    showTiming() {
      const s = (ms) => (ms / 1000).toFixed(1);
      const parts = [`音訊${timing.audioS.toFixed(1)}s`, `排隊${s(timing.waitMs)}s`, `辨識${s(timing.asrMs)}s`];
      if (timing.joinWaitMs != null) parts.push(`等句尾${s(timing.joinWaitMs)}s`);
      if (timing.translateMs != null) parts.push(`翻譯${s(timing.translateMs)}s`);
      tEl.textContent = `${fmtTime(t0)} · ${parts.join(' · ')}`;
      console.log('[timing]', fmtTime(t0), timing);
    },
    setZh(s) {
      entry.zh = s;
      zhEl.textContent = s;
      zhEl.classList.remove('pending');
    },
    setPending(s) { zhEl.textContent = s; zhEl.classList.add('pending'); },
    markGrouped() { div.classList.add('grouped'); },
    // 這段原文已併入下方的整句翻譯
    setMerged() {
      div.classList.add('grouped');
      zhEl.textContent = '↓ 與下方合併成一句翻譯';
      zhEl.classList.add('pending');
    },
  };
  timing.addedAt = performance.now();
  transcript.push(entry);
  return entry;
}

// 被略過的段落（辨識為空、被當成幻覺濾掉、有聲音的部分太短、辨識出錯）：
// 不進逐字稿，只在畫面上留一行淡色提示，才知道「漏掉」是漏在哪一關
function addSkipNote(t0, reason, kind = '') {
  els.list.querySelector('.empty')?.remove();
  const div = document.createElement('div');
  div.className = kind ? `skip ${kind}` : 'skip';
  div.textContent = t0 == null ? reason : `${fmtTime(t0)} · ${reason}`;
  const nearBottom = els.list.scrollHeight - els.list.scrollTop - els.list.clientHeight < 80;
  els.list.appendChild(div);
  if (nearBottom) els.list.scrollTop = els.list.scrollHeight;
  console.log('[skip]', div.textContent); // 沒有時間的提示（空白、跳轉）不要印成 00:00
}

function fmtTime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = Math.floor(s % 60);
  const mmss = `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
  return h ? `${h}:${mmss}` : mmss;
}

// ===== 影片時間 =====
// 顯示的時間是影片的播放時間（讀分頁裡 <video> 的 currentTime），不是開始口譯後經過的時間。
// 讀不到（沒有 video、影片在跨網域 iframe 裡、權限失效）就退回從開始口譯起算的時間。
let videoTabId = null;
let videoWarned = false;
let prevSeg = null;              // 上一段（含被略過的段落）的時間，用來標示空白與跳轉

// 在影片分頁裡執行：挑正在播放的、畫面最大的 video
function pickVideoTime() {
  const vs = [...document.querySelectorAll('video')];
  if (!vs.length) return null;
  const score = (v) => (!v.paused && !v.ended ? 1e12 : 0) + v.videoWidth * v.videoHeight;
  const v = vs.sort((a, b) => score(b) - score(a))[0];
  return { t: v.currentTime, at: Date.now(), paused: v.paused, rate: v.playbackRate || 1 };
}

async function readVideo() {
  if (videoTabId == null) return null;
  try {
    const [r] = await chrome.scripting.executeScript({ target: { tabId: videoTabId }, func: pickVideoTime });
    return r?.result || null;
  } catch (err) {
    if (!videoWarned) { videoWarned = true; console.warn('[video] 讀不到影片時間，改用從開始口譯起算的時間', err); }
    return null;
  }
}

// 切段當下呼叫（越早越準）：回傳 Promise<{ start, end, cap0, cap1, rate, video }>
function segmentTimes(t0, info) {
  const flushWall = Date.now();
  return readVideo().then((read) => {
    const dur = info.end - t0;
    if (!read) return { start: t0, end: info.end, cap0: t0, cap1: info.end, rate: 1, video: false };
    const start = videoStartFrom(read, flushWall, info.now - t0);
    return { start, end: start + dur * (read.rate || 1), cap0: t0, cap1: info.end, rate: read.rate || 1, video: true };
  });
}

// 依序（辨識佇列的順序）呼叫：跟上一段比，有空白或跳轉就留一行提示
function noteGap(times, gapPeak) {
  const prev = prevSeg;
  prevSeg = times;
  if (!prev || prev.video !== times.video) return;
  const g = classifyGap(prev, times);
  if (g.kind === 'seek') {
    // 前轉／後轉：跳轉前後的句子不相干，不要合併成一句翻譯，也不要當成 HY-MT 的上下文
    flushGroup();
    mtContext.length = 0;
    mtRecentZh.length = 0;
    addSkipNote(null, `↪ 影片跳到 ${fmtTime(times.start)}`, 'seek');
  } else if (g.kind === 'gap') {
    const hint = gapPeak >= GAP_QUIET_RATIO
      ? `期間最大音量到門檻的 ${Math.round(gapPeak * 100)}%，可能有小聲說話被當成靜音`
      : '期間都很安靜';
    // 兩端時間只顯示到整秒（無條件捨去），秒數若也取整數會對不上（08～11 卻寫 4 秒），所以顯示到小數一位
    addSkipNote(null, `⋯ ${fmtTime(prev.end)}～${fmtTime(times.start)} ${g.s.toFixed(1)} 秒沒有辨識到聲音（${hint}）`, 'gap');
  }
}

function showEmpty() {
  els.list.innerHTML = '<div class="empty">還沒有內容。播放影片後，大約每 3～8 秒會出現一段。</div>';
}
showEmpty();

// ===== 句子合併：等到句尾再整句翻譯 =====
// 日文常把一句話拆在好幾段音訊裡，單段翻譯會失去上下文。
// 辨識出的原文先立刻顯示，翻譯則等到句尾標點，或「說話者停下來了」才把累積的原文合併送出。
// 「停下來了」不用固定逾時判斷（下一段要錄 3～5 秒再辨識 1～2 秒才會到，固定逾時一定先過期），
// 而是看切段器：目前沒有正在錄的段落、已經安靜夠久、辨識佇列也空了，才算講完。
const MAX_GROUP_CHARS = 80;     // 累積超過這麼多字就不等句尾，直接翻
const MAX_GROUP_SEGS = 4;       // 最多合併幾段
const GROUP_QUIET_S = 1.0;      // 距離最後一個有聲音框至少安靜這麼久，才視為說話者停下來了
const GROUP_MAX_WAIT_MS = 12000; // 保險：從第一段原文出現起，最多等這麼久
const SENTENCE_END = /[。！？!?.…]["'」』）)]*$/;
const CJK = /[぀-ヿ㐀-鿿가-힯]/;

let group = { entries: [], srcSetting: null };
let segmenter = null;           // 目前的切段器；沒在擷取時為 null

function resetGroup() {
  group = { entries: [], srcSetting: null };
}

// 每 250ms 檢查一次：說話者是不是停下來了
function checkGroup() {
  if (!group.entries.length || els.mode.value === 'live') return;
  const waited = performance.now() - group.entries[0].timing.addedAt;
  const st = segmenter && segmenter.status();
  const speechOver = !st || (!st.open && st.quietS >= GROUP_QUIET_S && backlog === 0);
  if (speechOver || waited >= GROUP_MAX_WAIT_MS) flushGroup();
}
setInterval(checkGroup, 250);

function queueForTranslation(entry, srcSetting) {
  group.srcSetting = srcSetting;
  group.entries.push(entry);
  const chars = group.entries.reduce((n, e) => n + e.text.length, 0);
  if (els.mode.value === 'live' || SENTENCE_END.test(entry.text.trim()) ||
      chars >= MAX_GROUP_CHARS || group.entries.length >= MAX_GROUP_SEGS) {
    flushGroup();
  } else {
    group.entries.forEach((e) => e.setPending('等待整句講完再翻譯…'));
  }
}

async function flushGroup() {
  const { entries, srcSetting } = group;
  resetGroup();
  if (!entries.length) return;

  const last = entries[entries.length - 1];
  if (entries.length > 1) {
    entries.slice(0, -1).forEach((e) => e.setMerged());
    last.markGrouped();
  }
  last.setPending('翻譯中…');
  last.timing.joinWaitMs = Math.round(performance.now() - entries[0].timing.addedAt);

  const texts = entries.map((e) => e.text);
  const unit = { entries, last, text: texts.join(CJK.test(texts[0]) ? '' : ' ') };
  try {
    // 自動偵測：辨識模型已經判斷出語言就直接用，不然才用 Chrome 的 LanguageDetector
    const src = srcSetting !== 'auto' ? srcSetting : (last.lang || await detectLang(unit.text));
    if (src === 'zh') return translateUnit(unit, src, null);
    if (useMt()) return translateUnit(unit, src, null); // HY-MT 是多語模型，不需要知道原文語言
    if (!src) { last.setZh('（無法判斷語言）'); return; }

    const tr = await getTranslator(src, false);
    if (tr) return translateUnit(unit, src, tr);
    last.setPending('等待下載翻譯模型…');
    if (!waitingForTranslator.has(src)) waitingForTranslator.set(src, []);
    waitingForTranslator.get(src).push(unit);
    showTranslatorButton();
  } catch (err) {
    last.setZh(`（翻譯失敗：${err.message}）`);
  }
}

// ===== 處理一段語音：辨識 → 排入整句翻譯 =====
let queue = Promise.resolve();
let backlog = 0;
let runNote = '';               // 狀態列上「GPU 加速 / 改用 CPU」的說明，辨識完每段都會重新顯示
let runNoteIsWarn = false;

// 有聲音但太短而被丟掉的段落：也照佇列順序處理，提示才不會插到前面還在辨識的段落之前
function enqueueDrop(t0, voicedS, info) {
  const timesP = segmentTimes(t0, info);
  queue = queue.then(async () => {
    const times = await timesP;
    noteGap(times, info.gapPeak);
    addSkipNote(times.start, `略過：有聲音的部分只有 ${voicedS.toFixed(1)}s，太短`);
  });
}

function enqueueSegment(audio, cap0, srcSetting, info) {
  const timesP = segmentTimes(cap0, info);
  backlog++;
  if (backlog > 3) setStatus(`辨識速度跟不上（積了 ${backlog} 段），可以改用較小的模型`, true);
  const audioS = audio.length / SR; // transcribe 會轉移 buffer，先記下長度
  const tEnq = performance.now();
  queue = queue.then(async () => {
    const times = await timesP;
    const t0 = times.start;
    noteGap(times, info.gapPeak);
    try {
      const tStart = performance.now();
      const res = await transcribe(audio, srcSetting === 'auto' ? null : srcSetting);
      const { lang } = res;
      const text = collapseRepeats(res.text);
      if (text !== res.text) console.log('[asr] 收斂重複片段', res.text.length, '→', text.length, '字');
      const asrMs = Math.round(performance.now() - tStart);
      if (!text) {
        addSkipNote(t0, `略過：辨識結果為空（音訊 ${audioS.toFixed(1)}s，可能太小聲或只有雜音）`);
        return;
      }
      if (HALLUCINATIONS.some((r) => r.test(text))) {
        addSkipNote(t0, `略過：被當成幻覺濾掉「${text}」`);
        return;
      }
      const entry = addEntry(t0, text, { audioS, waitMs: Math.round(tStart - tEnq), asrMs });
      entry.lang = lang; // 辨識模型自己判斷的語言（Qwen3-ASR 自動偵測時才有）
      entry.showTiming();

      if (srcSetting === 'zh') { entry.setZh(text); return; }
      queueForTranslation(entry, srcSetting);
    } catch (err) {
      setStatus(`辨識錯誤：${err.message}`, true);
      addSkipNote(t0, `略過：辨識錯誤 ${err.message}`);
    } finally {
      backlog--;
      if (running && backlog <= 3) setStatus(`口譯中…（${runNote}；側欄關閉就會停止）`, runNoteIsWarn);
    }
  });
}

// ===== 擷取分頁聲音並切段 =====
let running = false;
let stream = null, ctx = null, workletNode = null;

async function start() {
  const srcSetting = els.srcLang.value;
  let model;
  try { model = getModel(); } catch (err) { setStatus(err.message, true); return; }
  if (isCohere(model) && srcSetting === 'auto') {
    setStatus('Cohere Transcribe 不能自動偵測語言，請在「原文語言」選擇影片的語言', true);
    return;
  }
  const engine = els.engine.value;
  els.start.disabled = true;
  setStatus('準備中…');

  // 趁這次點擊還有效，先把翻譯 / 語言偵測模型的下載啟動起來
  // （選 HY-MT1.5 時也準備好 Chrome 內建翻譯，HY-MT1.5 載入失敗時改用）
  try {
    if (srcSetting === 'auto') {
      if ('LanguageDetector' in self) detector = LanguageDetector.create();
    } else if (srcSetting !== 'zh') {
      getTranslator(srcSetting, true).catch((err) => setStatus(err.message, true));
    }
  } catch (err) { setStatus(err.message, true); }

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    let streamId;
    try {
      streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
      videoTabId = tab.id;
    } catch (err) {
      throw new Error('無法擷取這個分頁的聲音。請在影片分頁上「再點一次工具列的擴充功能圖示」，然後再按開始。（' + err.message + '）');
    }

    stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } },
      video: false,
    });

    ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    source.connect(ctx.destination); // 擷取後分頁會靜音，這行把聲音接回喇叭

    setStatus('載入辨識模型…');
    const info = await loadModel(model);

    let mtNote = '';
    mtContext.length = 0; // 不要把上一部影片的句子當成這次的上下文
    mtRecentZh.length = 0;
    if (engine !== 'hymt' || srcSetting === 'zh') unloadMt();
    if (engine === 'hymt' && srcSetting !== 'zh') {
      mtFailed = false;
      fileProgress.clear();
      try {
        await loadMt();
        mtNote = '翻譯：HY-MT1.5';
      } catch (err) {
        mtFailed = true;
        mtNote = 'HY-MT1.5 載入失敗，改用 Chrome 內建翻譯';
        console.warn('[mt] 載入失敗', err);
        setStatus(`${mtNote}（${err.message.slice(0, 120)}）`, true);
      }
    }

    await ctx.audioWorklet.addModule('capture-worklet.js');
    workletNode = new AudioWorkletNode(ctx, 'capture');
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(workletNode).connect(mute).connect(ctx.destination);

    prevSeg = null;
    videoWarned = false;
    const hasVideo = !!(await readVideo());
    segmenter = makeSegmenter(
      ctx.sampleRate,
      (audio, t0, info) => enqueueSegment(audio, t0, srcSetting, info),
      (t0, voicedS, info) => enqueueDrop(t0, voicedS, info),
    );
    workletNode.port.onmessage = (e) => segmenter?.push(e.data);

    stream.getAudioTracks()[0].addEventListener('ended', () => stop());
    running = true;
    els.stop.disabled = false;
    const deviceNote = info.device === 'webgpu' ? 'GPU 加速'
      : info.fellBack ? 'WebGPU 無法執行這個模型，改用 CPU，會慢很多；建議換回內建模型'
      : '沒有 GPU，使用 CPU，會比較慢';
    const timeNote = hasVideo ? '時間：影片時間' : '時間：從開始口譯起算（讀不到影片時間）';
    runNote = [deviceNote, mtNote, timeNote].filter(Boolean).join('；');
    runNoteIsWarn = !!info.fellBack || mtFailed;
    setStatus(`口譯中…（${runNote}；側欄關閉就會停止）`, runNoteIsWarn);
    if (info.fellBack) console.warn('[asr] WebGPU 失敗原因：', info.reason);
  } catch (err) {
    setStatus(err.message, true);
    cleanup();
    els.start.disabled = false;
  }
}

function cleanup() {
  workletNode?.port.close();
  workletNode?.disconnect();
  stream?.getTracks().forEach((t) => t.stop());
  ctx?.close().catch(() => {});
  workletNode = null; stream = null; ctx = null; segmenter = null;
}

function stop() {
  if (!running) return;
  running = false;
  cleanup();
  els.start.disabled = false;
  els.stop.disabled = true;
  setStatus('已停止。還在排隊的段落會繼續處理完。');
  queue = queue.then(() => flushGroup()); // 還沒湊成句的原文也要翻掉
}

// 把原始取樣率降到 16kHz，並依停頓切段
function makeSegmenter(inRate, onSegment, onDrop) {
  const ratio = inRate / SR;
  let frac = 0;
  let frame = new Float32Array(FRAME), fLen = 0;
  let frames = [], rmsArr = [];   // rmsArr 與 frames 一一對應：每個音框的音量
  let voiced = 0, trailingSilent = 0;
  let pre = [];                   // 還沒開始錄段落時，最近幾個安靜的音框（當作開頭留白）
  let totalOut = 0;
  let segStart = 0;
  let quiet = 0;                  // 距離最後一個有聲音框已經過了幾個音框（不論有沒有在錄段落）
  let gapPeak = 0;                // 段落之間的空白裡，最大音量 ÷ 門檻（看有沒有小聲說話被當成靜音）
  let segGapPeak = 0;             // 目前這段開始前那段空白的 gapPeak

  let threshold = SILENCE_RMS;    // 目前的有聲音門檻（隨底噪調整）
  let softThreshold = SILENCE_RMS; // 低於門檻、但有音高時的門檻
  let softVoiced = 0;             // 靠音高補判成有聲音的音框數（log 用）
  let bgPeriodicity = 1;          // 背景音框的平均週期性；一開始當成音樂，確認背景不是音樂之後才補判
  const recent = [];              // 最近幾個音框的音量，用來估底噪
  let frameCount = 0;
  const isLoud = (r) => r > threshold;

  function updateThreshold(r) {
    recent.push(r);
    if (recent.length > NOISE_WINDOW_FRAMES) recent.shift();
    if (recent.length < NOISE_WARMUP_FRAMES) return;
    const sorted = [...recent].sort((a, b) => a - b);
    const noise = sorted[Math.floor(NOISE_PERCENTILE * (sorted.length - 1))];
    threshold = Math.min(Math.max(MAX_THRESHOLD, noise * HIGH_FLOOR_MULT), Math.max(MIN_THRESHOLD, noise * NOISE_MULT));
    softThreshold = Math.min(threshold, Math.max(MIN_THRESHOLD, noise * SOFT_NOISE_MULT));
    if (++frameCount % 200 === 0) console.log(`[vad] 底噪 ${noise.toFixed(4)} → 門檻 ${threshold.toFixed(4)}（自動增益 ×${agc.gain.toFixed(1)}；背景週期性 ${bgPeriodicity.toFixed(2)}${bgPeriodicity < MUSIC_PERIODICITY ? '' : '（像音樂，不補判）'}，靠音高補判 ${softVoiced} 個音框）`);
  }

  // 把說話部分的音量拉到 TARGET_RMS 左右（取 90 百分位，不被突波影響）；只放大
  function gainFor() {
    const v = rmsArr.filter(isLoud).sort((a, b) => a - b);
    if (!v.length) return 1;
    const p90 = v[Math.floor(0.9 * (v.length - 1))];
    return Math.min(MAX_GAIN, Math.max(1, TARGET_RMS / p90));
  }

  function flush() {
    // end：這段結束的擷取時間；now：目前擷取到的時間；gapPeak：這段之前的空白裡，最大音量是門檻的幾倍
    const info = { end: segStart + frames.length / 10, now: totalOut / SR, gapPeak: segGapPeak };
    // 有聲音的部分很短時，看音高決定要不要送：0.2 秒要有一個音框有人聲音高，0.1 秒（只有一個音框）要音高非常明顯
    const short = frames.length && voiced * 0.1 < MIN_VOICED_S;
    const bestPitch = short ? Math.max(0, ...frames.map((f, i) => (isLoud(rmsArr[i]) ? periodicity(f, SR) : 0))) : 0;
    const keep = frames.length && (!short ||
      (voiced * 0.1 >= MIN_PITCHED_VOICED_S && bestPitch >= VOICED_PERIODICITY) ||
      (voiced >= 1 && bestPitch >= STRONG_PERIODICITY));
    if (keep) {
      const g = gainFor();
      const audio = new Float32Array(frames.length * FRAME);
      frames.forEach((f, i) => audio.set(f, i * FRAME));
      if (g > 1) for (let i = 0; i < audio.length; i++) audio[i] = Math.max(-1, Math.min(1, audio[i] * g));
      onSegment(audio, segStart, info);
    } else if (frames.length && voiced * 0.1 >= MIN_NOTE_VOICED_S) {
      onDrop?.(segStart, voiced * 0.1, info);
    } else if (frames.length) {
      // 太短、連提示都不留的聲音：不要讓它把空白切斷，否則空白提示會只看後半段、誤寫成「期間都很安靜」
      const peak = Math.max(...rmsArr) / threshold;
      gapPeak = Math.max(segGapPeak, peak);
      console.log(`[vad] 丟掉很短的聲音：有聲音 ${(voiced * 0.1).toFixed(1)}s、音量是門檻的 ${Math.round(peak * 100)}%、音高 ${bestPitch.toFixed(2)}`);
    }
    frames = []; rmsArr = []; voiced = 0; trailingSilent = 0; pre = [];
  }

  // 累積到上限仍沒遇到停頓（快語速）：回頭找最安靜的音框切，後面的音訊留給下一段，不丟失
  function cutAtDip() {
    const from = Math.min(frames.length - 1, Math.floor(CUT_SEARCH_FROM_S * 10));
    let idx = from;
    for (let i = from; i < frames.length; i++) if (rmsArr[i] <= rmsArr[idx]) idx = i;

    const tailF = frames.slice(idx + 1), tailR = rmsArr.slice(idx + 1);
    const nextStart = segStart + (idx + 1) / 10;
    frames = frames.slice(0, idx + 1); rmsArr = rmsArr.slice(0, idx + 1);
    voiced = rmsArr.filter(isLoud).length;
    flush();

    const tailVoiced = tailR.filter(isLoud).length;
    if (!tailVoiced) {              // 剩下的全是安靜：不開新段，只留作下一段的開頭留白
      pre = tailF.slice(-PRE_ROLL_FRAMES).map((f, i, a) => ({ f, r: tailR[tailR.length - a.length + i] }));
      return;
    }
    frames = tailF; rmsArr = tailR; voiced = tailVoiced; segStart = nextStart;
    segGapPeak = 0; // 緊接著上一段，中間沒有空白
    trailingSilent = 0;
    for (let i = tailR.length - 1; i >= 0 && !isLoud(tailR[i]); i--) trailingSilent++;
  }

  const agc = makeAgc({ target: TARGET_RMS, maxGain: INPUT_MAX_GAIN });

  function onFrame(f) {
    agc.process(f); // 先把很小聲的影片拉到正常音量，再判斷有沒有人講話
    let sum = 0;
    for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
    let r = Math.sqrt(sum / f.length);
    updateThreshold(r);
    // 沒到門檻但有人聲的音高：當成剛好超過門檻（這個值也用來算送辨識前要放大幾倍）
    if (r <= threshold) {
      const p = periodicity(f, SR);
      if (r <= softThreshold) bgPeriodicity += (p - bgPeriodicity) * 0.05; // 背景：約 2 秒的移動平均
      else if (p >= VOICED_PERIODICITY && bgPeriodicity < MUSIC_PERIODICITY) {
        r = threshold * 1.01;
        softVoiced++;
      }
    }
    const loud = isLoud(r);
    quiet = loud ? 0 : quiet + 1;

    if (!frames.length) {
      if (!loud) {
        gapPeak = Math.max(gapPeak, r / threshold);                 // 段落從有聲音的地方開始；先記下最近幾個安靜音框當開頭留白
        pre.push({ f, r });
        if (pre.length > PRE_ROLL_FRAMES) pre.shift();
        return;
      }
      segGapPeak = gapPeak;
      gapPeak = 0;
      frames = pre.map((p) => p.f); rmsArr = pre.map((p) => p.r);
      segStart = Math.max(0, totalOut / SR - (frames.length + 1) / 10);
      pre = [];
    }
    frames.push(f); rmsArr.push(r);
    if (loud) { voiced++; trailingSilent = 0; } else trailingSilent++;

    const dur = frames.length / 10;
    const trailS = trailingSilent / 10;
    if ((dur >= MIN_SEG_S && trailS >= END_SILENCE_S) ||
        (dur >= LONG_SEG_S && trailS >= LONG_END_SILENCE_S)) flush();
    else if (dur >= MAX_SEG_S) cutAtDip();
  }

  return {
    // open：正有一段在錄（說話者還在講）；quietS：最後一次有聲音到現在的秒數
    status() { return { open: frames.length > 0, quietS: quiet / 10, threshold }; },
    push(chunk) {
      // 簡單平均降取樣
      let pos = frac;
      while (pos + ratio <= chunk.length) {
        const a = Math.max(0, Math.floor(pos)), b = Math.min(chunk.length, Math.floor(pos + ratio));
        let s = 0;
        for (let i = a; i < b; i++) s += chunk[i];
        frame[fLen++] = s / Math.max(1, b - a);
        totalOut++;
        if (fLen === FRAME) { onFrame(frame); frame = new Float32Array(FRAME); fLen = 0; }
        pos += ratio;
      }
      frac = pos - chunk.length;
    },
  };
}

// ===== 按鈕 =====
els.start.onclick = start;
els.stop.onclick = stop;
els.clear.onclick = () => { prevSeg = null; transcript.length = 0; mtContext.length = 0; mtRecentZh.length = 0; resetGroup(); showEmpty(); };
els.exp.onclick = () => {
  if (!transcript.length) return;
  const text = transcript.map((e) => `[${fmtTime(e.t0)}]\n${e.text}\n${e.zh ? e.zh + '\n' : ''}`).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
  a.download = `口譯逐字稿-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '')}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
