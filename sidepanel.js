// ===== 設定 =====
const TARGET = 'zh-Hant';
const SR = 16000;               // Whisper 需要 16kHz
const FRAME = SR / 10;          // 100ms 一個音框，用來判斷有沒有人說話
const SILENCE_RMS = 0.008;      // 低於這個音量視為靜音
const MIN_SEG_S = 2;            // 一段至少 2 秒才切
const MAX_SEG_S = 10;           // 最長 10 秒強制切
const END_SILENCE_S = 0.5;      // 講完停頓 0.5 秒就切段
const MIN_VOICED_S = 0.5;       // 有聲音的部分不到 0.5 秒就丟掉（避免 Whisper 對靜音亂講）

const WHISPER_LANG = {
  en: 'english', ja: 'japanese', ko: 'korean', fr: 'french',
  de: 'german', es: 'spanish', zh: 'chinese',
};

// 靜音或音樂時 Whisper 常見的幻覺句
const HALLUCINATIONS = [
  /^thank(s| you)( for watching)?[.!]?$/i, /^you$/i, /^\.+$/, /^ご視聴ありがとうございました/,
  /^字幕/, /^\[.*\]$/, /^\(.*\)$/, /^♪+$/,
];

// ===== DOM =====
const $ = (id) => document.getElementById(id);
const els = {
  srcLang: $('srcLang'), model: $('model'), start: $('startBtn'), stop: $('stopBtn'),
  exp: $('exportBtn'), clear: $('clearBtn'), status: $('status'), progress: $('progress'),
  dl: $('dlTranslatorBtn'), list: $('list'),
};

function setStatus(text, isErr = false) {
  els.status.textContent = text;
  els.status.classList.toggle('err', isErr);
}

// ===== 設定記憶 =====
chrome.storage.local.get(['srcLang', 'model']).then((s) => {
  if (s.srcLang) els.srcLang.value = s.srcLang;
  if (s.model) els.model.value = s.model;
});
els.srcLang.onchange = () => chrome.storage.local.set({ srcLang: els.srcLang.value });
els.model.onchange = () => chrome.storage.local.set({ model: els.model.value });

// ===== Whisper worker =====
const worker = new Worker('asr-worker.js', { type: 'module' });
let workerReady = null;
const pendingAsr = new Map();
const fileProgress = new Map();

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'progress') {
    fileProgress.set(m.file, [m.loaded, m.total]);
    let l = 0, t = 0;
    for (const [a, b] of fileProgress.values()) { l += a; t += b; }
    els.progress.hidden = false;
    els.progress.firstElementChild.style.width = `${(l / t) * 100}%`;
    setStatus(`下載辨識模型中… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB（只有第一次需要）`);
  } else if (m.type === 'loaded') {
    els.progress.hidden = true;
    workerReady?.resolve(m);
  } else if (m.type === 'result' || m.type === 'error') {
    if (m.type === 'error' && workerReady && !workerReady.done) {
      workerReady.reject(new Error(m.message));
      return;
    }
    const p = pendingAsr.get(m.id);
    pendingAsr.delete(m.id);
    if (!p) return;
    m.type === 'result' ? p.resolve(m.text) : p.reject(new Error(m.message));
  }
};

function loadModel(model) {
  fileProgress.clear();
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  workerReady = {
    promise,
    resolve: (v) => { workerReady.done = true; resolve(v); },
    reject: (e) => { workerReady.done = true; reject(e); },
  };
  worker.postMessage({ type: 'load', model });
  return promise;
}

let asrSeq = 0;
function transcribe(audio, language) {
  const id = ++asrSeq;
  return new Promise((resolve, reject) => {
    pendingAsr.set(id, { resolve, reject });
    worker.postMessage({ type: 'transcribe', id, audio, language }, [audio.buffer]);
  });
}

// ===== Chrome 內建翻譯 / 語言偵測 =====
const translators = new Map();    // src -> Promise<Translator>
let detector = null;
const waitingForTranslator = new Map(); // src -> [entry...]

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
      for (const entry of list) translateEntry(entry, src, tr);
    } catch (err) {
      for (const entry of list) entry.setZh(`（翻譯失敗：${err.message}）`);
    }
  }
};

async function translateEntry(entry, src, tr) {
  const t = performance.now();
  try {
    entry.setZh(await tr.translate(entry.text));
  } catch (err) {
    entry.setZh(`（翻譯失敗：${err.message}）`);
  }
  entry.timing.translateMs = Math.round(performance.now() - t);
  entry.showTiming();
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
  };
  transcript.push(entry);
  return entry;
}

function fmtTime(s) {
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function showEmpty() {
  els.list.innerHTML = '<div class="empty">還沒有內容。播放影片後，大約每 2～10 秒會出現一段。</div>';
}
showEmpty();

// ===== 處理一段語音：辨識 → 翻譯 =====
let queue = Promise.resolve();
let backlog = 0;

function enqueueSegment(audio, t0, srcSetting) {
  backlog++;
  if (backlog > 3) setStatus(`辨識速度跟不上（積了 ${backlog} 段），可以改用較小的模型`, true);
  const audioS = audio.length / SR; // transcribe 會轉移 buffer，先記下長度
  const tEnq = performance.now();
  queue = queue.then(async () => {
    try {
      const tStart = performance.now();
      const text = await transcribe(audio, WHISPER_LANG[srcSetting] || null);
      const asrMs = Math.round(performance.now() - tStart);
      if (!text || HALLUCINATIONS.some((r) => r.test(text))) return;
      const entry = addEntry(t0, text, { audioS, waitMs: Math.round(tStart - tEnq), asrMs });
      entry.showTiming();

      let src = srcSetting === 'auto' ? await detectLang(text) : srcSetting;
      if (!src) { entry.setZh('（無法判斷語言）'); return; }
      if (src === 'zh') { entry.setZh(text); return; }

      const tr = await getTranslator(src, false);
      if (tr) return translateEntry(entry, src, tr);
      entry.setPending('等待下載翻譯模型…');
      if (!waitingForTranslator.has(src)) waitingForTranslator.set(src, []);
      waitingForTranslator.get(src).push(entry);
      showTranslatorButton();
    } catch (err) {
      setStatus(`辨識錯誤：${err.message}`, true);
    } finally {
      backlog--;
      if (running && backlog <= 3) setStatus('口譯中…（側欄關閉就會停止）');
    }
  });
}

// ===== 擷取分頁聲音並切段 =====
let running = false;
let stream = null, ctx = null, workletNode = null;

async function start() {
  const srcSetting = els.srcLang.value;
  const model = els.model.value;
  els.start.disabled = true;
  setStatus('準備中…');

  // 趁這次點擊還有效，先把翻譯 / 語言偵測模型的下載啟動起來
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

    await ctx.audioWorklet.addModule('capture-worklet.js');
    workletNode = new AudioWorkletNode(ctx, 'capture');
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(workletNode).connect(mute).connect(ctx.destination);

    const seg = makeSegmenter(ctx.sampleRate, (audio, t0) => enqueueSegment(audio, t0, srcSetting));
    workletNode.port.onmessage = (e) => seg.push(e.data);

    stream.getAudioTracks()[0].addEventListener('ended', () => stop());
    running = true;
    els.stop.disabled = false;
    setStatus(`口譯中…（${info.device === 'webgpu' ? 'GPU 加速' : '沒有 GPU，使用 CPU，會比較慢'}；側欄關閉就會停止）`);
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
  workletNode = null; stream = null; ctx = null;
}

function stop() {
  if (!running) return;
  running = false;
  cleanup();
  els.start.disabled = false;
  els.stop.disabled = true;
  setStatus('已停止。還在排隊的段落會繼續處理完。');
}

// 把原始取樣率降到 16kHz，並依停頓切段
function makeSegmenter(inRate, onSegment) {
  const ratio = inRate / SR;
  let frac = 0;
  let frame = new Float32Array(FRAME), fLen = 0;
  let frames = [], voiced = 0, trailingSilent = 0;
  let totalOut = 0;
  let segStart = 0;

  function flush() {
    if (frames.length && voiced * 0.1 >= MIN_VOICED_S) {
      const audio = new Float32Array(frames.length * FRAME);
      frames.forEach((f, i) => audio.set(f, i * FRAME));
      onSegment(audio, segStart);
    }
    frames = []; voiced = 0; trailingSilent = 0;
  }

  function onFrame(f) {
    let sum = 0;
    for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
    const loud = Math.sqrt(sum / f.length) > SILENCE_RMS;

    if (!frames.length) {
      if (!loud) return;           // 段落從有聲音的地方開始
      segStart = totalOut / SR;
    }
    frames.push(f);
    if (loud) { voiced++; trailingSilent = 0; } else trailingSilent++;

    const dur = frames.length / 10;
    if ((dur >= MIN_SEG_S && trailingSilent / 10 >= END_SILENCE_S) || dur >= MAX_SEG_S) flush();
  }

  return {
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
els.clear.onclick = () => { transcript.length = 0; showEmpty(); };
els.exp.onclick = () => {
  if (!transcript.length) return;
  const text = transcript.map((e) => `[${fmtTime(e.t0)}]\n${e.text}\n${e.zh}\n`).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
  a.download = `口譯逐字稿-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '')}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
