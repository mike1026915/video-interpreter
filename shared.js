// 側欄和各個 worker 共用的設定與小工具（module，worker 和頁面都能 import）

// ONNX Runtime 的 wasm 檔。transformers.min.js 和 ort.webgpu.min.mjs 是同一版 ORT，共用這一組。
// transformers.js 4.x 預設會從 CDN 載入；MV3 不能載入遠端程式碼，所以一律指向本地 lib/。升級 ORT 只要改這裡。
const LIB = new URL('./lib/', import.meta.url).href;
export const ORT_WASM_PATHS = {
  mjs: LIB + 'ort-wasm-simd-threaded.asyncify.mjs',
  wasm: LIB + 'ort-wasm-simd-threaded.asyncify.wasm',
};

// 依模型 ID 判斷要用哪條路徑
export const isQwen3Asr = (model) => /qwen3-asr/i.test(model);       // qwen3-asr-worker.js（onnxruntime-web）
export const isCohere = (model) => /cohere-transcribe/i.test(model);  // asr-worker.js，只能 WebGPU、要指定語言

// 不管哪個模型，連續重複 4 次以上的片段（1～12 個字）都收斂成 2 次，翻譯也就不會跟著重複
export function collapseRepeats(text) {
  return text.replace(/(.{1,12}?)\1{3,}/gu, '$1$1');
}

// HY-MT1.5 有時會把當作上下文的前一句也翻進來，例如「這一句的譯文。上一句的譯文。」
// 把譯文切成句子，跟前幾句的譯文幾乎相同的句子刪掉；全部都被刪掉時保留原輸出。
const PUNCT = /[\s、。，,.！!？?…「」『』（）()"'：:；;~～\-—]/g;
const splitSentences = (s) => s.match(/[^。！？!?…]+[。！？!?…]*|[。！？!?…]+/g) || [s];
function bigrams(s) {
  const m = new Map();
  for (let i = 0; i < s.length - 1; i++) { const b = s.slice(i, i + 2); m.set(b, (m.get(b) || 0) + 1); }
  return m;
}
function dice(a, b) {
  if (a.length < 2 || b.length < 2) return a === b ? 1 : 0;
  const A = bigrams(a), B = bigrams(b);
  let inter = 0;
  for (const [k, n] of A) inter += Math.min(n, B.get(k) || 0);
  return (2 * inter) / (a.length - 1 + b.length - 1);
}
export function dropEchoedContext(text, previous) {
  const prev = previous.flatMap(splitSentences).map((s) => s.replace(PUNCT, '')).filter((s) => s.length >= 4);
  if (!prev.length) return text;
  const kept = splitSentences(text).filter((s) => {
    const n = s.replace(PUNCT, '');
    if (n.length < 4) return true; // 太短的（「是啊」）可能真的重複說了，不刪
    return !prev.some((p) => p === n || p.includes(n) || n.includes(p) || dice(n, p) >= 0.7);
  });
  const out = kept.join('').trim();
  return out || text;
}

// HY-MT1.5 偶爾會把提示詞本身也輸出，例如「參考上述的信息，將下面的文本翻譯成中文吧。」
// 同時提到「翻譯」和提示詞裡的字眼（文本、上文、上述、信息…）的句子就刪掉；全部都被刪掉時回傳空字串。
const PROMPT_ECHO = /(翻譯|翻译).*(文本|上文|上述|上面|信息|資訊|资讯|繁體|繁体|解釋|解释)|(文本|上文|上述|上面|信息|資訊|资讯|繁體|繁体|解釋|解释).*(翻譯|翻译)/;
export function dropPromptEcho(text) {
  return splitSentences(text).filter((s) => !PROMPT_ECHO.test(s)).join('').trim();
}

// 帶上下文翻譯的結果看起來可疑，要不帶上下文重翻一次：
// 有提示詞，或是譯文的句數比原文多 2 句以上（多出來的通常是把上下文換句話說又翻了一次）
// 數句子時不把「…」當句尾：「每晚都會喝…每晚都會喝…」是一句話裡的停頓，不是多翻了句子
const countSentences = (s) => (s.match(/[^。！？!?]+[。！？!?]*/g) || []).filter((x) => x.replace(PUNCT, '').length).length;
export function suspiciousContextOutput(source, zh) {
  return dropPromptEcho(zh) !== zh.trim() || countSentences(zh) > countSentences(source) + 1 ||
    dropHangul(zh, source) !== zh.trim();
}

// HY-MT1.5 偶爾會翻成韓文（「そうか。」→「그렇군요。」）：原文沒有韓文時，含韓文的句子刪掉
const HANGUL = /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/;
export function dropHangul(text, source) {
  if (HANGUL.test(source)) return text.trim();
  return splitSentences(text).filter((s) => !HANGUL.test(s)).join('').trim();
}

// Qwen3-ASR 帶背景資訊時，聽不清楚的地方偶爾會直接照抄：前 2 段辨識結果（recent），
// 或使用者填的背景資訊、影片標題（fixed；可能是別的語言，例如中文敘述被複述成日文）。
// 辨識結果裡有一句（6 個字以上；「ありがとう」這種短句可能真的又說了一次）符合下面任一條就當成照抄（呼叫端會不帶背景資訊重新辨識）：
//  - 跟前文或 fixed 的某一小段幾乎一樣（包含或雙字組相似度 ≥ 0.8）
//  - 對 fixed：句子裡非平假名的字（漢字、片假名、英數）至少 6 個，而且 90% 以上都出現在 fixed 裡
//    （只拿 6 個字以上的敘述片段來比，短關鍵字不算；所以講出關鍵字、甚至兩個關鍵字湊一句都不會誤判）
const KANA_HIRA = /[\u3040-\u309f]/g;
export function echoesContext(text, recent = [], fixed = '') {
  const norm = (s) => s.replace(PUNCT, '');
  const prev = recent.flatMap(splitSentences).map(norm).filter((s) => s.length >= 6);
  const fixedParts = fixed ? fixed.split(/[\s。！？!?…、，,]+/).map(norm).filter((s) => s.length >= 6) : [];
  const fixedChars = new Set(fixedParts.join('').replace(KANA_HIRA, ''));   // 短關鍵字（人名、作品名）不算進來
  const like = (n, list) => list.some((p) => p.includes(n) || n.includes(p) || dice(n, p) >= 0.8);
  return splitSentences(text).some((s) => {
    const n = norm(s);
    if (n.length < 6) return false;
    if (like(n, prev) || like(n, fixedParts)) return true;
    const content = [...n.replace(KANA_HIRA, '')];
    return content.length >= 6 && content.filter((c) => fixedChars.has(c)).length / content.length >= 0.9;
  });
}

// 講話中途硬切時，下一段開頭會重複帶上一段結尾的 0.3 秒，辨識結果開頭可能重複上一段最後幾個字。
// 新結果的開頭（2～maxChars 個字，忽略標點）跟上一段的結尾相同，就刪掉；找不到就原樣保留。
const EDGE_PUNCT = /[\s、。，,.！!？?…「」『』（）()~～ー-]+/g;
export function dropOverlap(prev, text, maxChars = 8) {
  const tail = prev.replace(EDGE_PUNCT, '').slice(-maxChars);
  if (!tail) return text;
  // 新結果的前幾個「非標點字」各在原字串的哪個位置
  const pos = [];
  for (let i = 0; i < text.length && pos.length < maxChars; i++) if (!/[\s、。，,.！!？?…「」『』（）()~～ー-]/.test(text[i])) pos.push(i);
  const plain = pos.map((i) => text[i]).join('');
  for (let k = Math.min(maxChars, plain.length, tail.length); k >= 2; k--) {
    if (tail.endsWith(plain.slice(0, k))) {
      const rest = text.slice(pos[k - 1] + 1).replace(/^[\s、。，,.！!？?…]+/, '');
      return rest || text;
    }
  }
  return text;
}

// ===== 自動增益（只用在送去辨識的那一路，喇叭聽到的音量不變）=====
// 很小聲的影片：講話的音量可能比靜音門檻的下限還低，整段被當成靜音。
// 所以在判斷有沒有人講話之前，先依最近 10 秒的音量把聲音整體拉大。
// - 用最近 10 秒音框音量的 95 百分位（大致是講話時的音量）估計需要放大幾倍，最多 maxGain 倍、只放大不縮小
// - 要放大的倍數變小時立刻降（突然變大聲不會爆音），變大時慢慢升（約 2 秒）
// - 每個音框再以峰值限制，放大後不超過 0.99
export function makeAgc({ target = 0.1, maxGain = 16, windowFrames = 100, rise = 0.05 } = {}) {
  const recent = [];
  let gain = 1;
  return {
    get gain() { return gain; },
    // frame：原始音框（會被就地放大）；回傳這個音框實際用的倍數
    process(frame) {
      let sum = 0, peak = 0;
      for (let i = 0; i < frame.length; i++) {
        const v = frame[i];
        sum += v * v;
        const a = v < 0 ? -v : v;
        if (a > peak) peak = a;
      }
      recent.push(Math.sqrt(sum / frame.length));
      if (recent.length > windowFrames) recent.shift();
      const sorted = [...recent].sort((a, b) => a - b);
      const p95 = sorted[Math.floor(0.95 * (sorted.length - 1))];
      const want = p95 > 0 ? Math.min(maxGain, Math.max(1, target / p95)) : gain;
      gain = want < gain ? want : gain + (want - gain) * rise;
      const g = peak > 0 ? Math.min(gain, 0.99 / peak) : gain;
      if (g > 1) for (let i = 0; i < frame.length; i++) frame[i] *= g;
      return g;
    },
  };
}

// ===== 人聲的週期性 =====
// 有聲的說話（母音）是週期性的：基頻 70～400Hz，波形每隔一個基頻週期就重複一次。
// 底噪、風聲、雜音沒有這種重複。回傳 0～1：音框在這個範圍內最像「隔一段時間重複」的程度（正規化自相關的最大值）。
// 小聲說話跟背景差不多大時，光看音量分不出來，靠這個補判。
export function periodicity(frame, sr = 16000, minHz = 70, maxHz = 400) {
  const n = Math.min(frame.length, 800); // 50ms 就夠看出週期，省一半計算
  let mean = 0;
  for (let i = 0; i < n; i++) mean += frame[i];
  mean /= n;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = frame[i] - mean;
  const minLag = Math.floor(sr / maxHz), maxLag = Math.ceil(sr / minHz);
  let best = 0;
  for (let lag = minLag; lag <= maxLag && lag < n / 2; lag++) {
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i + lag < n; i++) {
      const a = x[i], b = x[i + lag];
      xy += a * b; xx += a * a; yy += b * b;
    }
    if (xx > 0 && yy > 0) best = Math.max(best, xy / Math.sqrt(xx * yy));
  }
  return best;
}

// ===== 影片時間 =====
// read：從影片分頁讀到的 { t: currentTime, at: 讀取當下的 Date.now(), paused, rate }
// flushWall：切段當下的 Date.now()；backS：這段的開頭在切段當下之前幾秒（擷取時間）
// 回傳這段開頭在影片裡的時間（秒）
export function videoStartFrom(read, flushWall, backS) {
  const rate = read.rate || 1;
  const sinceFlush = read.paused ? 0 : ((read.at - flushWall) / 1000) * rate;
  return Math.max(0, read.t - sinceFlush - backS * rate);
}

// 前後兩段之間：kind = 'seek'（拖了進度條）、'gap'（連續播放中有一段沒辨識到聲音，s 秒）、'none'
// prev / cur：{ start, end（影片時間或擷取時間）, cap0, cap1（擷取時間）, rate }
export const GAP_NOTE_S = 2;
export function classifyGap(prev, cur) {
  const gap = cur.start - prev.end;
  const capGap = cur.cap0 - prev.cap1;
  if (gap < -1) return { kind: 'seek' };                                  // 往回跳
  if (gap > capGap * (cur.rate || 1) + 2) return { kind: 'seek' };        // 影片時間走得比實際播放還多：往後跳
  if (gap >= GAP_NOTE_S) return { kind: 'gap', s: gap };                 // 暫停不會讓影片時間前進，所以不會被誤判
  return { kind: 'none' };
}

// adapter：WebGPU adapter（沒有則為 null）；ok：有 WebGPU；
// f16：支援 shader-f16（Linux 的 NVIDIA 等 Vulkan 環境常常沒有，fp16 模型會跑不動）
export async function gpuInfo() {
  try {
    const gpu = globalThis.navigator?.gpu;
    const adapter = gpu ? await gpu.requestAdapter({ powerPreference: 'high-performance' }) : null;
    return { adapter, ok: !!adapter, f16: !!adapter?.features.has('shader-f16') };
  } catch {
    return { adapter: null, ok: false, f16: false };
  }
}
