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
