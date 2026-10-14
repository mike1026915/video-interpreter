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
