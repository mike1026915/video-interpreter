// Qwen3-ASR（layout v2 ONNX，見 jiangzhuo9357/Qwen3-ASR-*-ONNX 的模型卡）的純運算部分。
// 不碰網路、不碰 WebGPU，所以可以直接在 Node 測試；worker 只負責下載檔案、建立 session。
//
// 流程：音訊 → Whisper 相容的 log-mel → encoder → 把音訊特徵塞進 prompt 的 <|audio_pad|> 位置
//       → decoder_init 一次 → decoder_step 逐字貪婪解碼 → 去掉「language X<asr_text>」→ BPE 解碼成文字

// ===== log-mel：16kHz、n_fft 400、hop 160、週期 Hann 窗、置中反射補邊、功率譜、log10、clamp 到 max−8 =====
const N_FFT = 400;
const HOP = 160;
const N_FREQ = N_FFT / 2 + 1;

// 400 點不是 2 的次方，用混合基數 FFT（400 = 4·4·5·5），比直接 DFT 快約 10 倍
const FACTORS = [4, 4, 5, 5];
let fftTables = null;
function getFftTables() {
  if (fftTables) return fftTables;
  const cos = new Float64Array(N_FFT), sin = new Float64Array(N_FFT);
  for (let t = 0; t < N_FFT; t++) {
    cos[t] = Math.cos((2 * Math.PI * t) / N_FFT);
    sin[t] = -Math.sin((2 * Math.PI * t) / N_FFT);
  }
  const win = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  // 每一層遞迴各自的暫存區，避免每個音框重新配置
  const scratch = FACTORS.map(() => ({ re: new Float64Array(N_FFT), im: new Float64Array(N_FFT) }));
  fftTables = { cos, sin, win, scratch };
  return fftTables;
}

// 把 x[off], x[off+stride], …（共 n 點）的 DFT 寫到 outRe/outIm[o .. o+n)
function fftRec(xRe, xIm, off, stride, n, outRe, outIm, o, level) {
  if (n === 1) { outRe[o] = xRe[off]; outIm[o] = xIm[off]; return; }
  const { cos, sin, scratch } = fftTables;
  const p = FACTORS[level], m = n / p;
  const { re: yRe, im: yIm } = scratch[level];
  // 子序列 j：x[off + stride*(j + p*q)]，DFT 結果放在 y[o + j*m ..]
  for (let j = 0; j < p; j++) fftRec(xRe, xIm, off + stride * j, stride * p, m, yRe, yIm, o + j * m, level + 1);
  const step = N_FFT / n; // W_n^u = W_N^(u*step)
  for (let k = 0; k < m; k++) {
    for (let s = 0; s < p; s++) {
      const kk = k + s * m;
      let re = 0, im = 0;
      for (let j = 0; j < p; j++) {
        const t = ((j * kk) % n) * step;
        const a = yRe[o + j * m + k], b = yIm[o + j * m + k];
        re += a * cos[t] - b * sin[t];
        im += a * sin[t] + b * cos[t];
      }
      outRe[o + kk] = re;
      outIm[o + kk] = im;
    }
  }
}

// filters：{ n_mels, n_freqs, data: number[n_mels][n_freqs] }（模型附的 mel_filters.json）
// 回傳 { data: Float32Array(n_mels * T)，排列為 [n_mels][T]，T }
export function logMel(audio, filters) {
  const { win } = getFftTables();
  const nMels = filters.n_mels;
  if (filters.n_freqs !== N_FREQ) throw new Error(`mel_filters 的頻率數 ${filters.n_freqs} 不是 ${N_FREQ}`);
  const fb = new Float32Array(nMels * N_FREQ);
  filters.data.forEach((row, m) => fb.set(row, m * N_FREQ));
  // 每個 mel 濾波器只有一小段頻率不是 0，只算那一段
  const lo = new Int32Array(nMels), hi = new Int32Array(nMels);
  for (let m = 0; m < nMels; m++) {
    let a = 0, b = N_FREQ;
    while (a < N_FREQ && fb[m * N_FREQ + a] === 0) a++;
    while (b > a && fb[m * N_FREQ + b - 1] === 0) b--;
    lo[m] = a; hi[m] = b;
  }

  // 置中：前後各反射補 n_fft/2
  const pad = N_FFT / 2;
  const n = audio.length;
  if (n < pad + 1) throw new Error('音訊太短，無法計算 log-mel');
  const padded = new Float32Array(n + 2 * pad);
  padded.set(audio, pad);
  for (let i = 0; i < pad; i++) {
    padded[pad - 1 - i] = audio[i + 1];
    padded[pad + n + i] = audio[n - 2 - i];
  }
  const T = Math.floor(n / HOP); // 1 + floor(n / hop) 個音框，最後一個丟掉

  const out = new Float32Array(nMels * T);
  const frame = new Float64Array(N_FFT);
  const zeros = new Float64Array(N_FFT);
  const specRe = new Float64Array(N_FFT), specIm = new Float64Array(N_FFT);
  const power = new Float32Array(N_FREQ);
  let maxVal = -Infinity;
  for (let t = 0; t < T; t++) {
    const off = t * HOP;
    for (let i = 0; i < N_FFT; i++) frame[i] = padded[off + i] * win[i];
    fftRec(frame, zeros, 0, 1, N_FFT, specRe, specIm, 0, 0);
    for (let k = 0; k < N_FREQ; k++) power[k] = specRe[k] * specRe[k] + specIm[k] * specIm[k];
    for (let m = 0; m < nMels; m++) {
      let s = 0;
      const base = m * N_FREQ;
      for (let k = lo[m]; k < hi[m]; k++) s += fb[base + k] * power[k];
      const v = Math.log10(Math.max(s, 1e-10));
      out[m * T + t] = v;
      if (v > maxVal) maxVal = v;
    }
  }
  const floor = maxVal - 8;
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + 4) / 4;
  return { data: out, T };
}

// ===== prompt =====
// prompt_config.json：prompt.prefix_ids + A 個 audio_pad + prompt.suffix_ids（+ 指定語言時的 language_prefix_ids）
// contextIds：背景資訊的 token（前文、人名、影片標題…）。Qwen3-ASR 把它放在 system 訊息裡，
// 辨識時會參考（同音詞、專有名詞比較準）。prefix 是「<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>」，
// 背景資訊插在 system 的 <|im_end|> 前面；prefix 裡找不到 <|im_end|>（eos_ids 之一）就不放。
export function buildPromptIds(numAudio, cfg, lang, contextIds = []) {
  const p = cfg.prompt;
  const forcedIds = lang ? cfg.language_prefix_ids?.[lang] : null;
  const end = contextIds.length ? p.prefix_ids.findIndex((id) => p.eos_ids.includes(id)) : -1;
  const prefix = end < 0 ? p.prefix_ids : [...p.prefix_ids.slice(0, end), ...contextIds, ...p.prefix_ids.slice(end)];
  const ids = [...prefix, ...new Array(numAudio).fill(p.audio_pad_id), ...p.suffix_ids, ...(forcedIds || [])];
  return { ids, audioStart: prefix.length, forced: !!forcedIds, context: end >= 0 };
}

// 生成的 token：「language Japanese<asr_text>文字…<eos>」。切掉 <asr_text> 之前的部分和結尾的 eos。
export function splitGenerated(gen, cfg) {
  const eos = new Set(cfg.prompt.eos_ids);
  const ids = gen.filter((t) => !eos.has(t));
  const cut = ids.lastIndexOf(cfg.prompt.asr_text_id);
  return cut < 0 ? { prefixIds: [], textIds: ids } : { prefixIds: ids.slice(0, cut), textIds: ids.slice(cut + 1) };
}

export function stripLanguageTag(text, cfg) {
  const names = Object.values(cfg.language_names || {}).sort((a, b) => b.length - a.length);
  for (const name of names) {
    const tag = `language ${name}`;
    if (text.toLowerCase().startsWith(tag.toLowerCase())) return text.slice(tag.length).trim();
  }
  return text;
}

// ===== byte-level BPE 解碼（Qwen 的 tokenizer 跟 GPT-2 一樣把每個位元組對應到一個可見字元）=====
function byteDecoderTable() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const map = new Map();
  bs.forEach((b, i) => map.set(String.fromCodePoint(cs[i]), b));
  return map;
}

// vocab：vocab.json（token → id）；added：added_tokens.json（特殊 token → id，解碼時略過）
export function makeBpeDecoder(vocab, added = {}) {
  const idToToken = [];
  for (const [tok, id] of Object.entries(vocab)) idToToken[id] = tok;
  const special = new Set(Object.values(added));
  const byteOf = byteDecoderTable();
  const utf8 = new TextEncoder();
  const dec = new TextDecoder('utf-8');
  return (ids) => {
    const bytes = [];
    for (const id of ids) {
      if (special.has(id)) continue;
      const tok = idToToken[id];
      if (tok == null) continue;
      for (const ch of tok) {
        const b = byteOf.get(ch);
        if (b != null) bytes.push(b);
        else bytes.push(...utf8.encode(ch));
      }
    }
    return dec.decode(new Uint8Array(bytes));
  };
}

// ===== byte-level BPE 編碼（只用來把「背景資訊」轉成 token 當提示）=====
// Qwen 的 BPE 是從 tiktoken 來的：合併的優先順序就是合併後 token 的 id（越小越先合併），
// 所以只要 vocab.json，不需要 11MB 的 tokenizer.json。切詞規則（pre-tokenize）跟 Qwen2 tokenizer 一樣，並先做 NFC。
const PRETOKENIZE = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
export function makeBpeEncoder(vocab) {
  const charOf = [];
  for (const [ch, b] of byteDecoderTable()) charOf[b] = ch;
  const utf8 = new TextEncoder();
  const rank = (tok) => vocab[tok] ?? Infinity;
  function bpe(chunk) {
    let parts = [...utf8.encode(chunk)].map((b) => charOf[b]);
    while (parts.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = rank(parts[i] + parts[i + 1]);
        if (r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      parts = [...parts.slice(0, at), parts[at] + parts[at + 1], ...parts.slice(at + 2)];
    }
    return parts.map((p) => vocab[p]).filter((id) => id != null);
  }
  return (text) => (text.normalize('NFC').match(PRETOKENIZE) || []).flatMap(bpe);
}

// ===== float16 =====
const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
export function f32ToF16Bits(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / NaN
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >>> shift;
    if ((mant >>> (shift - 1)) & 1 && ((mant & ((1 << (shift - 1)) - 1)) || (h & 1))) h++;
    return sign | h;
  }
  let h = sign | (exp << 10) | (mant >>> 13);
  if ((mant & 0x1000) && ((mant & 0x2fff) || (h & 1))) h++; // 四捨六入五成雙
  return h;
}
export function f16BitsToF32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}
export function toF16(f32) {
  const out = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) out[i] = f32ToF16Bits(f32[i]);
  return out;
}

// logits 可能是 Float32Array、Float16Array，或以 Uint16Array 存的 float16 位元
export function argmax(data) {
  let best = 0, bestV = -Infinity;
  if (data instanceof Uint16Array) {
    for (let i = 0; i < data.length; i++) {
      const v = f16BitsToF32(data[i]);
      if (v > bestV) { bestV = v; best = i; }
    }
  } else {
    for (let i = 0; i < data.length; i++) if (data[i] > bestV) { bestV = data[i]; best = i; }
  }
  return best;
}

// ===== 嵌入表：int8 × 每列 scale =====
export function makeEmbedder(cfg, table, scales) {
  const [vocab, hidden] = cfg.embedding.shape;
  const dtype = cfg.embedding.dtype;
  return {
    hidden,
    // 把第 id 列寫進 out[offset .. offset+hidden)
    row(id, out, offset) {
      if (id < 0 || id >= vocab) throw new Error(`token id 超出範圍：${id}`);
      const base = id * hidden;
      if (dtype === 'int8') {
        const s = scales[id];
        for (let j = 0; j < hidden; j++) out[offset + j] = table[base + j] * s;
      } else if (dtype === 'float16') {
        for (let j = 0; j < hidden; j++) out[offset + j] = f16BitsToF32(table[base + j]);
      } else {
        for (let j = 0; j < hidden; j++) out[offset + j] = table[base + j];
      }
    },
  };
}

// ===== 重複迴圈偵測 =====
// 量化模型遇到模糊音訊時，常卡在一小段 token 上一直重複（例如「もう、もう、もう、…」）直到上限。
// 結尾若是同一段（長度 1～maxPeriod）連續出現 minRepeats 次，回傳去掉多餘重複後應保留的長度；否則回傳 -1。
export function loopCut(ids, minRepeats = 4, maxPeriod = 12) {
  for (let p = 1; p <= maxPeriod; p++) {
    const need = p * minRepeats;
    if (ids.length < need) break;
    const end = ids.length;
    let ok = true;
    for (let i = end - need; i < end - p && ok; i++) if (ids[i] !== ids[i + p]) ok = false;
    if (ok) return end - need + p * 2; // 保留兩次（例如「もう、もう、」），比只留一次更接近原話
  }
  return -1;
}

// ===== 整段辨識 =====
// ort：onnxruntime-web 模組；sessions：{ encoder, init, step }；f16：decoder 是否用 float16 I/O
// 回傳 { text, language（模型自己判斷的語言名稱，有指定語言時為 null）, tokens }
export async function transcribe({ ort, sessions, cfg, embedder, decode, filters, audio, lang, f16, maxNewTokens, contextIds = [] }) {
  const mel = logMel(audio, filters);
  const encOut = await sessions.encoder.run({
    mel: new ort.Tensor('float32', mel.data, [1, filters.n_mels, mel.T]),
  });
  const feats = encOut.audio_features;
  const [, A, H] = feats.dims;
  if (H !== embedder.hidden) throw new Error(`encoder 輸出寬度 ${H} 與嵌入表 ${embedder.hidden} 不符`);
  const featData = feats.data;

  const { ids, audioStart, forced } = buildPromptIds(A, cfg, lang, contextIds);
  const S = ids.length;
  const x = new Float32Array(S * H);
  for (let i = 0; i < S; i++) {
    if (i >= audioStart && i < audioStart + A) continue;
    embedder.row(ids[i], x, i * H);
  }
  x.set(featData.subarray(0, A * H), audioStart * H);
  feats.dispose?.();

  const embTensor = (f32, n) => (f16
    ? new ort.Tensor('float16', toF16(f32), [1, n, H])
    : new ort.Tensor('float32', f32, [1, n, H]));
  const posTensor = (start, n) => {
    const p = new BigInt64Array(n);
    for (let i = 0; i < n; i++) p[i] = BigInt(start + i);
    return new ort.Tensor('int64', p, [1, n]);
  };

  const limit = Math.max(1, Math.min(maxNewTokens || cfg.prompt.max_new_tokens, cfg.prompt.max_new_tokens));
  const eos = new Set(cfg.prompt.eos_ids);
  let out = await sessions.init.run({ input_embeds: embTensor(x, S), position_ids: posTensor(0, S) });
  const gen = [];
  const row = new Float32Array(H);
  let pos = S;
  try {
    for (;;) {
      const next = argmax(out.logits.data);
      out.logits.dispose?.();
      gen.push(next);
      if (eos.has(next) || gen.length >= limit) break;
      const cut = loopCut(gen);
      if (cut >= 0) { gen.length = cut; break; } // 卡在重複迴圈：提早停，也省下後面的解碼時間
      embedder.row(next, row, 0);
      const prev = out;
      out = await sessions.step.run({
        input_embeds: embTensor(row, 1),
        position_ids: posTensor(pos, 1),
        past_keys: prev.present_keys,
        past_values: prev.present_values,
      });
      prev.present_keys.dispose?.();
      prev.present_values.dispose?.();
      pos++;
    }
  } finally {
    out.present_keys?.dispose?.();
    out.present_values?.dispose?.();
  }

  const { prefixIds, textIds } = splitGenerated(gen, cfg);
  const language = !forced && prefixIds.length ? decode(prefixIds).replace(/^language\s*/i, '').trim() : null;
  let text = decode(textIds).trim();
  // 沒指定語言、輸出裡又沒有 <asr_text>（到達上限，或量化模型漏掉標記）：
  // 只去掉開頭完整的「language <已知語言名稱>」，後面的辨識結果保留
  if (!forced && !prefixIds.length) text = stripLanguageTag(text, cfg);
  return { text, language, tokens: gen.length };
}
