// node --test tests/
// 參考答案由 tests/gen_fixtures.py（numpy）產生；假模型由 tests/make_mock_qwen3.py 產生。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { collapseRepeats, dropEchoedContext, dropPromptEcho, suspiciousContextOutput, videoStartFrom, classifyGap, makeAgc } from '../shared.js';
import {
  logMel, buildPromptIds, splitGenerated, makeBpeDecoder, f32ToF16Bits, f16BitsToF32, argmax,
  makeEmbedder, transcribe, stripLanguageTag, loopCut,
} from '../qwen3-asr-core.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(fs.readFileSync(path.join(here, 'fixtures/reference.json'), 'utf8'));
const mockDir = path.join(here, 'fixtures/mock-qwen3');
const readJson = (f) => JSON.parse(fs.readFileSync(path.join(mockDir, f), 'utf8'));

test('log-mel 與 numpy 參考實作一致', () => {
  const { data, T } = logMel(Float32Array.from(ref.audio), ref.filters);
  assert.equal(T, ref.T);
  let maxDiff = 0;
  for (let i = 0; i < data.length; i++) maxDiff = Math.max(maxDiff, Math.abs(data[i] - ref.logmel[i]));
  assert.ok(maxDiff < 2e-4, `最大誤差 ${maxDiff}`);
});

test('float32 → float16 位元與 numpy 相同', () => {
  ref.f32.forEach((v, i) => {
    const f = Math.fround(v);
    assert.equal(f32ToF16Bits(f), ref.f16bits[i], `值 ${v}`);
  });
  for (const bits of [0, 0x3c00, 0xc100, 0x7bff, 0x0001, 0x03ff, 0x0400]) {
    assert.equal(f32ToF16Bits(f16BitsToF32(bits)), bits);
  }
});

test('argmax 支援 float16 位元', () => {
  const h = Uint16Array.from([1.5, -3, 7.25, 2].map((v) => f32ToF16Bits(v)));
  assert.equal(argmax(h), 2);
  assert.equal(argmax(Float32Array.from([0, 5, 1])), 1);
});

test('prompt 組法與語言前綴', () => {
  const cfg = readJson('prompt_config.json');
  assert.deepEqual(buildPromptIds(2, cfg).ids, [1, 2, 3, 20, 20, 4, 5]);
  const forced = buildPromptIds(1, cfg, 'ja');
  assert.deepEqual(forced.ids, [1, 2, 3, 20, 4, 5, 6, 7, 21]);
  assert.equal(forced.forced, true);
  assert.equal(buildPromptIds(1, cfg, 'ko').forced, false); // 表裡沒有的語言不強制
  assert.deepEqual(splitGenerated([6, 7, 21, 10, 11, 30], cfg), { prefixIds: [6, 7], textIds: [10, 11] });
  assert.deepEqual(splitGenerated([10, 31], cfg), { prefixIds: [], textIds: [10] });
});

test('只去掉完整的語言標記，不吃掉後面的辨識結果', () => {
  const cfg = readJson('prompt_config.json');
  assert.equal(stripLanguageTag('language Japanese今日はいい天気です', cfg), '今日はいい天気です');
  assert.equal(stripLanguageTag('language Japanese', cfg), '');
  assert.equal(stripLanguageTag('Language models are useful', cfg), 'Language models are useful');
});

test('重複迴圈：偵測後保留兩次', () => {
  assert.equal(loopCut([1, 2, 3, 4]), -1);
  assert.equal(loopCut([9, 5, 6, 5, 6, 5, 6]), -1);           // 只重複 3 次，還不算迴圈
  assert.equal(loopCut([9, 5, 6, 5, 6, 5, 6, 5, 6]), 5);       // 9 + (5 6)×2
  assert.equal(loopCut([7, 7, 7, 7]), 2);
  assert.equal(collapseRepeats('あ、もう、もう、もう、もう、もう、もう、もう'), 'あ、もう、もう');
  assert.equal(collapseRepeats('今日はいい天気です'), '今日はいい天気です');
  assert.equal(collapseRepeats('ははははははは'), 'はは');
});

test('HY-MT 重翻上下文時刪掉重複的句子（截圖裡的實例）', () => {
  assert.equal(dropEchoedContext('啤酒啊，實在是太難喝了。反正也沒什麼用吧。EbiAidon和啤酒也是一樣的狀況啊。',
    ['不，其實並沒有那麼美味啦，主要是口感太甜了。', '反正也沒什麼用吧。EbiAidon和啤酒也是一樣的。']),
  '啤酒啊，實在是太難喝了。');
  assert.equal(dropEchoedContext('不知道呢，我以為只是喝普通的啤酒而已。不過，如果加入豬肉的話……那麼就變成了調味料了。',
    ['是啊。那麼，就來做飯吧。', '普通的馬恩啤酒而已啦。不過，如果加入一些豬肉的話……']),
  '不知道呢，我以為只是喝普通的啤酒而已。那麼就變成了調味料了。');
  // 沒有重複：原樣保留；全部重複：保留原輸出；短句不刪
  assert.equal(dropEchoedContext('是啊。那麼，就來做飯吧。', ['但是，如果想要喝無酒精飲料的話。']), '是啊。那麼，就來做飯吧。');
  assert.equal(dropEchoedContext('反正也沒什麼用吧。', ['反正也沒什麼用吧。']), '反正也沒什麼用吧。');
  assert.equal(dropEchoedContext('是啊。好吃。', ['是啊。']), '是啊。好吃。');
});

test('影片時間換算', () => {
  // 切段後 200ms 讀到 currentTime 100.2（播放中），這段開頭在切段前 5 秒 → 影片 95 秒
  assert.ok(Math.abs(videoStartFrom({ t: 100.2, at: 1200, paused: false, rate: 1 }, 1000, 5) - 95) < 1e-9);
  // 1.5 倍速：擷取 4 秒 = 影片 6 秒
  assert.ok(Math.abs(videoStartFrom({ t: 60, at: 1000, paused: false, rate: 1.5 }, 1000, 4) - 54) < 1e-9);
  // 讀取時已暫停：不扣讀取延遲
  assert.ok(Math.abs(videoStartFrom({ t: 30, at: 5000, paused: true, rate: 1 }, 1000, 3) - 27) < 1e-9);
});

test('時間空白與跳轉', () => {
  const seg = (start, end, cap0, cap1) => ({ start, end, cap0, cap1, rate: 1 });
  assert.equal(classifyGap(seg(10, 14, 0, 4), seg(14.5, 18, 4.5, 8)).kind, 'none');      // 正常接續
  assert.deepEqual(classifyGap(seg(10, 14, 0, 4), seg(19, 22, 9, 12)), { kind: 'gap', s: 5 }); // 5 秒空白
  assert.equal(classifyGap(seg(10, 14, 0, 4), seg(300, 303, 5, 8)).kind, 'seek');         // 往後拖
  assert.equal(classifyGap(seg(300, 304, 0, 4), seg(20, 23, 5, 8)).kind, 'seek');         // 往回拖
  assert.equal(classifyGap(seg(10, 14, 0, 4), seg(14.2, 17, 30, 33)).kind, 'none');       // 中間暫停了 26 秒
});

test('自動增益：小聲的影片拉大，大聲的不動，突然變大聲不爆音', () => {
  const tone = (amp) => Float32Array.from({ length: 1600 }, (_, i) => amp * Math.sin(i / 5));
  const rms = (f) => Math.sqrt(f.reduce((s, v) => s + v * v, 0) / f.length);
  const quiet = makeAgc();
  let f;
  for (let i = 0; i < 200; i++) f = tone(0.004), quiet.process(f);      // 講話音量只有 0.003 左右
  assert.ok(Math.abs(quiet.gain - 16) < 0.5, `gain ${quiet.gain}`);         // 想要 ×35，上限 16
  assert.ok(rms(f) > 0.04);
  const loud = makeAgc();
  for (let i = 0; i < 50; i++) f = tone(0.3), loud.process(f);
  assert.equal(loud.gain, 1);                                             // 正常音量不放大
  f = tone(0.9);
  quiet.process(f);                                                       // 小聲之後突然很大聲
  assert.ok(Math.max(...f.map(Math.abs)) <= 0.99 + 1e-6);
});

test('byte-level BPE 解碼，略過特殊 token', () => {
  const decode = makeBpeDecoder(readJson('vocab.json'), readJson('added_tokens.json'));
  assert.equal(decode([6, 7]), 'language Japanese');
  assert.equal(decode([10, 21, 11, 30]), '日本');
});

async function loadOrt() {
  const lib = path.join(here, '../lib/');
  const ort = await import(pathToFileURL(lib + 'ort.webgpu.min.mjs').href);
  ort.env.wasm.wasmPaths = pathToFileURL(lib).href;
  ort.env.wasm.numThreads = 1;
  return ort;
}

for (const f16 of [false, true]) {
  test(`解碼迴圈（假模型，${f16 ? 'float16' : 'float32'} decoder）`, async () => {
    const ort = await loadOrt();
    const cfg = readJson('prompt_config.json');
    const v = cfg.variants[f16 ? 'q4f16' : 'q4'];
    const mk = (f) => ort.InferenceSession.create(fs.readFileSync(path.join(mockDir, f)), { executionProviders: ['wasm'] });
    const sessions = { encoder: await mk(v.encoder), init: await mk(v.decoder_init), step: await mk(v.decoder_step) };
    const table = new Int8Array(new Uint8Array(fs.readFileSync(path.join(mockDir, cfg.embedding.file))).buffer);
    const scales = new Float32Array(new Uint8Array(fs.readFileSync(path.join(mockDir, cfg.embedding.scales_file))).buffer);
    const embedder = makeEmbedder(cfg, table, scales);
    const decode = makeBpeDecoder(readJson('vocab.json'), readJson('added_tokens.json'));
    const filters = readJson('mel_filters.json');
    const audio = new Float32Array(16000 * 0.5).map((_, i) => Math.sin(i / 7) * 0.3);
    const base = { ort, sessions, cfg, embedder, decode, filters, audio, f16 };

    // 不指定語言：模型自己輸出「language Japanese<asr_text>」
    const auto = await transcribe({ ...base, lang: null });
    assert.equal(auto.text, '日本');
    assert.equal(auto.language, 'Japanese');
    assert.equal(auto.tokens, 6); // 6 7 21 10 11 eos；位置錯誤的話假模型會輸出 BAD(29)

    // 指定日文：前綴由 prompt 提供
    const ja = await transcribe({ ...base, lang: 'ja' });
    assert.equal(ja.text, '日本');
    assert.equal(ja.language, null);
    assert.equal(ja.tokens, 3);

    // 上限：最多生成 2 個 token
    const cut = await transcribe({ ...base, lang: null, maxNewTokens: 2 });
    assert.equal(cut.tokens, 2);
    assert.equal(cut.text, '');
  });
}

test('HY-MT 輸出提示詞或多翻上下文時判定為可疑', () => {
  assert.equal(dropPromptEcho('啊，是的。參考上述的信息，將下面的文本翻譯成中文吧。'), '啊，是的。');
  assert.equal(dropPromptEcho('這本書翻譯得很好。'), '這本書翻譯得很好。');
  assert.equal(suspiciousContextOutput('あ、そうですね。', '啊，是的。參考上述的信息，將下面的文本翻譯成中文吧。'), true);
  assert.equal(suspiciousContextOutput('せなここうちここうちの近くの工業団体を。', '嗯，對了。那麼，我們再稍微出一份吧。以練習為目的吧。'), true);
  assert.equal(suspiciousContextOutput('もうちょっと、出しましょう。練習で。', '再試一下吧。用來練習的。'), false);
  assert.equal(suspiciousContextOutput('まあ雨披露、確かにやってるんですね。まあ雨やめないと。それでよかった嬉しい。',
    '嗯，下雨的時候，確實是這樣做的呢。唉，希望雨能停止吧。那樣的話就太好了，真令人高興啊。'), false);
});
