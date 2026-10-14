// node --test tests/
// 參考答案由 tests/gen_fixtures.py（numpy）產生；假模型由 tests/make_mock_qwen3.py 產生。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { collapseRepeats } from '../shared.js';
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
