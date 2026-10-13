"""產生 tests/fixtures/mock-qwen3/：跟 Qwen3-ASR layout v2 一樣的輸入輸出介面，但只有幾個小運算的假模型。
用來在 Node 裡測 qwen3-asr-core.js 的解碼迴圈（嵌入、音訊特徵插入、KV cache、位置、停止條件、語言前綴）。
執行：python3 tests/make_mock_qwen3.py（需要 numpy、onnx）

設計：詞彙 32、hidden 32，嵌入表每列是 one-hot，所以「最後一個輸入 token」可以從嵌入讀出來；
logits = 最後一列嵌入 × 轉移矩陣，形成固定的接龍：5→6→7→21→10→11→30(eos)。
step 會檢查 position_ids 是否等於 KV cache 長度，不對就輸出 token 29（BAD），測試因此能抓到位置錯誤。
"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "fixtures", "mock-qwen3")
os.makedirs(out, exist_ok=True)

V = H = 32
L, KVH, HD = 2, 1, 4
NEXT = {5: 6, 6: 7, 7: 21, 21: 10, 10: 11, 11: 30}
M = np.zeros((H, V), np.float32)
for a, b in NEXT.items():
    M[a, b] = 1.0
BAD = 29
bad_logits = np.zeros((1, 1, V), np.float32)
bad_logits[0, 0, BAD] = 10.0


def save(model, name):
    model.opset_import[0].version = 17
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(out, name))


def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


# encoder：mel [1,8,T] → audio_features [1,T,32]（轉置後重複 4 次）
enc = helper.make_graph(
    [helper.make_node("Transpose", ["mel"], ["t"], perm=[0, 2, 1]),
     helper.make_node("Concat", ["t", "t", "t", "t"], ["audio_features"], axis=2)],
    "enc",
    [helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, 8, "T"])],
    [helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, [1, "T", H])],
)
save(helper.make_model(enc), "encoder.onnx")


def decoder(step, f16):
    ft = TensorProto.FLOAT16 if f16 else TensorProto.FLOAT
    nodes, inits = [], [c("M", M), c("bad", bad_logits), c("idx_last", np.array([-1], np.int64)),
                         c("end", np.array([2**62], np.int64)), c("ax1", np.array([1], np.int64)),
                         c("f0", np.array([0], np.int64)), c("f4", np.array([4], np.int64)),
                         c("ax2", np.array([2], np.int64)), c("kvshape_tail", np.array([L, 1, KVH], np.int64)),
                         c("i3", np.array([3], np.int64)), c("i4", np.array([4], np.int64)),
                         c("u01", np.array([0, 1], np.int64))]
    x = "input_embeds"
    if f16:
        nodes.append(helper.make_node("Cast", [x], ["x32"], to=TensorProto.FLOAT))
        x = "x32"
    # 最後一列 → logits
    nodes += [helper.make_node("Slice", [x, "idx_last", "end", "ax1"], ["last"]),
              helper.make_node("MatMul", ["last", "M"], ["lg"])]
    # 新的 KV：前 4 維特徵 → [1,1,n,4] → expand 成 [L,1,KVH,n,4]
    nodes += [helper.make_node("Slice", [x, "f0", "f4", "ax2"], ["k0"]),
              helper.make_node("Unsqueeze", [ "k0", "u01"], ["k1"]),
              helper.make_node("Shape", ["k1"], ["k1s"]),
              helper.make_node("Slice", ["k1s", "i3", "end", "f0"], ["tail"]),
              helper.make_node("Shape", ["k0"], ["k0s"]),
              helper.make_node("Slice", ["k0s", "ax1", "ax2", "f0"], ["nlen"]),
              helper.make_node("Concat", ["kvshape_tail", "nlen", "f4"], ["kvshape"], axis=0),
              helper.make_node("Expand", ["k1", "kvshape"], ["newkv"])]
    inputs = [helper.make_tensor_value_info("input_embeds", ft, [1, "S", H]),
              helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, "S"])]
    if step:
        inputs += [helper.make_tensor_value_info("past_keys", ft, [L, 1, KVH, "P", HD]),
                   helper.make_tensor_value_info("past_values", ft, [L, 1, KVH, "P", HD])]
        pk, pv = "past_keys", "past_values"
        if f16:
            nodes += [helper.make_node("Cast", [pk], ["pk32"], to=TensorProto.FLOAT),
                      helper.make_node("Cast", [pv], ["pv32"], to=TensorProto.FLOAT)]
            pk, pv = "pk32", "pv32"
        nodes += [helper.make_node("Concat", [pk, "newkv"], ["pk_out"], axis=3),
                  helper.make_node("Concat", [pv, "newkv"], ["pv_out"], axis=3),
                  # 位置檢查：position_ids 應該等於 past 長度
                  helper.make_node("Shape", [pk], ["pks"]),
                  helper.make_node("Slice", ["pks", "i3", "i4", "f0"], ["plen"]),
                  helper.make_node("Reshape", ["position_ids", "ax1"], ["pos1"]),
                  helper.make_node("Equal", ["pos1", "plen"], ["ok"]),
                  helper.make_node("Where", ["ok", "lg", "bad"], ["lg2"])]
        lg = "lg2"
    else:
        nodes += [helper.make_node("Identity", ["newkv"], ["pk_out"]),
                  helper.make_node("Identity", ["newkv"], ["pv_out"])]
        lg = "lg"
    if f16:
        nodes += [helper.make_node("Cast", [lg], ["logits"], to=TensorProto.FLOAT16),
                  helper.make_node("Cast", ["pk_out"], ["present_keys"], to=TensorProto.FLOAT16),
                  helper.make_node("Cast", ["pv_out"], ["present_values"], to=TensorProto.FLOAT16)]
    else:
        nodes += [helper.make_node("Identity", [lg], ["logits"]),
                  helper.make_node("Identity", ["pk_out"], ["present_keys"]),
                  helper.make_node("Identity", ["pv_out"], ["present_values"])]
    outputs = [helper.make_tensor_value_info("logits", ft, [1, 1, V]),
               helper.make_tensor_value_info("present_keys", ft, [L, 1, KVH, "N", HD]),
               helper.make_tensor_value_info("present_values", ft, [L, 1, KVH, "N", HD])]
    g = helper.make_graph(nodes, "dec", inputs, outputs, inits)
    return helper.make_model(g)


for f16 in (False, True):
    sfx = "q4f16" if f16 else "int4"
    save(decoder(False, f16), f"decoder_init.{sfx}.onnx")
    save(decoder(True, f16), f"decoder_step.{sfx}.onnx")

# 嵌入表：int8 100 × scale 0.01 = one-hot
emb = (np.eye(V, H) * 100).astype(np.int8)
emb.tofile(os.path.join(out, "embed_tokens.int8.bin"))
np.full(V, 0.01, np.float32).tofile(os.path.join(out, "embed_scales.f32.bin"))


def bl(s):  # byte-level 編碼（GPT-2 對照表）
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = dict(zip(bs, cs))
    return "".join(chr(m[b]) for b in s.encode("utf-8"))


vocab = {f"tok{i}": i for i in range(V)}
vocab.update({bl("language"): 6, bl(" Japanese"): 7, bl("日"): 10, bl("本"): 11})
for k in [k for k, v in vocab.items() if v in (6, 7, 10, 11) and k.startswith("tok")]:
    del vocab[k]
added = {"<|audio_pad|>": 20, "<asr_text>": 21, "<|endoftext|>": 30, "<|im_end|>": 31, "<|im_start|>": 1}
with open(os.path.join(out, "vocab.json"), "w") as f:
    json.dump(vocab, f, ensure_ascii=False)
with open(os.path.join(out, "added_tokens.json"), "w") as f:
    json.dump(added, f)

cfg = {
    "layout_version": 2,
    "mel": {"n_mels": 8, "filters_file": "mel_filters.json"},
    "prompt": {"prefix_ids": [1, 2, 3], "suffix_ids": [4, 5], "audio_pad_id": 20, "asr_text_id": 21,
               "eos_ids": [30, 31], "max_new_tokens": 16},
    "language_prefix_ids": {"ja": [6, 7, 21]},
    "language_names": {"ja": "Japanese", "en": "English"},
    "embedding": {"file": "embed_tokens.int8.bin", "dtype": "int8", "shape": [V, H], "scales_file": "embed_scales.f32.bin"},
    "decoder": {"num_layers": L, "num_key_value_heads": KVH, "head_dim": HD, "hidden_size": H, "vocab_size": V},
    "variants": {
        "q4": {"encoder": "encoder.onnx", "decoder_init": "decoder_init.int4.onnx", "decoder_step": "decoder_step.int4.onnx",
               "weights": None, "required_features": []},
        "q4f16": {"encoder": "encoder.onnx", "decoder_init": "decoder_init.q4f16.onnx", "decoder_step": "decoder_step.q4f16.onnx",
                  "weights": None, "required_features": ["shader-f16"]},
    },
}
with open(os.path.join(out, "prompt_config.json"), "w") as f:
    json.dump(cfg, f, indent=1)
rng = np.random.default_rng(1)
with open(os.path.join(out, "mel_filters.json"), "w") as f:
    json.dump({"n_mels": 8, "n_freqs": 201, "data": (np.abs(rng.standard_normal((8, 201))) * 0.01).round(6).tolist()}, f)
print("ok", sorted(os.listdir(out)))
