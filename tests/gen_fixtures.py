"""產生 tests/fixtures/*.json：用 numpy 實作的參考答案，給 Node 測試比對。
執行：python3 tests/gen_fixtures.py（需要 numpy）
"""
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# ---- log-mel：跟 Whisper / Qwen3-ASR 前處理一樣（torch.stft center=True reflect、週期 Hann、丟最後一格）----
sr = 16000
t = np.arange(int(0.3 * sr)) / sr
audio = (0.5 * np.sin(2 * np.pi * (200 + 1500 * t) * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
n_mels = 8
filters = np.abs(rng.standard_normal((n_mels, 201))).astype(np.float32) * 0.01

x = np.pad(audio.astype(np.float64), 200, mode="reflect")
win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
frames = 1 + (x.size - 400) // 160
spec = np.stack([np.fft.rfft(x[i * 160:i * 160 + 400] * win) for i in range(frames)], axis=1)
power = np.abs(spec[:, :-1]) ** 2
mel = filters.astype(np.float64) @ power
log = np.log10(np.maximum(mel, 1e-10))
log = np.maximum(log, log.max() - 8.0)
log = (log + 4.0) / 4.0

# ---- float16 轉換 ----
vals = np.concatenate([
    rng.standard_normal(200) * 10, rng.standard_normal(50) * 1e-5, rng.standard_normal(50) * 1e-7,
    [0.0, -0.0, 1.0, -2.5, 65504.0, 70000.0, -1e9, 6.1e-5, 5.96e-8, 3e-8, 1e-9],
]).astype(np.float32)
f16 = vals.astype(np.float16).view(np.uint16)

with open(os.path.join(here, "fixtures", "reference.json"), "w") as f:
    json.dump({
        "audio": audio.tolist(),
        "filters": {"n_mels": n_mels, "n_freqs": 201, "data": filters.tolist()},
        "T": int(log.shape[1]),
        "logmel": log.astype(np.float32).ravel().tolist(),
        "f32": vals.tolist(),
        "f16bits": f16.tolist(),
    }, f)
print("ok", log.shape)
