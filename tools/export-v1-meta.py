"""只產生 andrewleech/qwen3-asr-onnx 的 export.py 在匯出 decoder 之後才做的中繼檔：
embed_tokens.bin（FP32 嵌入表）、tokenizer、config.json、preprocessor_config.json。

為什麼不直接用 export.py 匯出 decoder：layout v2 的 decoder 由 export_v2.py 重新匯出，v1 的 decoder 用不到；
而且 export.py 匯出完 decoder_step 後會 onnx.load 約 7GB 的外部資料檢查能不能內嵌，在 15GB 記憶體的機器上會被 OOM 砍掉
（實測 1.7B：python 用到約 13.9GB 被砍）。

用法：QWEN3_ASR_ONNX_DIR=<qwen3-asr-onnx clone> python export-v1-meta.py <來源模型> <輸出目錄>
"""
import os
import sys

sys.path.insert(0, os.environ["QWEN3_ASR_ONNX_DIR"])
os.chdir(os.environ["QWEN3_ASR_ONNX_DIR"])
import torch  # noqa: E402
from export import copy_tokenizer, load_model, verify_special_tokens, write_config, write_preprocessor_config  # noqa: E402

model_id, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
model = load_model(model_id, dtype=torch.float32)
verify_special_tokens(model_id)
w = model.thinker.model.embed_tokens.weight.data.cpu().float().numpy()
w.tofile(os.path.join(out, "embed_tokens.bin"))
print("embed_tokens.bin", w.shape)
copy_tokenizer(model_id, out)
write_config(model, out)
write_preprocessor_config(out)
