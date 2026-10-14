#!/usr/bin/env bash
# 把 Hugging Face 上的 Qwen3-ASR（或它的微調版，例如日文版 neosophie/Qwen3-ASR-1.7B-JA）
# 轉成這個擴充功能能用的瀏覽器版（layout v2 ONNX：q4 + q4f16），再上傳到你自己的 Hugging Face 帳號。
#
# 用法：tools/convert-qwen3-asr.sh <來源模型> <上傳到的 repo> [工作目錄]
#   例：HF_TOKEN=hf_xxx tools/convert-qwen3-asr.sh neosophie/Qwen3-ASR-1.7B-JA 你的帳號/Qwen3-ASR-1.7B-JA-ONNX
#   上傳到的 repo 填 - 就只轉換、不上傳（結果在 <工作目錄>/qwen3-asr-onnx/output/v2）。
#
# 需要：Linux 或 macOS、Python 3.10 以上、git、約 16GB 記憶體、約 30GB 硬碟（1.7B 在 15GB 記憶體、30GB 硬碟實測轉得完），能連 huggingface.co、*.hf.co（模型檔的下載網域）、github.com、pypi.org。
# 上傳需要有寫入權限的 Hugging Face token（環境變數 HF_TOKEN，或先 `huggingface-cli login`）。
# 1.7B 在 4 核心 CPU 上大約 30～60 分鐘（大部分是下載和匯出 encoder / decoder）。
#
# 流程就是 Sokuji 專案發布 jiangzhuo9357/Qwen3-ASR-1.7B-ONNX 的步驟（kizuna-ai-lab/sokuji 的
# benchmark/qwen3-asr-webgpu），搬成一支腳本：
#   andrewleech/qwen3-asr-onnx 匯出 FP32 ONNX → 原生 fp16 encoder → layout v2（prefill 吃 input_embeds）
#   → 圖最佳化 → int8 嵌入表 → int4 decoder → q4f16 decoder → mel filters → prompt_config.json → 上傳
# 兩個 repo 都是執行時才 clone，這裡不附它們的程式碼。
set -eo pipefail

SRC=${1:?用法：$0 <來源模型> <上傳到的 repo 或 -> [工作目錄]}
DST=${2:?用法：$0 <來源模型> <上傳到的 repo 或 -> [工作目錄]}
W=$(mkdir -p "${3:-$PWD/qwen3-asr-convert}" && cd "${3:-$PWD/qwen3-asr-convert}" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
log() { printf '\n== %s\n' "$*"; }

cd "$W"
log "取得轉換工具"
[ -d qwen3-asr-onnx ] || git clone https://github.com/andrewleech/qwen3-asr-onnx
if [ ! -d sokuji ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/kizuna-ai-lab/sokuji
  (cd sokuji && git sparse-checkout set benchmark/qwen3-asr-webgpu)
fi
P=$W/qwen3-asr-onnx
B=$W/sokuji/benchmark/qwen3-asr-webgpu
# prefill 只輸出最後一個位置的 logits（不然每段都要從 GPU 讀回約 120MB）
(cd "$P" && { git apply --reverse --check "$B/qwen3-asr-onnx-last-token-logits.patch" 2>/dev/null \
  || git apply "$B/qwen3-asr-onnx-last-token-logits.patch"; })
# 匯出程式把 prompt 裡的「system」「user」寫成 9125、882（Llama 3 的 id，在 Qwen 詞表是「 Current」「 time」），
# 改成 Qwen 的 8948、872，prompt_config.json 的 prefix_ids 才會跟模型訓練時一樣
python3 - "$P/src/prompt.py" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
t = re.sub(r"^(\s+)9125,", r"\g<1>8948,  # system（原本誤用 Llama 3 的 9125）", s, count=1, flags=re.M)
t = re.sub(r"^(\s+)882,", r"\g<1>872,  # user（原本誤用 Llama 3 的 882）", t, count=1, flags=re.M)
if t != s: open(p, "w").write(t); print("修正 src/prompt.py 的 system / user token")
PY

log "安裝 Python 套件（第一次比較久）"
[ -x "$P/.venv/bin/python" ] || python3 -m venv "$P/.venv"
PY=$P/.venv/bin/python
"$PY" -m pip -q install --upgrade pip
# PyTorch 的 CPU 版比較小；連不到 download.pytorch.org 時退回 PyPI 的預設版本
"$PY" -m pip -q install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || "$PY" -m pip -q install torch
# 版本跟 Sokuji 發布 jiangzhuo9357/Qwen3-ASR-*-ONNX 時一樣（較新的 onnx / onnxruntime 會讓 q4f16 的圖整理失敗）
"$PY" -m pip -q install "transformers==4.57.6" "accelerate==1.12.0" "onnx==1.22.*" "onnxruntime==1.29.*" onnxscript soundfile librosa \
  huggingface_hub onnxconverter-common
# qwen-asr 的相依套件裡有 nagisa（日文斷詞，要編譯、新版 Python 沒有現成套件）和 gradio 等，轉換都用不到：
# 不裝相依套件，nagisa / soynlp 只在 import 時需要（forced aligner 才會用），裝不起來就放空模組佔位
"$PY" -m pip -q install --no-deps qwen-asr
SP=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
for m in nagisa soynlp; do
  "$PY" -c "import $m" 2>/dev/null || echo "# 轉換用不到（只有 forced aligner 會用），佔位讓 qwen_asr 能 import" > "$SP/$m.py"
done
export QWEN3_ASR_ONNX_DIR=$P PY

V1=$P/output/v1
V2=$P/output/v2
cd "$P"
# 記憶體 / 硬碟很緊時（15GB / 30GB）的做法，1.7B 實測過：
# - export.py 不帶 --skip-encoder 會連 v1 的 decoder 一起匯出（v2 會重新匯出，用不到），
#   而且匯出完會 onnx.load 約 7GB 的外部資料，在 15GB 的機器上 OOM；所以 v1 只匯出 encoder，
#   嵌入表 / tokenizer / config 由 export-v1-meta.py 另外產生。
# - export_v2 用空的 --src（不複製），結束後把 v1 的檔案「搬」進 v2，省約 2.5GB。
# - Hugging Face 快取（約 4GB）在 export_v2 之後就沒用了，刪掉。
log "1/8 匯出 FP32 encoder 與中繼檔（會先從 Hugging Face 下載 $SRC）"
[ -f "$V1/encoder.onnx" ] || "$PY" export.py --model "$SRC" --output "$V1" --skip-decoder
[ -f "$V1/embed_tokens.bin" ] || "$PY" "$HERE/export-v1-meta.py" "$SRC" "$V1"
log "2/8 原生 fp16 encoder（給有 shader-f16 的顯示卡）"
[ -f "$V1/encoder.fp16.onnx" ] || "$PY" export_encoder_native_fp16.py --model "$SRC" --output "$V1/encoder.fp16.onnx" --verify
log "3/8 layout v2：prefill 改吃 input_embeds，模型裡不放嵌入表"
EMPTY=$(mktemp -d)
"$PY" "$B/export_v2/export_v2.py" --model "$SRC" --src "$EMPTY" --out "$V2"
rmdir "$EMPTY"
mv "$V1"/* "$V2"/ && rmdir "$V1"
rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${SRC//\//--}"
log "4/8 圖最佳化（RMSNorm / encoder 融合）"
"$PY" optimize_graphs.py --input "$V2"
log "5/8 int8 嵌入表"
"$PY" "$B/export_v2/embed_int8.py" "$V2"
rm -f "$V2/embed_tokens.bin"
log "6/8 int4 decoder"
# quantize_v2.sh 在同一個 process 量化兩個 decoder，第二個載入時會 OOM（15GB）：分兩個 process 跑，
# 再做它剩下的 share_weights（兩個 decoder 共用一份權重檔）。FP32 decoder 量化完就刪（各約 7GB）。
for d in decoder_init decoder_step; do
  "$PY" quantize_nbits.py --input "$V2" --output "$V2" --bits 4 --block-size 32 --accuracy-level 4 --decoders $d 2>&1 \
    | grep -v 'Progress:' | tail -3
done
rm -f "$V2"/decoder_init.onnx* "$V2"/decoder_step.onnx*
"$PY" share_weights.py "$V2" --suffix int4 --verify 2>&1 | tail -12
log "7/8 q4f16 decoder"
bash "$B/export_v2/q4f16_v2.sh" "$V2"
log "8/8 mel filters、vocab.json / added_tokens.json、prompt_config.json"
"$PY" "$B/mkfilters.py"
cp "$B/www/mel_filters.json" "$V2/"
# 擴充功能讀 vocab.json + added_tokens.json（不讀 11MB 的 tokenizer.json），匯出程式只留了 tokenizer.json，從它產生
"$PY" - "$V2" <<'PY'
import json, os, sys
d = sys.argv[1]
t = json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8"))
json.dump(t["model"]["vocab"], open(os.path.join(d, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
json.dump({a["content"]: a["id"] for a in t["added_tokens"]}, open(os.path.join(d, "added_tokens.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("vocab.json", len(t["model"]["vocab"]), "added_tokens.json", len(t["added_tokens"]))
PY
"$PY" "$B/export_v2/make_prompt_config.py" "$V2" int8 "$([ "$DST" = - ] && echo "$SRC" || echo "$DST")"

ls -l "$V2"
if [ "$DST" = - ]; then
  log "完成（沒有上傳）：$V2"
  exit 0
fi

log "上傳到 https://huggingface.co/$DST"
CARD=$W/README-$(echo "$DST" | tr / _).md
sed -e "s|{{SRC}}|$SRC|g" -e "s|{{DST}}|$DST|g" "$HERE/qwen3-asr-model-card.md" > "$CARD"
"$PY" "$B/upload_hf.py" --repo "$DST" --dir "$V2" --readme "$CARD"
log "完成。擴充功能的「辨識模型」選「自訂模型」，填 $DST"
