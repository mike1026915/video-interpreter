---
license: apache-2.0
base_model: {{SRC}}
pipeline_tag: automatic-speech-recognition
library_name: onnxruntime
tags:
  - qwen3-asr
  - onnx
  - webgpu
---

# {{DST}}

[{{SRC}}](https://huggingface.co/{{SRC}}) converted to the browser layout (layout v2) used by
[`jiangzhuo9357/Qwen3-ASR-1.7B-ONNX`](https://huggingface.co/jiangzhuo9357/Qwen3-ASR-1.7B-ONNX):
encoder + `decoder_init` (prefill on `input_embeds`) + `decoder_step`, int8 embedding table,
q4 and q4f16 variants. Every constant the runtime needs is in `prompt_config.json`; the
interface is the same as the base layout-v2 model card.

Converted with `tools/convert-qwen3-asr.sh`, which runs the export pipeline of
[andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx) and the layout-v2
scripts of [kizuna-ai-lab/sokuji](https://github.com/kizuna-ai-lab/sokuji)
(`benchmark/qwen3-asr-webgpu`). Weights and license follow the source model.
