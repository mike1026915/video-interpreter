// 把分頁音訊（混成單聲道）一批批送回主執行緒
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(4096);
    this.len = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const n = input[0].length;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < input.length; c++) s += input[c][i];
        this.buf[this.len++] = s / input.length;
        if (this.len === this.buf.length) {
          this.port.postMessage(this.buf.slice(0));
          this.len = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('capture', CaptureProcessor);
