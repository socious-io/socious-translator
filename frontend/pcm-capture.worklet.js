// Emits 20 ms Int16 PCM frames at 16 kHz via port.postMessage(ArrayBuffer)
// IMPORTANT: We also write SILENCE to the node's output so the engine keeps pulling us.
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    this.inRate = sampleRate; // context rate (often 48000)
    this.outRate = opts?.processorOptions?.downsampleTo || 16000;
    this.frameMs = opts?.processorOptions?.frameMs || 20;
    this.samplesPerFrame = Math.round(this.outRate * this.frameMs / 1000); // e.g., 320
    this._buf = new Float32Array(0);
    this._zero = new Float32Array(128); // one render quantum size per channel
  }
  static get parameterDescriptors() { return []; }

  _downsample(f32) {
    if (this.inRate === this.outRate) return f32;
    const ratio = this.inRate / this.outRate;
    const outLen = Math.floor(f32.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) out[i] = f32[Math.floor(i * ratio)];
    return out;
  }
  _f32_to_i16(f32) {
    const i16 = new Int16Array(f32.length);
    for (let i=0;i<f32.length;i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return i16;
  }

  process(inputs, outputs /*, parameters */) {
    const ch0 = inputs[0]?.[0];
    const out = outputs[0];

    // --- ensure the node produces output (silence), so the graph keeps pulling us
    if (out && out[0]) out[0].set(this._zero);

    if (!ch0) return true;

    // accumulate input
    const concat = new Float32Array(this._buf.length + ch0.length);
    concat.set(this._buf, 0);
    concat.set(ch0, this._buf.length);

    // downsample to outRate
    const down = this._downsample(concat);
    const frames = Math.floor(down.length / this.samplesPerFrame);

    // post 20 ms frames as Int16
    for (let f=0; f<frames; f++) {
      const start = f * this.samplesPerFrame;
      const end   = start + this.samplesPerFrame;
      const slice = down.subarray(start, end);
      const i16 = this._f32_to_i16(slice);
      this.port.postMessage(i16.buffer, [i16.buffer]);
    }

    // keep remainder in our ring
    const consumed = frames * this.samplesPerFrame;
    const keepFromInputIdx = Math.floor(consumed * (this.inRate / this.outRate));
    this._buf = concat.subarray(keepFromInputIdx);
    return true;
  }
}
registerProcessor("pcm-capture", PcmCaptureProcessor);
