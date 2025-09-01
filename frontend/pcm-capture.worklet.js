// Emits 20 ms Int16 PCM frames at 16 kHz via port.postMessage(ArrayBuffer)
// Also writes silence to output so the engine keeps pulling us.
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    var p = (options && options.processorOptions) ? options.processorOptions : {};
    this.inRate = sampleRate;                   // context rate (often 48000)
    this.outRate = p.downsampleTo || 16000;
    this.frameMs = p.frameMs || 20;
    this.samplesPerFrame = Math.round(this.outRate * this.frameMs / 1000); // e.g., 320
    this._buf = new Float32Array(0);
    this._zero = new Float32Array(128);         // one render quantum per channel
  }

  static get parameterDescriptors() { return []; }

  _downsample(f32) {
    if (this.inRate === this.outRate) return f32;
    var ratio = this.inRate / this.outRate;
    var outLen = Math.floor(f32.length / ratio);
    var out = new Float32Array(outLen);
    for (var i = 0; i < outLen; i++) out[i] = f32[Math.floor(i * ratio)];
    return out;
  }

  _f32_to_i16(f32) {
    var i16 = new Int16Array(f32.length);
    for (var i = 0; i < f32.length; i++) {
      var s = f32[i];
      if (s > 1) s = 1; else if (s < -1) s = -1;
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return i16;
  }

  process(inputs, outputs) {
    var ch0 = (inputs[0] && inputs[0][0]) ? inputs[0][0] : null;
    var out = outputs[0];

    // keep the pull alive with silence
    if (out && out[0]) out[0].set(this._zero);

    if (!ch0) return true;

    // accumulate input
    var concat = new Float32Array(this._buf.length + ch0.length);
    concat.set(this._buf, 0);
    concat.set(ch0, this._buf.length);

    // downsample
    var down = this._downsample(concat);
    var frames = Math.floor(down.length / this.samplesPerFrame);

    // post 20 ms frames
    for (var f = 0; f < frames; f++) {
      var start = f * this.samplesPerFrame;
      var end   = start + this.samplesPerFrame;
      var slice = down.subarray(start, end);
      var i16 = this._f32_to_i16(slice);
      // transfer the underlying buffer to avoid copies
      this.port.postMessage(i16.buffer, [i16.buffer]);
    }

    // keep remainder in buffer (convert back to input index)
    var consumed = frames * this.samplesPerFrame;
    var ratio = this.inRate / this.outRate;
    var keepFromInputIdx = Math.floor(consumed * ratio);
    this._buf = concat.subarray(keepFromInputIdx);
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
