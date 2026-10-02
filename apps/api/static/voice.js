// Hands-free control: wake by name, wake by double clap, spoken replies.
//
// ClapDetector and matchWakeWord are pure (unit-tested with Node in tests/js/);
// VoiceController wires them to the microphone, speech recognition and speech
// synthesis in the browser.

/**
 * Finds double claps in a stream of audio peak levels.
 *
 * A clap is a sharp spike that dies away within `decayMs`; speech and music stay
 * loud for longer, so they are rejected. Two claps `minGapMs`..`maxGapMs` apart
 * make a double clap.
 */
export class ClapDetector {
  constructor({
    minLevel = 0.3, // absolute floor for a spike (0..1 peak amplitude)
    ratio = 6, // ...and it must be this many times louder than background noise
    decayMs = 120,
    minGapMs = 120,
    maxGapMs = 750,
    cooldownMs = 1500,
  } = {}) {
    Object.assign(this, { minLevel, ratio, decayMs, minGapMs, maxGapMs, cooldownMs });
    this.floor = 0.02;
    this.prev = 0;
    this.candidate = null; // time a possible clap started
    this.pending = null; // time of a confirmed first clap
    this.cooldownUntil = -Infinity;
  }

  /** Feed one frame's peak level at time t (ms). Returns "clap", "double" or null. */
  process(level, t) {
    const threshold = Math.max(this.minLevel, this.floor * this.ratio);
    let event = null;
    if (this.candidate !== null) {
      if (level < threshold * 0.5) {
        event = this._confirm(this.candidate);
        this.candidate = null;
      } else if (t - this.candidate > this.decayMs) {
        this.candidate = null; // stayed loud: a voice or music, not a clap
      }
    } else if (level >= threshold && this.prev < threshold * 0.6 && t >= this.cooldownUntil) {
      this.candidate = t;
    }
    if (level < threshold) this.floor = this.floor * 0.98 + level * 0.02;
    this.prev = level;
    return event;
  }

  _confirm(t) {
    if (this.pending !== null) {
      const gap = t - this.pending;
      if (gap >= this.minGapMs && gap <= this.maxGapMs) {
        this.pending = null;
        this.cooldownUntil = t + this.cooldownMs;
        return "double";
      }
    }
    this.pending = t;
    return "clap";
  }
}

function editDistance(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return row[b.length];
}

const WORD = /[\p{L}\p{N}']+/gu;

/**
 * Looks for the wake word in a transcript. Speech recognition often mishears
 * names slightly, so words of 5+ letters may be one letter off.
 * Returns { command } (the text after the wake word, possibly "") or null.
 */
export function matchWakeWord(transcript, wakeWord) {
  const wake = (wakeWord.toLowerCase().match(WORD) || []);
  if (!wake.length) return null;
  const words = [...transcript.matchAll(WORD)];
  const similar = (heard, want) =>
    heard === want || (want.length >= 5 && editDistance(heard, want) <= 1);
  for (let i = 0; i + wake.length <= words.length; i++) {
    if (wake.every((w, j) => similar(words[i + j][0].toLowerCase(), w))) {
      const last = words[i + wake.length - 1];
      const rest = transcript.slice(last.index + last[0].length);
      return { command: rest.replace(/^[\s,.!?:;—-]+/, "").trim() };
    }
  }
  return null;
}

/** Splits text into chunks short enough for speech synthesis to read reliably. */
export function speechChunks(text, max = 220) {
  const clean = text
    .replace(/```[\s\S]*?```/g, " code block omitted. ")
    .replace(/[*_#>`|]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const sentences = clean.match(/[^.!?]+[.!?]*/g) || [];
  const chunks = [];
  let current = "";
  for (const s of sentences) {
    if ((current + s).length > max && current) {
      chunks.push(current.trim());
      current = "";
    }
    current += s;
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

const PEAK_WORKLET = `
class PeakMeter extends AudioWorkletProcessor {
  constructor() { super(); this.blocks = 0; this.peak = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) { const v = Math.abs(ch[i]); if (v > this.peak) this.peak = v; }
    if (++this.blocks >= 4) { this.port.postMessage(this.peak); this.peak = 0; this.blocks = 0; }
    return true;
  }
}
registerProcessor("peak-meter", PeakMeter);
`;

/**
 * States: "idle" (no wake mode on), "armed" (waiting for name or clap),
 * "listening" (taking a command), "thinking", "speaking".
 */
export class VoiceController {
  constructor({ wakeWord, onState, onCommand, onLevel, onTranscript, onLog, onTypeInstead }) {
    this.wakeWord = wakeWord;
    this.cb = { onState, onCommand, onLevel, onTranscript, onLog, onTypeInstead };
    this.nameWake = false;
    this.clapWake = false;
    this.speakReplies = false;
    this.state = "idle";
    this.awaiting = false; // woken, waiting for the command
    this.recRunning = false;
    this.detector = new ClapDetector();
    this.audio = null;
  }

  get speechSupported() {
    return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  _setState(state) {
    this.state = state;
    this.cb.onState?.(state);
  }

  _restState() {
    this._setState(this.nameWake || this.clapWake ? "armed" : "idle");
  }

  async setNameWake(on) {
    if (on && !this.speechSupported) {
      this.cb.onLog?.("Voice wake needs Chrome, Edge or Safari.");
      return false;
    }
    this.nameWake = on;
    if (on) await this._audioContext(); // unlock audio for the chime on this click
    this._syncRecognition();
    if (this.state === "idle" || this.state === "armed") this._restState();
    return on;
  }

  async setClapWake(on) {
    if (on) {
      try {
        await this._startMic();
      } catch (e) {
        this.cb.onLog?.(`Microphone unavailable: ${e.message}`);
        return false;
      }
    } else {
      this._stopMic();
    }
    this.clapWake = on;
    if (this.state === "idle" || this.state === "armed") this._restState();
    return on;
  }

  setSpeakReplies(on) {
    this.speakReplies = on;
    if (!on) speechSynthesis.cancel();
  }

  /** Start taking a command now (tap-to-talk, or after the wake word/clap). */
  async wake() {
    if (this.state === "speaking") speechSynthesis.cancel();
    if (!this.speechSupported) {
      this.cb.onLog?.("Speech recognition isn't supported in this browser: type your command.");
      this.cb.onTypeInstead?.();
      return;
    }
    await this._chime();
    this.awaiting = true;
    this._setState("listening");
    clearTimeout(this.awaitTimer);
    this.awaitTimer = setTimeout(() => {
      if (this.awaiting) {
        this.awaiting = false;
        this.cb.onTranscript?.("");
        this._syncRecognition();
        this._restState();
      }
    }, 8000);
    this._syncRecognition();
  }

  /** The app calls this when it starts working on a command typed or spoken. */
  busy() {
    this.awaiting = false;
    clearTimeout(this.awaitTimer);
    this._setState("thinking");
    this._syncRecognition();
  }

  /** The app calls this with the final reply text. */
  async done(replyText) {
    if (this.speakReplies && replyText) await this._speak(replyText);
    this._restState();
    this._syncRecognition();
  }

  // ---- speech recognition -------------------------------------------------

  _wantRecognition() {
    return (this.nameWake || this.awaiting) && this.state !== "thinking" && this.state !== "speaking";
  }

  _syncRecognition() {
    const want = this._wantRecognition();
    if (want && !this.recRunning) this._startRecognition();
    if (!want && this.recRunning) this.rec.abort();
  }

  _startRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!this.rec) {
      const r = new SR();
      r.continuous = true;
      r.interimResults = true;
      r.lang = navigator.language || "en-US";
      r.onresult = (e) => this._onResult(e);
      r.onerror = (e) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
          this.nameWake = false;
          this.awaiting = false;
          this.cb.onLog?.("Microphone or speech recognition permission was denied.");
          this._restState();
        } else if (e.error !== "no-speech" && e.error !== "aborted") {
          this.cb.onLog?.(`Speech recognition: ${e.error}`);
        }
      };
      // Browsers end recognition after silence; keep it going while wanted.
      r.onend = () => {
        this.recRunning = false;
        if (this._wantRecognition()) setTimeout(() => this._wantRecognition() && this._startRecognition(), 300);
      };
      this.rec = r;
    }
    try {
      this.rec.start();
      this.recRunning = true;
    } catch {
      // already started
    }
  }

  _onResult(e) {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const text = e.results[i][0].transcript;
      if (!e.results[i].isFinal) {
        interim += text;
        continue;
      }
      if (this.awaiting) {
        if (text.trim()) this._dispatch(text.trim());
        return;
      }
      if (this.nameWake) {
        const hit = matchWakeWord(text, this.wakeWord);
        if (hit && hit.command.length > 1) {
          this._dispatch(hit.command);
          return;
        }
        if (hit) {
          this.wake();
          return;
        }
      }
    }
    if (this.awaiting || (this.nameWake && interim && matchWakeWord(interim, this.wakeWord))) {
      this.cb.onTranscript?.(interim);
    }
  }

  _dispatch(command) {
    this.cb.onTranscript?.("");
    this.busy();
    this.cb.onCommand?.(command);
  }

  // ---- microphone + clap detection --------------------------------------

  async _audioContext() {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    return this.ctx;
  }

  async _startMic() {
    if (this.audio) return;
    const ctx = await this._audioContext();
    // Raw signal: echo cancellation and noise suppression flatten clap transients.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    if (!this.workletLoaded) {
      const url = URL.createObjectURL(new Blob([PEAK_WORKLET], { type: "text/javascript" }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      this.workletLoaded = true;
    }
    const source = ctx.createMediaStreamSource(stream);
    const meter = new AudioWorkletNode(ctx, "peak-meter");
    meter.port.onmessage = ({ data: level }) => {
      this.cb.onLevel?.(level);
      if (this.detector.process(level, performance.now()) === "double") this._onDoubleClap();
    };
    source.connect(meter);
    this.audio = { stream, source, meter };
  }

  _stopMic() {
    if (!this.audio) return;
    this.audio.source.disconnect();
    this.audio.meter.port.onmessage = null;
    this.audio.stream.getTracks().forEach((t) => t.stop());
    this.audio = null;
    this.cb.onLevel?.(0);
  }

  _onDoubleClap() {
    if (!this.clapWake) return;
    if (this.state === "speaking") {
      speechSynthesis.cancel(); // clap to interrupt
      return;
    }
    if (this.state === "armed" || this.state === "idle") this.wake();
  }

  // ---- sound out ------------------------------------------------------------

  async _chime() {
    try {
      const ctx = await this._audioContext();
      const t = ctx.currentTime;
      for (const [i, freq] of [880, 1320].entries()) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.0001, t + i * 0.09);
        gain.gain.exponentialRampToValueAtTime(0.15, t + i * 0.09 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.09 + 0.12);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t + i * 0.09);
        osc.stop(t + i * 0.09 + 0.13);
      }
    } catch {
      // no audio output: the visual state change is enough
    }
  }

  _voice() {
    const voices = speechSynthesis.getVoices();
    return (
      voices.find((v) => /daniel/i.test(v.name) && v.lang.startsWith("en")) ||
      voices.find((v) => v.lang === "en-GB") ||
      voices.find((v) => v.lang.startsWith((navigator.language || "en").slice(0, 2))) ||
      null
    );
  }

  _speak(text) {
    const chunks = speechChunks(text);
    if (!chunks.length) return Promise.resolve();
    this._setState("speaking");
    this._syncRecognition(); // don't let the assistant hear itself
    return new Promise((resolve) => {
      const voice = this._voice();
      chunks.forEach((chunk, i) => {
        const u = new SpeechSynthesisUtterance(chunk);
        if (voice) u.voice = voice;
        u.rate = 1.03;
        if (i === chunks.length - 1) {
          u.onend = resolve;
          u.onerror = resolve;
        }
        speechSynthesis.speak(u);
      });
    });
  }
}
