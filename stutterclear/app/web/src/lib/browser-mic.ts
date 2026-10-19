// Live from the browser's microphone: capture at 16 kHz with an AudioWorklet and stream 20 ms frames
// of float32 audio to the server (/ws/mic), which runs the same live pipeline as a local session.
// Used when the app is hosted (the server has no mic) or when "This browser" is picked as the mic.

import { PAGE_ID, type PlaybackInfo, type Summary } from "@/lib/api"

// Firefox only connects a mic to an AudioContext at the device's own rate (usually 48 kHz). Then the
// worklet resamples to 16 kHz itself: a windowed-sinc low-pass below 8 kHz (so nothing folds back
// into the band the detector hears), then linear interpolation at the 16 kHz instants.
export const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() {
    super()
    this.ratio = sampleRate / 16000
    if (this.ratio === 1) return
    const n = 63, fc = 7200 / sampleRate
    this.h = new Float32Array(n)
    let sum = 0
    for (let i = 0; i < n; i++) {
      const m = i - (n - 1) / 2
      const sinc = m === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * m) / (Math.PI * m)
      const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1))
      this.h[i] = sinc * w
      sum += this.h[i]
    }
    for (let i = 0; i < n; i++) this.h[i] /= sum
    this.hist = new Float32Array(n)
    this.pos = 0
    this.count = 0
    this.t = 0
    this.prev = 0
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (!ch) return true
    if (this.ratio === 1) {
      this.port.postMessage(ch.slice(0))
      return true
    }
    const n = this.h.length, out = []
    for (let i = 0; i < ch.length; i++) {
      this.hist[this.pos] = ch[i]
      this.pos = (this.pos + 1) % n
      let y = 0
      for (let k = 0; k < n; k++) y += this.h[k] * this.hist[(this.pos + k) % n]
      while (this.t <= this.count) {
        out.push(this.prev + (y - this.prev) * (this.t - this.count + 1))
        this.t += this.ratio
      }
      this.prev = y
      this.count++
    }
    if (out.length) this.port.postMessage(Float32Array.from(out))
    return true
  }
}
registerProcessor("tap", Tap)
`

// The microphone, raw (the stutter detector was trained on unprocessed audio), with errors a person
// can act on instead of the browser's own wording.
export async function openMic(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("The microphone only works on a secure page: open this site with https:// (or on localhost).")
  }
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    })
  } catch (e) {
    const name = (e as DOMException).name
    if (name === "NotAllowedError" || name === "SecurityError")
      throw new Error("Microphone access is blocked. Allow it for this site (the icon left of the address), then try again.")
    if (name === "NotFoundError" || name === "OverconstrainedError")
      throw new Error("No microphone found. Plug one in, or pick one in your system's sound settings.")
    if (name === "NotReadableError" || name === "AbortError")
      throw new Error("The microphone is busy in another app (a call?). Free it there and try again.")
    throw e
  }
}

// How a session ended: the summary and listen-back, and why if the server ended it itself.
export type MicEnd = { summary: Summary | null; playback?: PlaybackInfo; ended?: string | null }

export class BrowserMic {
  private ws: WebSocket | null = null
  private ctx: AudioContext | null = null
  private stream: MediaStream | null = null
  private pending: Float32Array[] = []
  private pendingLen = 0
  private stopping = false
  private aid: DelayNode | null = null

  // onEnd: the server ended the session itself (time limit, no audio) and sent its result, or the
  // connection dropped mid-session (network, server restart) and there is none (null). Not called
  // after stop().
  async start(
    settings: { mode: string; detector: string; aidMs?: number },
    onEnd?: (r: MicEnd | null) => void,
  ): Promise<{ detector: string }> {
    this.stream = await openMic()
    const proto = location.protocol === "https:" ? "wss" : "ws"
    const ws = new WebSocket(`${proto}://${location.host}/ws/mic`)
    ws.binaryType = "arraybuffer"
    this.ws = ws
    const ready = new Promise<{ detector: string }>((resolve, reject) => {
      ws.onopen = () => ws.send(JSON.stringify({ ...settings, page: PAGE_ID }))
      ws.onmessage = (m) => resolve(JSON.parse(m.data))
      ws.onclose = (e) => reject(new Error(e.reason || "The live connection closed."))
    })
    const info = await ready
    ws.onmessage = (m) => {
      if (this.stopping) return
      this.stopping = true
      onEnd?.(JSON.parse(m.data))
    }
    ws.onclose = () => {
      if (!this.stopping) onEnd?.(null)
    }

    let source: MediaStreamAudioSourceNode
    try {
      this.ctx = new AudioContext({ sampleRate: 16000 })
      source = this.ctx.createMediaStreamSource(this.stream)
    } catch {
      await this.ctx?.close().catch(() => undefined) // Firefox: the device's own rate; the worklet resamples
      this.ctx = new AudioContext()
      source = this.ctx.createMediaStreamSource(this.stream)
    }
    await this.ctx.resume().catch(() => undefined) // some browsers start it suspended
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }))
    await this.ctx.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)
    const tap = new AudioWorkletNode(this.ctx, "tap")
    tap.port.onmessage = (e) => this.push(e.data as Float32Array)
    source.connect(tap)
    // Fluency aid: the speaker hears their own voice a little late. The detector still gets the raw mic.
    const aidMs = settings.aidMs ?? 0
    if (aidMs > 0) {
      this.aid = this.ctx.createDelay(0.5)
      this.aid.delayTime.value = aidMs / 1000
      source.connect(this.aid)
      this.aid.connect(this.ctx.destination)
    }
    return info
  }

  private push(frame: Float32Array) {
    this.pending.push(frame)
    this.pendingLen += frame.length
    if (this.pendingLen < 320) return // send 20 ms at a time
    const out = new Float32Array(this.pendingLen)
    let o = 0
    for (const f of this.pending) {
      out.set(f, o)
      o += f.length
    }
    this.pending = []
    this.pendingLen = 0
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(out.buffer)
  }

  // Ends the session: the server replies with the same summary and playback info as a local stop.
  async stop(): Promise<MicEnd | null> {
    this.stopping = true
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.ctx?.close().catch(() => undefined)
    const ws = this.ws
    this.ws = null
    this.ctx = null
    this.stream = null
    if (!ws || ws.readyState !== WebSocket.OPEN) return null
    return new Promise((resolve) => {
      const done = (v: MicEnd | null) => {
        resolve(v)
        ws.close()
      }
      ws.onmessage = (m) => done(JSON.parse(m.data))
      ws.onclose = () => resolve(null)
      window.setTimeout(() => done(null), 5000)
      ws.send("stop")
    })
  }
}
