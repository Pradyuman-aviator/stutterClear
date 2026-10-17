// Live from the browser's microphone: capture at 16 kHz with an AudioWorklet and stream 20 ms frames
// of float32 audio to the server (/ws/mic), which runs the same live pipeline as a local session.
// Used when the app is hosted (the server has no mic) or when "This browser" is picked as the mic.

import type { PlaybackInfo, Summary } from "@/lib/api"

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) this.port.postMessage(ch.slice(0))
    return true
  }
}
registerProcessor("tap", Tap)
`

export class BrowserMic {
  private ws: WebSocket | null = null
  private ctx: AudioContext | null = null
  private stream: MediaStream | null = null
  private pending: Float32Array[] = []
  private pendingLen = 0
  private stopping = false

  // onLost: the connection dropped mid-session (network, server restart), not a normal stop.
  async start(settings: { mode: string; detector: string }, onLost?: () => void): Promise<{ detector: string }> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      // Raw voice: the stutter detector was trained on unprocessed audio.
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    })
    const proto = location.protocol === "https:" ? "wss" : "ws"
    const ws = new WebSocket(`${proto}://${location.host}/ws/mic`)
    ws.binaryType = "arraybuffer"
    this.ws = ws
    const ready = new Promise<{ detector: string }>((resolve, reject) => {
      ws.onopen = () => ws.send(JSON.stringify(settings))
      ws.onmessage = (m) => resolve(JSON.parse(m.data))
      ws.onclose = (e) => reject(new Error(e.reason || "The live connection closed."))
    })
    const info = await ready
    ws.onmessage = null
    ws.onclose = () => {
      if (!this.stopping) onLost?.()
    }

    this.ctx = new AudioContext({ sampleRate: 16000 })
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }))
    await this.ctx.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)
    const tap = new AudioWorkletNode(this.ctx, "tap")
    tap.port.onmessage = (e) => this.push(e.data as Float32Array)
    this.ctx.createMediaStreamSource(this.stream).connect(tap)
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
  async stop(): Promise<{ summary: Summary | null; playback?: PlaybackInfo } | null> {
    this.stopping = true
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.ctx?.close().catch(() => undefined)
    const ws = this.ws
    this.ws = null
    this.ctx = null
    this.stream = null
    if (!ws || ws.readyState !== WebSocket.OPEN) return null
    return new Promise((resolve) => {
      const done = (v: { summary: Summary | null; playback?: PlaybackInfo } | null) => {
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
