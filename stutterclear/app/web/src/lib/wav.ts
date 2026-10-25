// Browsers record compressed audio (WebM/Opus, MP4/AAC) and the server has no decoder for those, so
// the page decodes any recording or uploaded file itself and sends plain 16 kHz mono WAV.

// `seconds` is the whole recording's length; the WAV holds at most its first `maxSeconds`.
export async function toWav16k(blob: Blob, maxSeconds = Infinity): Promise<{ wav: Blob; seconds: number }> {
  const ctx = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer())
  } catch {
    throw new Error("That file could not be read as audio.")
  } finally {
    ctx.close().catch(() => undefined)
  }
  const frames = Math.max(1, Math.min(Math.ceil(decoded.duration * 16000), Math.floor(maxSeconds * 16000)))
  const off = new OfflineAudioContext(1, frames, 16000) // one channel: stereo is mixed down
  const src = off.createBufferSource()
  src.buffer = decoded
  src.connect(off.destination)
  src.start()
  const rendered = await off.startRendering()
  return { wav: encodeWav(rendered.getChannelData(0), 16000), seconds: decoded.duration }
}

function encodeWav(pcm: Float32Array, sr: number): Blob {
  const buf = new ArrayBuffer(44 + pcm.length * 2)
  const v = new DataView(buf)
  const text = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
  text(0, "RIFF")
  v.setUint32(4, 36 + pcm.length * 2, true)
  text(8, "WAVE")
  text(12, "fmt ")
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true) // PCM
  v.setUint16(22, 1, true) // mono
  v.setUint32(24, sr, true)
  v.setUint32(28, sr * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  text(36, "data")
  v.setUint32(40, pcm.length * 2, true)
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buf], { type: "audio/wav" })
}
