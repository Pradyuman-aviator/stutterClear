// Follow a pasted script from live captions. Words match if they are equal, if one is a prefix
// of the other, or if their edit similarity is at least 0.75. The pointer never jumps more than
// 30 words.

export function normaliseScript(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .split(/\s+/)
    .map((w) => w.replace(/^[^a-z0-9']+|[^a-z0-9']+$/gi, ""))
    .filter(Boolean)
}

function similar(a: string, b: string) {
  if (a === b || a.startsWith(b) || b.startsWith(a)) return true
  const n = a.length
  const m = b.length
  if (!n || !m) return false
  const row = Array.from({ length: m + 1 }, (_, i) => i)
  for (let i = 1; i <= n; i++) {
    let prev = row[0]
    row[0] = i
    for (let j = 1; j <= m; j++) {
      const cur = row[j]
      row[j] = a[i - 1] === b[j - 1] ? prev : 1 + Math.min(prev, row[j], row[j - 1])
      prev = cur
    }
  }
  return 1 - row[m] / Math.max(n, m) >= 0.75
}

export function followScript(script: string[], spoken: string[], pointer: number) {
  const from = Math.max(0, pointer - 5)
  const to = Math.min(script.length, pointer + 30)
  const heard = spoken.slice(-6)
  if (!heard.length) return pointer
  let bestAt = pointer
  let best = 0
  for (let start = from; start < to; start++) {
    let matched = 0
    let j = start
    for (const word of heard) {
      while (j < to && !similar(script[j], word)) j++
      if (j >= to) break
      matched++
      j++
    }
    if (matched > best) {
      best = matched
      bestAt = j
    }
  }
  if (best === 0) return pointer
  return Math.min(bestAt, pointer + 30)
}
