// Follow a pasted script from live captions. Words match if they are equal, if one is a prefix
// of the other (3 letters or more: a cut-off "w" would match half the script), or if their edit
// similarity is at least 0.75. Heard words that are not in the script (a filler, a caption error,
// an aside) are skipped. The pointer moves only on two matched words in order, at most MAX_GAP
// script words apart, and never more than 30 words.

const MAX_GAP = 3 // script words a caption may have missed between two matched words

function tokens(text: string) {
  return text
    .replace(/[“”]/g, '"')
    .split(/\s+/)
    .map((raw) => ({
      w: raw.toLowerCase().replace(/^[^a-z0-9']+|[^a-z0-9']+$/gi, ""),
      end: /[.!?]/.test(raw),
    }))
    .filter((t) => t.w)
}

export function normaliseScript(text: string): string[] {
  return tokens(text).map((t) => t.w)
}

// True for each normalised word that ended a sentence in the original script.
export function sentenceEnds(text: string): boolean[] {
  return tokens(text).map((t) => t.end)
}

function similar(a: string, b: string) {
  if (a === b) return true
  if (Math.min(a.length, b.length) >= 3 && (a.startsWith(b) || b.startsWith(a))) return true
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
  // An alignment starts on script[start]; each later heard word may skip up to MAX_GAP script words.
  for (let start = from; start < to; start++) {
    let matched = 0
    let j = start
    for (const word of heard) {
      const limit = Math.min(to, matched ? j + MAX_GAP + 1 : start + 1)
      let k = j
      while (k < limit && !similar(script[k], word)) k++
      if (k >= limit) continue // not here: skip the heard word, keep the place
      matched++
      j = k + 1
    }
    if (matched > best) {
      best = matched
      bestAt = j // just after the last matched word
    }
  }
  if (best < Math.min(2, heard.length)) return pointer
  return Math.min(bestAt, pointer + 30)
}
