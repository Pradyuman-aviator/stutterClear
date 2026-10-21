// Type-checks the film, one check at a time on this machine (several can be asked for together).
//   node scripts/typecheck.mjs [path-filter]
// With a filter (e.g. scenes/hook.tsx) only the errors in matching files are printed; the exit code
// is 1 when there are any.
import { execFileSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const LOCK = path.join(ROOT, ".locks", "tsc")
const filter = process.argv[2]?.replace(/\\/g, "/")

fs.mkdirSync(path.dirname(LOCK), { recursive: true })
for (;;) {
  try {
    fs.mkdirSync(LOCK)
    break
  } catch {
    try {
      if (Date.now() - fs.statSync(LOCK).mtimeMs > 5 * 60 * 1000) fs.rmSync(LOCK, { recursive: true, force: true })
    } catch {
      // released meanwhile
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
}
let out = ""
try {
  execFileSync(process.execPath, [path.join(ROOT, "node_modules", "typescript", "bin", "tsc"), "--noEmit", "-p", ROOT], { cwd: ROOT, encoding: "utf8" })
} catch (e) {
  out = `${e.stdout ?? ""}${e.stderr ?? ""}`
} finally {
  fs.rmSync(LOCK, { recursive: true, force: true })
}
const lines = out.split(/\r?\n/).filter((l) => l.trim() && (!filter || l.replace(/\\/g, "/").includes(filter)))
console.log(lines.length ? lines.join("\n") : `no type errors${filter ? ` in ${filter}` : ""}`)
process.exitCode = lines.some((l) => l.includes("error TS")) ? 1 : 0
